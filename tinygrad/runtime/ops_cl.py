from __future__ import annotations
from typing import cast
import ctypes, functools, hashlib
from tinygrad.runtime.autogen import opencl as cl
from tinygrad.runtime.support import c
from tinygrad.helpers import to_char_p_p, from_mv, OSX, DEBUG, mv_address, suppress_finalizing
from tinygrad.renderer.cstyle import OpenCLRenderer, IntelRenderer
from tinygrad.device import BufferSpec, LRUAllocator, Compiled, Compiler, CompileError, CompilerSet
from tinygrad.dtype import ImageDType

CC_CB = c.CFUNCTYPE[None, [c.POINTER[ctypes.c_char], c.POINTER[None], cl.size_t, c.POINTER[None]]]
BP_CB = c.CFUNCTYPE[None, [cl.cl_program, c.POINTER[None]]]

# see test/external/external_osx_profiling.py to determine this ratio. it's in like GPU clocks or something
OSX_TIMING_RATIO = (125/3) if OSX else 1.0

cl_errors = {attr: k for k in dir(cl) if k.startswith("CL_") and isinstance(attr:=getattr(cl, k), int) and attr <= 0}
def check(status):
  if status != 0: raise RuntimeError(f"OpenCL Error {status}: {cl_errors.get(status, 'Unknown error')}")
def checked(ret, status): return (check(status.value), ret)[1]

class CLCompiler(Compiler):
  def __init__(self, dev:CLDevice, compile_key:str):
    self.dev = dev
    super().__init__(f"compile_cl_{compile_key}")
  def compile(self, src:str) -> bytes:
    program = checked(cl.clCreateProgramWithSource(self.dev.context, 1, to_char_p_p([src.encode()]), None, status := ctypes.c_int32()), status)
    build_opts = b"-cl-fast-relaxed-math" if ("Mali" in self.dev.device_name or "Adreno" in self.dev.device_name or "FD" in self.dev.device_name) else None
    build_status: int = cl.clBuildProgram(program, 1, self.dev.device_id, build_opts, BP_CB(), None)
    if build_status != 0:
      cl.clGetProgramBuildInfo(program, self.dev.device_id, cl.CL_PROGRAM_BUILD_LOG, 0, None, log_size := ctypes.c_size_t())
      cl.clGetProgramBuildInfo(program, self.dev.device_id, cl.CL_PROGRAM_BUILD_LOG,
                               log_size.value, mstr := ctypes.create_string_buffer(log_size.value), None)
      raise CompileError(f"OpenCL Compile Error\n\n{mstr.value.decode()}")
    check(cl.clGetProgramInfo(program, cl.CL_PROGRAM_BINARY_SIZES, ctypes.sizeof(ctypes.c_size_t), binary_sizes := (ctypes.c_size_t * 1)(), None))
    check(cl.clGetProgramInfo(program, cl.CL_PROGRAM_BINARIES, ctypes.sizeof(ctypes.c_void_p),
                              (ctypes.c_void_p * 1)(ctypes.addressof(binary := ctypes.create_string_buffer(binary_sizes[0]))), None))
    check(cl.clReleaseProgram(program))
    return bytes(binary)

class CLProgram:
  def __init__(self, device:CLDevice, name:str, lib:bytes, arg_dtypes=[], **kwargs):
    self.dev, self.name, self.lib, self.arg_dtypes = device, name, device.cl_compiler.compile_cached(lib.decode()), arg_dtypes
    self.program = checked(cl.clCreateProgramWithBinary(device.context, 1, device.device_id, (ctypes.c_size_t * 1)(len(self.lib)),
                                                        to_char_p_p([self.lib], ctypes.c_ubyte), binary_status := ctypes.c_int32(),
                                                        errcode_ret := ctypes.c_int32()), errcode_ret)
    check(binary_status.value)
    check(cl.clBuildProgram(self.program, 1, device.device_id, None, BP_CB(), None)) # NOTE: OSX requires this
    self.kernel = checked(cl.clCreateKernel(self.program, name.encode(), status := ctypes.c_int32()), status)

  def __del__(self):
    try: check(cl.clReleaseKernel(self.kernel))
    except (TypeError, AttributeError): pass
    try: check(cl.clReleaseProgram(self.program))
    except (TypeError, AttributeError): pass

  def __call__(self, *bufs:cl.cl_mem, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]|None=None, vals:tuple[int, ...]=(),
               wait=False, **kw) -> float|None:
    i = 0
    if not hasattr(self, '_prev_bufs'): self._prev_bufs = [None] * 16
    prev = self._prev_bufs
    for i,b in enumerate(bufs):
      if i < len(prev) and prev[i] is b: continue
      if i >= len(prev): prev.extend([None] * (i + 1 - len(prev)))
      prev[i] = b
      for real_i, dt in self.arg_dtypes[i]:
        if isinstance(dt, ImageDType):
          fmt = cl.cl_image_format(cl.CL_RGBA, {2:cl.CL_HALF_FLOAT, 4:cl.CL_FLOAT}[dt.itemsize])
          desc = cl.cl_image_desc(cl.CL_MEM_OBJECT_IMAGE2D, dt.shape[1], dt.shape[0], image_row_pitch=dt.pitch, buffer=b)
          img = checked(cl.clCreateImage(self.dev.context, cl.CL_MEM_READ_WRITE, fmt, desc, None, status:=ctypes.c_int32()), status)
          check(cl.clSetKernelArg(self.kernel, real_i, ctypes.sizeof(img), ctypes.byref(img)))
        else: check(cl.clSetKernelArg(self.kernel, real_i, ctypes.sizeof(b), ctypes.byref(b)))
    for i,v in enumerate(vals,start=i+1): check(cl.clSetKernelArg(self.kernel, i, 4, ctypes.byref(ctypes.c_int32(v))))
    if local_size is not None: global_size = cast(tuple[int,int,int], tuple(int(g*l) for g,l in zip(global_size, local_size)))
    cache_key = (global_size, local_size)
    if not hasattr(self, '_size_cache_key') or self._size_cache_key != cache_key:
      ndim = len(global_size)
      self._gs_arr = (ctypes.c_size_t * ndim)(*global_size)
      self._ls_arr = (ctypes.c_size_t * ndim)(*local_size) if local_size else None
      self._ndim = ndim
      self._size_cache_key = cache_key
    if wait and self.dev.is_rusticl:
      import time as _time
      check(cl.clEnqueueNDRangeKernel(self.dev.queue, self.kernel, self._ndim, None, self._gs_arr,
                                      self._ls_arr, 0, None, None))
      _t0 = _time.perf_counter()
      check(cl.clFinish(self.dev.queue))
      return _time.perf_counter() - _t0
    event = cl.cl_event() if wait else None
    check(cl.clEnqueueNDRangeKernel(self.dev.queue, self.kernel, self._ndim, None, self._gs_arr,
                                    self._ls_arr, 0, None, event))
    if not wait:
      self.dev._flush_counter = getattr(self.dev, '_flush_counter', 0) + 1
      if self.dev._flush_counter % 6 == 0: cl.clFlush(self.dev.queue)
    if wait:
      assert event is not None
      import time as _time
      _t0 = _time.perf_counter()
      check(cl.clWaitForEvents(1, event))
      _t1 = _time.perf_counter()
      start, end = ctypes.c_uint64(), ctypes.c_uint64()
      if cl.clGetEventProfilingInfo(event, cl.CL_PROFILING_COMMAND_START, 8, ctypes.byref(start), None) == 0 and \
         cl.clGetEventProfilingInfo(event, cl.CL_PROFILING_COMMAND_END, 8, ctypes.byref(end), None) == 0:
        return float(end.value-start.value) * OSX_TIMING_RATIO * 1e-9
      return _t1 - _t0
    return None

class CLAllocator(LRUAllocator['CLDevice']):
  def _alloc(self, size:int, options:BufferSpec) -> cl.cl_mem:
    flags, host_ptr = cl.CL_MEM_READ_WRITE, None
    if options is not None and options.external_ptr is not None:
      flags |= cl.CL_MEM_USE_HOST_PTR
      host_ptr = ctypes.c_void_p(options.external_ptr)
    return checked(cl.clCreateBuffer(self.dev.context, flags, size, host_ptr, status := ctypes.c_int32()), status)
  @suppress_finalizing
  def _free(self, opaque:cl.cl_mem, options:BufferSpec): check(cl.clReleaseMemObject(opaque))
  def _copyin(self, dest:cl.cl_mem, src:memoryview):
    if mv_address(src) % 16: src = memoryview(bytearray(src))
    check(cl.clEnqueueWriteBuffer(self.dev.queue, dest, False, 0, len(src)*src.itemsize, from_mv(src), 0, None, None))
    self.dev.pending_copyin.append(src)    # NOTE: these can't be freed until the GPU actually executes this command
  def _copyout(self, dest:memoryview, src:cl.cl_mem):
    check(cl.clEnqueueReadBuffer(self.dev.queue, src, False, 0, len(dest)*dest.itemsize, from_mv(dest), 0, None, None))
    self.dev.synchronize()

class CLDevice(Compiled):
  device_ids = None                 # this is global and only initted once
  def __init__(self, device:str=""):
    if CLDevice.device_ids is None:
      check(cl.clGetPlatformIDs(0, None, num_platforms := ctypes.c_uint32()))
      check(cl.clGetPlatformIDs(num_platforms.value, platform_ids := (cl.cl_platform_id * num_platforms.value)(), None))
      for device_type in [cl.CL_DEVICE_TYPE_GPU, cl.CL_DEVICE_TYPE_DEFAULT]:
        err = cl.clGetDeviceIDs(platform_ids[0], device_type, 0, None, num_devices := ctypes.c_uint32())
        if err == 0 and num_devices.value != 0: break
      if DEBUG >= 1: print(f"CLDevice: got {num_platforms.value} platforms and {num_devices.value} devices")
      CLDevice.device_ids = c.init_c_var((cl.cl_device_id * num_devices.value),
                                         lambda x: check(cl.clGetDeviceIDs(platform_ids[0], device_type, num_devices, x, None)))

    self.device_id = CLDevice.device_ids[0 if ":" not in device else int(device.split(":")[1])]
    self.device_name = (cl.clGetDeviceInfo(self.device_id, cl.CL_DEVICE_NAME, 256,
                                           buf:=ctypes.create_string_buffer(256), None), buf.value.decode())[1]
    self.driver_version = (cl.clGetDeviceInfo(self.device_id, cl.CL_DRIVER_VERSION, 256,
                                              buf:=ctypes.create_string_buffer(256), None), buf.value.decode())[1]
    if DEBUG >= 1: print(f"CLDevice: opening {self.device_name} with version {self.driver_version}")
    self.context = checked(cl.clCreateContext(None, 1, self.device_id, CC_CB(), None, status := ctypes.c_int32()), status)
    self.is_rusticl = "FD" in self.device_name or "Adreno" in self.device_name or "Mali" in self.device_name
    queue_flags = 0 if self.is_rusticl else cl.CL_QUEUE_PROFILING_ENABLE
    self.queue = checked(cl.clCreateCommandQueue(self.context, self.device_id, queue_flags, status), status)
    self.pending_copyin: list[memoryview] = []
    self.device_exts = (cl.clGetDeviceInfo(self.device_id, cl.CL_DEVICE_EXTENSIONS, 4096,
                                           ctypes.byref(buf := ctypes.create_string_buffer(4096)),
                                           ctypes.byref(total := ctypes.c_size_t())),
                                           ctypes.string_at(buf, size=total.value).decode())[1]

    renderer = IntelRenderer if "cl_intel_subgroup_matrix_multiply_accumulate" in self.device_exts else OpenCLRenderer
    self.cl_compiler = CLCompiler(self, f"{hashlib.md5(self.device_name.encode() + self.driver_version.encode()).hexdigest()}")
    super().__init__(device, CLAllocator(self), CompilerSet([(renderer, None)]), functools.partial(CLProgram, self))

  def synchronize(self):
    check(cl.clFinish(self.queue))
    self.pending_copyin.clear()
