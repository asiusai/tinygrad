from __future__ import annotations
from typing import cast
import ctypes, functools, hashlib, os
from tinygrad.runtime.autogen import opencl as cl
from tinygrad.runtime.support import c
from tinygrad.helpers import to_char_p_p, from_mv, OSX, DEBUG, mv_address, suppress_finalizing, getenv
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
    if getenv("CL_CUSTOM_R16") and name == "r_16_64_32_4_4_24_3_3":
      lib = b"""
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
static inline half4 gelu4(half4 x) {
  half4 one = (half4)(((half)(1.0f)), ((half)(1.0f)), ((half)(1.0f)), ((half)(1.0f)));
  return (one / (one + exp2((x + (((half)(0.044715f)) * x * x * x)) * ((half)(-2.302208198144325f))))) * x;
}
__kernel void r_16_64_32_4_4_24_3_3(__global half* data0_524288, __global half* data1_786432, __global half* data2_13824, __global half* data3_64) {
  int idx0 = get_global_id(0);
  int idx1 = get_global_id(1);
  int idx2 = get_global_id(2);
  half4 zero = (half4)(((half)(0.0f)), ((half)(0.0f)), ((half)(0.0f)), ((half)(0.0f)));
  half4 acc0 = zero, acc1 = zero, acc2 = zero, acc3 = zero;
  bool alu0 = (0 < idx0);
  bool alu1 = (0 < idx1);
  for (int Ridx0 = 0; Ridx0 < 24; Ridx0++) {
    int alu18 = ((idx0 << 3) + (idx1 << 9) + (Ridx0 << 15));
    int alu19 = ((idx2 * 864) + (Ridx0 * 9));
    half4 w0 = (half4)(*(data2_13824 + alu19), *(data2_13824 + alu19 + 216), *(data2_13824 + alu19 + 432), *(data2_13824 + alu19 + 648));
    half4 w1 = (half4)(*(data2_13824 + alu19 + 1), *(data2_13824 + alu19 + 217), *(data2_13824 + alu19 + 433), *(data2_13824 + alu19 + 649));
    half4 w2 = (half4)(*(data2_13824 + alu19 + 2), *(data2_13824 + alu19 + 218), *(data2_13824 + alu19 + 434), *(data2_13824 + alu19 + 650));
    half4 w3 = (half4)(*(data2_13824 + alu19 + 3), *(data2_13824 + alu19 + 219), *(data2_13824 + alu19 + 435), *(data2_13824 + alu19 + 651));
    half4 w4 = (half4)(*(data2_13824 + alu19 + 4), *(data2_13824 + alu19 + 220), *(data2_13824 + alu19 + 436), *(data2_13824 + alu19 + 652));
    half4 w5 = (half4)(*(data2_13824 + alu19 + 5), *(data2_13824 + alu19 + 221), *(data2_13824 + alu19 + 437), *(data2_13824 + alu19 + 653));
    half4 w6 = (half4)(*(data2_13824 + alu19 + 6), *(data2_13824 + alu19 + 222), *(data2_13824 + alu19 + 438), *(data2_13824 + alu19 + 654));
    half4 w7 = (half4)(*(data2_13824 + alu19 + 7), *(data2_13824 + alu19 + 223), *(data2_13824 + alu19 + 439), *(data2_13824 + alu19 + 655));
    half4 w8 = (half4)(*(data2_13824 + alu19 + 8), *(data2_13824 + alu19 + 224), *(data2_13824 + alu19 + 440), *(data2_13824 + alu19 + 656));
    half v00 = ((alu0 & alu1) ? *(data1_786432 + (alu18 - 257)) : ((half)(0.0f)));
    half v10 = (alu0 ? *(data1_786432 + (alu18 - 1)) : ((half)(0.0f)));
    half v20 = (alu0 ? *(data1_786432 + (alu18 + 255)) : ((half)(0.0f)));
    half4 row0a = (alu1 ? *((__global half4*)((data1_786432 + (alu18 - 256)))) : zero);
    half4 row0b = (alu1 ? *((__global half4*)((data1_786432 + (alu18 - 252)))) : zero);
    half4 row1a = *((__global half4*)((data1_786432 + alu18)));
    half4 row1b = *((__global half4*)((data1_786432 + (alu18 + 4))));
    half4 row2a = *((__global half4*)((data1_786432 + (alu18 + 256))));
    half4 row2b = *((__global half4*)((data1_786432 + (alu18 + 260))));
    acc0 = acc0 + (v00 * w0) + (row0a.x * w1) + (row0a.y * w2) + (v10 * w3) + (row1a.x * w4) + (row1a.y * w5) + (v20 * w6) + (row2a.x * w7) + (row2a.y * w8);
    acc1 = acc1 + (row0a.y * w0) + (row0a.z * w1) + (row0a.w * w2) + (row1a.y * w3) + (row1a.z * w4) + (row1a.w * w5) + (row2a.y * w6) + (row2a.z * w7) + (row2a.w * w8);
    acc2 = acc2 + (row0a.w * w0) + (row0b.x * w1) + (row0b.y * w2) + (row1a.w * w3) + (row1b.x * w4) + (row1b.y * w5) + (row2a.w * w6) + (row2b.x * w7) + (row2b.y * w8);
    acc3 = acc3 + (row0b.y * w0) + (row0b.z * w1) + (row0b.w * w2) + (row1b.y * w3) + (row1b.z * w4) + (row1b.w * w5) + (row2b.y * w6) + (row2b.z * w7) + (row2b.w * w8);
  }
  half4 bias = *((__global half4*)((data3_64 + (idx2 << 2))));
  int alu53 = ((idx0 << 2) + (idx1 << 7) + (idx2 << 15));
  *((__global half4*)((data0_524288 + alu53))) = gelu4((half4)(acc0.x + bias.x, acc1.x + bias.x, acc2.x + bias.x, acc3.x + bias.x));
  *((__global half4*)((data0_524288 + alu53 + 8192))) = gelu4((half4)(acc0.y + bias.y, acc1.y + bias.y, acc2.y + bias.y, acc3.y + bias.y));
  *((__global half4*)((data0_524288 + alu53 + 16384))) = gelu4((half4)(acc0.z + bias.z, acc1.z + bias.z, acc2.z + bias.z, acc3.z + bias.z));
  *((__global half4*)((data0_524288 + alu53 + 24576))) = gelu4((half4)(acc0.w + bias.w, acc1.w + bias.w, acc2.w + bias.w, acc3.w + bias.w));
}
"""
    elif getenv("CL_CUSTOM_DENSE_REDUCE") and name == "r_512_256_4":
      lib = b"""
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void r_512_256_4(__global half* data0_512, __global half* data1_512, __global half* data2_1024, __global half* data3_524288, __global half* data4_512) {
  int out_idx = get_group_id(0);
  int lid = get_local_id(0);
  __local half partials[256];

  half acc = ((half)(0.0f));
  int lsize = get_local_size(0);
  for (int Ridx0 = lid; Ridx0 < 256; Ridx0 += lsize) {
    int alu1 = (Ridx0 << 2);
    half4 val0 = *((__global half4*)((data2_1024 + alu1)));
    half4 val1 = *((__global half4*)((data3_524288 + ((out_idx << 10) + alu1))));
    acc = acc + (val0.x * val1.x) + (val0.y * val1.y) + (val0.z * val1.z) + (val0.w * val1.w);
  }

  partials[lid] = acc;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int offset = lsize >> 1; offset > 0; offset >>= 1) {
    if (lid < offset) partials[lid] += partials[lid + offset];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0) {
    half alu4 = (*(data1_512 + out_idx)) + partials[0] + (*(data4_512 + out_idx));
    *(data0_512 + out_idx) = ((((half)(0.0f)) < alu4) ? alu4 : ((half)(0.0f)));
  }
}
"""
    elif getenv("CL_CUSTOM_DENSE_REDUCE") and name == "r_512_4_256_4":
      lib = b"""
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void r_512_4_256_4(__global half* data0_2048, __global half* data1_1024, __global half* data2_2097152, __global half* data3_2048) {
  int out_idx = get_group_id(0);
  int lid = get_local_id(0);
  __local half4 partials[256];

  half4 acc = (half4)(((half)(0.0f)), ((half)(0.0f)), ((half)(0.0f)), ((half)(0.0f)));
  int lsize = get_local_size(0);
  for (int Ridx0 = lid; Ridx0 < 256; Ridx0 += lsize) {
    int alu4 = (Ridx0 << 2);
    half4 val0 = *((__global half4*)((data1_1024 + alu4)));
    int alu5 = ((out_idx << 12) + alu4);
    half4 val4 = *((__global half4*)((data2_2097152 + alu5)));
    half4 val1 = *((__global half4*)((data2_2097152 + alu5 + 1024)));
    half4 val2 = *((__global half4*)((data2_2097152 + alu5 + 2048)));
    half4 val3 = *((__global half4*)((data2_2097152 + alu5 + 3072)));
    acc.x = acc.x + (val0.x * val4.x) + (val0.y * val4.y) + (val0.z * val4.z) + (val0.w * val4.w);
    acc.y = acc.y + (val0.x * val1.x) + (val0.y * val1.y) + (val0.z * val1.z) + (val0.w * val1.w);
    acc.z = acc.z + (val0.x * val2.x) + (val0.y * val2.y) + (val0.z * val2.z) + (val0.w * val2.w);
    acc.w = acc.w + (val0.x * val3.x) + (val0.y * val3.y) + (val0.z * val3.z) + (val0.w * val3.w);
  }

  partials[lid] = acc;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int offset = lsize >> 1; offset > 0; offset >>= 1) {
    if (lid < offset) partials[lid] += partials[lid + offset];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0) {
    int alu11 = (out_idx << 2);
    *((__global half4*)((data0_2048 + alu11))) = partials[0] + *((__global half4*)((data3_2048 + alu11)));
  }
}
"""
    elif getenv("CL_CUSTOM_DENSE_REDUCE") and name == "r_512_512_4":
      lib = b"""
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void r_512_512_4(__global half* data0_512, __global half* data1_2048, __global half* data2_1048576, __global half* data3_512) {
  int out_idx = get_group_id(0);
  int lid = get_local_id(0);
  __local half partials[256];

  half acc = ((half)(0.0f));
  int lsize = get_local_size(0);
  for (int Ridx0 = lid; Ridx0 < 512; Ridx0 += lsize) {
    int alu1 = (Ridx0 << 2);
    half4 val0 = *((__global half4*)((data1_2048 + alu1)));
    half4 val1 = *((__global half4*)((data2_1048576 + ((out_idx << 11) + alu1))));
    acc = acc + (val0.x * val1.x) + (val0.y * val1.y) + (val0.z * val1.z) + (val0.w * val1.w);
  }

  partials[lid] = acc;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int offset = lsize >> 1; offset > 0; offset >>= 1) {
    if (lid < offset) partials[lid] += partials[lid + offset];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0) {
    half alu4 = partials[0] + (*(data3_512 + out_idx));
    *(data0_512 + out_idx) = ((((half)(0.0f)) < alu4) ? alu4 : ((half)(0.0f)));
  }
}
"""
    elif getenv("CL_CUSTOM_DENSE_REDUCE") and name == "r_256_4_128_4":
      lib = b"""
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void r_256_4_128_4(__global half* data0_1024, __global half* data1_512, __global half* data2_524288, __global half* data3_1024) {
  int out_idx = get_group_id(0);
  int lid = get_local_id(0);
  __local half4 partials[256];

  half4 acc = (half4)(((half)(0.0f)), ((half)(0.0f)), ((half)(0.0f)), ((half)(0.0f)));
  int lsize = get_local_size(0);
  for (int Ridx0 = lid; Ridx0 < 128; Ridx0 += lsize) {
    int alu4 = (Ridx0 << 2);
    half4 val0 = *((__global half4*)((data1_512 + alu4)));
    int alu5 = ((out_idx << 11) + alu4);
    half4 val4 = *((__global half4*)((data2_524288 + alu5)));
    half4 val1 = *((__global half4*)((data2_524288 + alu5 + 512)));
    half4 val2 = *((__global half4*)((data2_524288 + alu5 + 1024)));
    half4 val3 = *((__global half4*)((data2_524288 + alu5 + 1536)));
    acc.x = acc.x + (val0.x * val4.x) + (val0.y * val4.y) + (val0.z * val4.z) + (val0.w * val4.w);
    acc.y = acc.y + (val0.x * val1.x) + (val0.y * val1.y) + (val0.z * val1.z) + (val0.w * val1.w);
    acc.z = acc.z + (val0.x * val2.x) + (val0.y * val2.y) + (val0.z * val2.z) + (val0.w * val2.w);
    acc.w = acc.w + (val0.x * val3.x) + (val0.y * val3.y) + (val0.z * val3.z) + (val0.w * val3.w);
  }

  partials[lid] = acc;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int offset = lsize >> 1; offset > 0; offset >>= 1) {
    if (lid < offset) partials[lid] += partials[lid + offset];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0) {
    half4 res = partials[0] + *((__global half4*)((data3_1024 + (out_idx << 2))));
    res = max(res, (half4)(((half)(0.0f)), ((half)(0.0f)), ((half)(0.0f)), ((half)(0.0f))));
  *((__global half4*)((data0_1024 + (out_idx << 2)))) = res;
  }
}
"""
    if getenv("CL_RESTRICT"):
      lib = lib.replace(b"__global half* data", b"__global half* restrict data")
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
        else:
          check(cl.clSetKernelArg(self.kernel, real_i, ctypes.sizeof(b), ctypes.byref(b)))
    for i,v in enumerate(vals,start=i+1): check(cl.clSetKernelArg(self.kernel, i, 4, ctypes.byref(ctypes.c_int32(v))))
    dense_wg = int(getenv("CL_DENSE_WG", 32))
    if getenv("CL_CUSTOM_DENSE_REDUCE") and self.name == "r_256_4_128_4":
      global_size, local_size = (256, 1, 1), (dense_wg, 1, 1)
    elif getenv("CL_CUSTOM_DENSE_REDUCE") and self.name in {"r_512_256_4", "r_512_4_256_4", "r_512_512_4"}:
      global_size, local_size = (512, 1, 1), (dense_wg, 1, 1)
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
      flush_every = int(getenv("CL_FLUSH_EVERY", 6))
      if flush_every > 0 and self.dev._flush_counter % flush_every == 0: cl.clFlush(self.dev.queue)
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
    return checked(cl.clCreateBuffer(self.dev.context, cl.CL_MEM_READ_WRITE, size, None, status := ctypes.c_int32()), status)
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
