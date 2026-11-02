"""Synchronous Qualcomm HTP context binaries, dispatched through Tinygrad programs."""
import ctypes, math, os, time, weakref
from multiprocessing.util import Finalize
import numpy as np
from tinygrad.device import Compiled, HostAllocator, Program, TinyELF
from tinygrad.dtype import _to_np_dtype
from tinygrad.runtime.autogen import qnn

def check(status: int):
  if status: raise RuntimeError(f'QNN error {status:#x}')

def provider(lib, name, struct, version_field, major, minor):
  fn = getattr(lib, name)
  fn.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(struct))), ctypes.POINTER(ctypes.c_uint32)]
  fn.restype = ctypes.c_uint64
  providers, count = ctypes.POINTER(ctypes.POINTER(struct))(), ctypes.c_uint32()
  check(fn(ctypes.byref(providers), ctypes.byref(count)))
  for i in range(count.value):
    p = providers[i].contents
    version = getattr(p, version_field)
    if version_field == 'apiVersion': version = version.coreApiVersion
    if version.major == major and version.minor >= minor: return p
  raise RuntimeError(f'QNN provider requires API {major}.{minor}')

def versioned(obj, prefix, supported):
  if obj.version not in supported: raise RuntimeError(f'Unsupported QNN {prefix} version {obj.version}')
  return getattr(obj, f'{prefix}{obj.version}')

def tensor_info(tensor): return versioned(tensor, 'v', (1, 2))

class QNNProgram(Program):
  def __init__(self, dev, obj:TinyELF):
    self.dev, self.context, self.system_context, self.graph = dev, ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
    self.binary = ctypes.create_string_buffer(obj.lib)
    self.inputs: tuple[qnn.Qnn_Tensor_t, ...] | ctypes.Array[qnn.Qnn_Tensor_t] = ()
    self.outputs: tuple[qnn.Qnn_Tensor_t, ...] | ctypes.Array[qnn.Qnn_Tensor_t] = ()
    self.bindings = []
    dev.programs.add(self)
    try:
      check(dev.system.systemContextCreate(ctypes.byref(self.system_context)))
      info, size = ctypes.POINTER(qnn.QnnSystemContext_BinaryInfo_t)(), ctypes.c_uint64()
      check(dev.system.systemContextGetBinaryInfo(self.system_context, self.binary, len(obj.lib), ctypes.byref(info), ctypes.byref(size)))
      context = versioned(info.contents, 'contextBinaryInfoV', (1, 2, 3))
      if context.numGraphs != 1: raise RuntimeError('QNN programs require one graph')
      graph = versioned(context.graphs[0], 'graphInfoV', (1, 2, 3))
      self.inputs = (qnn.Qnn_Tensor_t * graph.numGraphInputs)(*[graph.graphInputs[i] for i in range(graph.numGraphInputs)])
      self.outputs = (qnn.Qnn_Tensor_t * graph.numGraphOutputs)(*[graph.graphOutputs[i] for i in range(graph.numGraphOutputs)])
      tensors: list[qnn.Qnn_Tensor_t] = [*self.outputs, *self.inputs]
      signatures = [(signature, 0) for signature in obj.signature]
      if len(self.outputs) > 1 and len(obj.signature) == len(self.inputs) + 1:
        # One external output buffer can hold every head, preserving each head's native quantization.
        name, slot, dtype, shape = obj.signature[0]
        sizes = [math.prod(tensor_info(t).dimensions[i] for i in range(tensor_info(t).rank)) for t in self.outputs]
        if math.prod(shape) != sum(sizes): raise ValueError('QNN packed output shape mismatch')
        signatures, offset = [], 0
        for size in sizes:
          signatures.append(((name, slot, dtype, (size,)), offset))
          offset += size * dtype.itemsize
        signatures.extend((signature, 0) for signature in obj.signature[1:])
      if len(tensors) != len(signatures): raise ValueError('QNN buffer count does not match the program signature')
      for tensor, ((_, slot, dtype, shape), byte_offset) in zip(tensors, signatures):
        t = tensor_info(tensor)
        dims = tuple(t.dimensions[i] for i in range(t.rank))
        if math.prod(shape) != math.prod(dims): raise ValueError(f'QNN shape mismatch: {shape} vs {dims}')
        external_dtype = np.dtype(_to_np_dtype(dtype))
        native_dtype = {qnn.QNN_DATATYPE_FLOAT_32: np.float32, qnn.QNN_DATATYPE_FLOAT_16: np.float16,
                        qnn.QNN_DATATYPE_UINT_8: np.uint8, qnn.QNN_DATATYPE_UINT_16: np.uint16,
                        qnn.QNN_DATATYPE_INT_8: np.int8, qnn.QNN_DATATYPE_INT_16: np.int16,
                        qnn.QNN_DATATYPE_INT_32: np.int32, qnn.QNN_DATATYPE_UINT_32: np.uint32,
                        qnn.QNN_DATATYPE_UFIXED_POINT_8: np.uint8, qnn.QNN_DATATYPE_UFIXED_POINT_16: np.uint16,
                        qnn.QNN_DATATYPE_SFIXED_POINT_8: np.int8, qnn.QNN_DATATYPE_SFIXED_POINT_16: np.int16}[t.dataType]
        quant = t.dataType in (qnn.QNN_DATATYPE_UFIXED_POINT_8, qnn.QNN_DATATYPE_UFIXED_POINT_16,
                              qnn.QNN_DATATYPE_SFIXED_POINT_8, qnn.QNN_DATATYPE_SFIXED_POINT_16)
        scale, offset = 1.0, 0
        if quant:
          if t.quantizeParams.quantizationEncoding != qnn.QNN_QUANTIZATION_ENCODING_SCALE_OFFSET:
            raise NotImplementedError('QNN I/O requires per-tensor quantization')
          scale, offset = t.quantizeParams.scaleOffsetEncoding.scale, t.quantizeParams.scaleOffsetEncoding.offset
          if not math.isfinite(scale) or scale <= 0: raise ValueError('Invalid QNN I/O quantization scale')
        native = np.empty(math.prod(dims), dtype=native_dtype)
        t.memType, t.clientBuf.data, t.clientBuf.dataSize = qnn.QNN_TENSORMEMTYPE_RAW, native.ctypes.data, native.nbytes
        self.bindings.append((t, slot, byte_offset, external_dtype, native, scale, offset, quant))
      check(dev.api.contextCreateFromBinary(dev.backend, dev.handle, None, self.binary, len(obj.lib), ctypes.byref(self.context), None))
      check(dev.api.graphRetrieve(self.context, graph.graphName, ctypes.byref(self.graph)))
    except Exception:
      self.close()
      raise

  def __call__(self, *bufs, global_size=(1,1,1), local_size=(1,1,1), vals=(), wait=False, timeout=None):
    if vals: raise ValueError('QNN programs require fixed shapes')
    if not self.context.value: raise RuntimeError('QNN program is closed')
    start = time.perf_counter()
    views = []
    for t, slot, byte_offset, dtype, native, scale, offset, quant in self.bindings:
      address = bufs[slot] + byte_offset
      view = np.frombuffer((ctypes.c_ubyte * (native.size * dtype.itemsize)).from_address(address), dtype=dtype)
      views.append(view)
      if dtype == native.dtype and (not quant or (scale == 1 and offset == 0)): t.clientBuf.data = address
    for binding, view in zip(self.bindings[len(self.outputs):], views[len(self.outputs):]):
      t, _, _, dtype, native, scale, offset, quant = binding
      if dtype == native.dtype and (not quant or (scale == 1 and offset == 0)): continue
      if quant:
        if not np.isfinite(view).all(): raise ValueError('Nonfinite QNN quantized input')
        bounds = np.iinfo(native.dtype)
        np.copyto(native, np.clip(np.rint(view.astype(np.float32) / scale - offset), bounds.min, bounds.max), casting='unsafe')
      elif dtype != native.dtype: np.copyto(native, view, casting='unsafe')
    check(self.dev.api.graphExecute(self.graph, self.inputs, len(self.inputs), self.outputs, len(self.outputs), None, None))
    for binding, view in zip(self.bindings[:len(self.outputs)], views[:len(self.outputs)]):
      _, _, _, dtype, native, scale, offset, quant = binding
      if dtype == native.dtype and (not quant or (scale == 1 and offset == 0)): continue
      if quant: np.copyto(view, (native.astype(np.float32) + offset) * scale, casting='unsafe')
      elif dtype != native.dtype: np.copyto(view, native, casting='unsafe')
    return time.perf_counter() - start if wait else None

  def close(self):
    if self.context.value:
      check(self.dev.api.contextFree(self.context, None))
      self.context = ctypes.c_void_p()
    if self.system_context.value:
      check(self.dev.system.systemContextFree(self.system_context))
      self.system_context = ctypes.c_void_p()

  def __del__(self):
    if getattr(self, 'context', None) or getattr(self, 'system_context', None): self.close()

class QNNDevice(Compiled):
  def __init__(self, device:str):
    self.programs: weakref.WeakSet[QNNProgram] = weakref.WeakSet()
    self.power_id: ctypes.c_uint32 | None = None
    self.perf: qnn.QnnHtpDevice_PerfInfrastructure_t
    self.backend, self.handle = ctypes.c_void_p(), ctypes.c_void_p()
    path = os.environ.get('QNN_LIB_PATH', '')
    self.lib = ctypes.CDLL(os.path.join(path, 'libQnnHtp.so'))
    self.system_lib = ctypes.CDLL(os.path.join(path, 'libQnnSystem.so'))
    self.api = provider(self.lib, 'QnnInterface_getProviders', qnn.QnnInterface_t, 'apiVersion', 2, 32).v2_32
    self.system = provider(self.system_lib, 'QnnSystemInterface_getProviders', qnn.QnnSystemInterface_t, 'systemApiVersion', 1, 7).v1_7
    check(self.api.backendCreate(None, None, ctypes.byref(self.backend)))
    try: check(self.api.deviceCreate(None, None, ctypes.byref(self.handle)))
    except Exception:
      check(self.api.backendFree(self.backend))
      raise
    super().__init__(device, HostAllocator(self), [], QNNProgram)
    try: self.configure_performance(os.environ.get('QNN_PERF', 'turbo'))
    except Exception:
      self.finalize()
      raise
    # Forked multiprocessing workers bypass ordinary atexit callbacks with os._exit.
    self._process_finalizer = Finalize(None, self.finalize, exitpriority=10)

  def configure_performance(self, profile):
    if profile not in ('default', 'nominal', 'turbo'): raise ValueError(f'Unknown QNN performance profile: {profile}')
    if profile == 'default':
      if self.power_id is not None:
        check(self.perf.destroyPowerConfigId(self.power_id))
        self.power_id = None
      return
    corner = {'nominal': qnn.DCVS_VOLTAGE_VCORNER_NOM, 'turbo': qnn.DCVS_VOLTAGE_VCORNER_TURBO}[profile]
    infrastructure = ctypes.POINTER(qnn.QnnHtpDevice_Infrastructure_t)()
    check(self.api.deviceGetInfrastructure(ctypes.byref(infrastructure)))
    if infrastructure.contents.infraType != qnn.QNN_HTP_DEVICE_INFRASTRUCTURE_TYPE_PERF:
      raise RuntimeError('QNN performance infrastructure is unavailable')
    self.perf = infrastructure.contents.perfInfra
    if self.power_id is None:
      power_id = ctypes.c_uint32()
      check(self.perf.createPowerConfigId(0, 0, ctypes.byref(power_id)))
      self.power_id = power_id
    config = qnn.QnnHtpPerfInfrastructure_PowerConfig_t(option=qnn.QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3)
    d = config.dcvsV3Config
    d.contextId, d.setDcvsEnable, d.dcvsEnable = self.power_id.value, 1, 0
    d.powerMode = qnn.QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE
    d.setSleepLatency, d.sleepLatency, d.setBusParams, d.setCoreParams = 1, 100, 1, 1
    d.busVoltageCornerMin = d.busVoltageCornerTarget = d.busVoltageCornerMax = corner
    d.coreVoltageCornerMin = d.coreVoltageCornerTarget = d.coreVoltageCornerMax = corner
    pointer_type = ctypes.POINTER(qnn.QnnHtpPerfInfrastructure_PowerConfig_t)
    configs = (pointer_type * 2)(ctypes.pointer(config), pointer_type())
    check(self.perf.setPowerConfig(self.power_id, configs))

  def synchronize(self, timeout=None): pass  # graphExecute is blocking.

  def finalize(self):
    for program in list(self.programs): program.close()
    if self.power_id is not None:
      check(self.perf.destroyPowerConfigId(self.power_id))
      self.power_id = None
    if self.handle.value:
      check(self.api.deviceFree(self.handle))
      self.handle = ctypes.c_void_p()
    if self.backend.value:
      check(self.api.backendFree(self.backend))
      self.backend = ctypes.c_void_p()
