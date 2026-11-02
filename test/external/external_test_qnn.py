"""Run with DEV=QNN on an HTP device. Creates its test context through the native API."""
import ctypes, functools, pickle, unittest
import numpy as np
from tinygrad import Device, Tensor, TinyJit
from tinygrad.device import TinyELF
from tinygrad.dtype import dtypes
from tinygrad.runtime.autogen import qnn
from tinygrad.runtime.ops_qnn import QNNProgram, check
from tinygrad.uop.ops import UOp, Ops, KernelInfo

def add_context(dev, second_output=False):
  context, graph = ctypes.c_void_p(), ctypes.c_void_p()
  check(dev.api.contextCreate(dev.backend, dev.handle, None, ctypes.byref(context)))
  try:
    name = ctypes.create_string_buffer(b'add_test')
    check(dev.api.graphCreate(context, name, None, ctypes.byref(graph)))
    tensors, keepalive = [], []
    definitions = [('input', 0, (1,1,1,16), 2/65535, -32768),
                   ('constant', 4, (1,), .5, 0), ('output', 1, (1,1,1,16), 2/65535, -16384)]
    if second_output: definitions.append(('second', 1, (1,1,1,16), 2/65535, 0))
    for name, kind, shape, scale, offset in definitions:
      tensor = qnn.Qnn_Tensor_t(version=1)
      label, dims = ctypes.create_string_buffer(name.encode()), (ctypes.c_uint32 * len(shape))(*shape)
      keepalive.extend((label, dims))
      t = tensor.v1
      t.name, t.type, t.dataType, t.rank, t.dimensions = label, kind, qnn.QNN_DATATYPE_UFIXED_POINT_16, len(shape), dims
      t.quantizeParams.encodingDefinition, t.quantizeParams.quantizationEncoding = 1, 0
      t.quantizeParams.scaleOffsetEncoding.scale, t.quantizeParams.scaleOffsetEncoding.offset = scale, offset
      if kind == qnn.QNN_TENSOR_TYPE_STATIC:
        data = (ctypes.c_uint16 * 1)(1)
        keepalive.append(data)
        t.clientBuf.data, t.clientBuf.dataSize = ctypes.addressof(data), ctypes.sizeof(data)
      check(dev.api.tensorCreateGraphTensor(graph, ctypes.byref(tensor)))
      tensors.append(tensor)
    inputs, outputs = (qnn.Qnn_Tensor_t * 2)(*tensors[:2]), (qnn.Qnn_Tensor_t * 1)(tensors[2])
    op = qnn.Qnn_OpConfig_t(version=1)
    strings = [ctypes.create_string_buffer(s) for s in (b'add', b'qti.aisw', b'ElementWiseAdd')]
    op.v1.name, op.v1.packageName, op.v1.typeName = strings
    op.v1.numOfInputs, op.v1.inputTensors, op.v1.numOfOutputs, op.v1.outputTensors = 2, inputs, 1, outputs
    check(dev.api.graphAddNode(graph, op))
    if second_output:
      inputs, outputs = (qnn.Qnn_Tensor_t * 2)(tensors[2], tensors[1]), (qnn.Qnn_Tensor_t * 1)(tensors[3])
      second_name = ctypes.create_string_buffer(b'add_second')
      op.v1.name, op.v1.inputTensors, op.v1.outputTensors = second_name, inputs, outputs
      check(dev.api.graphAddNode(graph, op))
    check(dev.api.graphFinalize(graph, None, None))
    size, written = ctypes.c_uint64(), ctypes.c_uint64()
    check(dev.api.contextGetBinarySize(context, ctypes.byref(size)))
    binary = ctypes.create_string_buffer(size.value)
    check(dev.api.contextGetBinary(context, binary, size, ctypes.byref(written)))
    return binary.raw[:written.value]
  finally: check(dev.api.contextFree(context, None))

def kernel(binary, *args):
  sink = UOp.sink(*(x.base for x in args), arg=KernelInfo(name='qnn_test'))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=''), UOp(Ops.BINARY, arg=binary)))

@unittest.skipUnless(Device.DEFAULT == 'QNN', 'requires the QNN device')
class TestQNN(unittest.TestCase):
  @classmethod
  def setUpClass(cls): cls.binary = add_context(Device['QNN'])

  def test_jit_and_serialization_rebind_buffers(self):
    @TinyJit
    def run(output, source): Tensor.custom_kernel(output, source, fxn=functools.partial(kernel, self.binary))[0].realize()
    for i in range(8):
      values = np.linspace(-.9+i*.05, .7+i*.025, 16, dtype=np.float32).reshape(1,1,1,16)
      source, output = Tensor(values, device='QNN').realize(), Tensor(np.zeros_like(values), device='QNN').realize()
      if i == 4: run = pickle.loads(pickle.dumps(run))
      run(output, source)
      np.testing.assert_allclose(output.numpy(), values+.5, atol=1e-4, rtol=0)

  def test_reject_shape_mismatch(self):
    dev = Device['QNN']
    obj = TinyELF(self.binary, 'invalid', dev.renderer.target,
                  ((None, 0, dtypes.float32, (16,)), (None, 1, dtypes.float32, (15,))))
    with self.assertRaisesRegex(ValueError, 'shape mismatch'): QNNProgram(dev, obj)

  def test_packed_outputs_rebind_and_reject_bad_size(self):
    dev = Device['QNN']
    binary = add_context(dev, second_output=True)
    @TinyJit
    def run(output, source): Tensor.custom_kernel(output, source, fxn=functools.partial(kernel, binary))[0].realize()
    for i in range(6):
      values = np.linspace(-.9+i*.05, .7+i*.025, 16, dtype=np.float32)
      source, output = Tensor(values, device='QNN').realize(), Tensor.empty(32, device='QNN').realize()
      if i == 3: run = pickle.loads(pickle.dumps(run))
      run(output, source)
      np.testing.assert_allclose(output.numpy(), np.concatenate((values+.5, values+1)), atol=1e-4, rtol=0)
    obj = TinyELF(binary, 'invalid_packed', dev.renderer.target,
                  ((None, 0, dtypes.float32, (31,)), (None, 1, dtypes.float32, (16,))))
    with self.assertRaisesRegex(ValueError, 'packed output shape mismatch'): QNNProgram(dev, obj)

  def test_reject_nonfinite_input(self):
    for value in (np.nan, np.inf, -np.inf):
      source = Tensor(np.full((16,), value, np.float32), device='QNN').realize()
      output = Tensor(np.zeros((16,), np.float32), device='QNN').realize()
      with self.assertRaisesRegex(ValueError, 'Nonfinite'):
        Tensor.custom_kernel(output, source, fxn=functools.partial(kernel, self.binary))[0].realize()

  def test_close_is_idempotent(self):
    dev = Device['QNN']
    obj = TinyELF(self.binary, 'close', dev.renderer.target,
                  ((None, 0, dtypes.float32, (16,)), (None, 1, dtypes.float32, (16,))))
    program = QNNProgram(dev, obj)
    program.close()
    program.close()
    with self.assertRaisesRegex(RuntimeError, 'closed'): program()

  def test_performance_vote_lifetime(self):
    dev = Device['QNN']
    try:
      dev.configure_performance('nominal')
      identifier = dev.power_id.value
      dev.configure_performance('turbo')
      self.assertEqual(dev.power_id.value, identifier)
      dev.configure_performance('default')
      self.assertIsNone(dev.power_id)
      dev.configure_performance('default')
      with self.assertRaisesRegex(ValueError, 'profile'): dev.configure_performance('invalid')
    finally: dev.configure_performance('turbo')

if __name__ == '__main__': unittest.main()
