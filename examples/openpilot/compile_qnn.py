"""Package a compiled HTP context as a TinyJit model artifact on the target device."""
import argparse, ctypes, functools, hashlib, math
from pathlib import Path
import numpy as np
from tinygrad import Device, Tensor, TinyJit
from tinygrad.dtype import _to_np_dtype
from tinygrad.nn.onnx import OnnxPBParser, OnnxRunner
from tinygrad.runtime.autogen import qnn
from tinygrad.runtime.ops_qnn import check, tensor_info, versioned
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from examples.openpilot.compile_onnx import onnx_metadata
from examples.openpilot.helpers import allocate_inputs, dump_pickle, load_pickle

def qnn_kernel(binary, *args):
  sink = UOp.sink(*(x.base for x in args), arg=KernelInfo(name='qnn_model'))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=''), UOp(Ops.BINARY, arg=binary)))

def graph_names(binary):
  api = Device['QNN'].system
  context, info, size = ctypes.c_void_p(), ctypes.POINTER(qnn.QnnSystemContext_BinaryInfo_t)(), ctypes.c_uint64()
  data = ctypes.create_string_buffer(binary)
  check(api.systemContextCreate(ctypes.byref(context)))
  try:
    check(api.systemContextGetBinaryInfo(context, data, len(binary), ctypes.byref(info), ctypes.byref(size)))
    metadata = versioned(info.contents, 'contextBinaryInfoV', (1, 2, 3))
    if metadata.numGraphs != 1: raise ValueError('Expected one QNN graph')
    graph = versioned(metadata.graphs[0], 'graphInfoV', (1, 2, 3))
    return ([ctypes.string_at(tensor_info(graph.graphInputs[i]).name).decode() for i in range(graph.numGraphInputs)],
            [ctypes.string_at(tensor_info(graph.graphOutputs[i]).name).decode() for i in range(graph.numGraphOutputs)])
  finally: check(api.systemContextFree(context))

def bounded_concat(*values, axis):
  # Bound QCOM kernel input count when assembling many independently quantized heads.
  while len(values) > 8:
    values = tuple(Tensor.cat(*values[i:i+8], dim=axis).contiguous().realize() for i in range(0, len(values), 8))
  return Tensor.cat(*values, dim=axis).contiguous().realize()

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('onnx')
  parser.add_argument('context')
  parser.add_argument('output')
  parser.add_argument('--sample', required=True, help='NPZ containing representative original-model inputs')
  parser.add_argument('--pre', help='Non-neural input preparation ONNX, executed on DEV')
  parser.add_argument('--post', help='Non-neural output/state assembly ONNX, executed on DEV')
  args = parser.parse_args()
  if bool(args.pre) != bool(args.post): parser.error('--pre and --post must be supplied together')
  binary = Path(args.context).read_bytes()
  inputs, outputs = graph_names(binary)
  graph = OnnxPBParser(Path(args.onnx)).parse()['graph']
  def specs(values, names, device):
    types = {v['name']: v['parsed_type'] for v in values}
    return {name: (types[name].shape, np.dtype(_to_np_dtype(types[name].dtype)).name, device) for name in names}
  if args.pre:
    if Device.DEFAULT.split(':')[0] == 'QNN': parser.error('Composite packaging requires a tensor device, such as DEV=QCOM')
    pre, post = OnnxRunner(Path(args.pre)), OnnxRunner(Path(args.post))
    if Device.DEFAULT.split(':')[0] == 'QCOM': post.onnx_ops = post.onnx_ops | {'Concat': bounded_concat}
    pre_graph, post_graph = OnnxPBParser(Path(args.pre)).parse()['graph'], OnnxPBParser(Path(args.post)).parse()['graph']
    native_specs = specs(post_graph['input'], outputs, 'QNN')
    if {dtype for _, dtype, _ in native_specs.values()} != {'float32'}: raise ValueError('Composite native heads must be float32')
    native_sizes = {k: math.prod(shape) for k,(shape,_,_) in native_specs.items()}
    if not set(inputs) <= {v['name'] for v in pre_graph['output']}: raise ValueError('Preprocessing does not supply every native input')
    input_specs = specs(graph['input'], [v['name'] for v in graph['input']], Device.DEFAULT)
    output_specs = specs(graph['output'], [v['name'] for v in graph['output']], Device.DEFAULT)
  else:
    input_specs, output_specs = specs(graph['input'], inputs, 'QNN'), specs(graph['output'], outputs, 'QNN')
  properties, output_shapes = onnx_metadata(Path(args.onnx))
  metadata = {'metadata': properties, 'input_shapes': {k:v[0] for k,v in input_specs.items()}, 'output_shapes': output_shapes}
  sample = np.load(args.sample)
  for name, (shape, dtype, _) in input_specs.items():
    if sample[name].shape != shape or sample[name].dtype != np.dtype(dtype):
      raise ValueError(f'Sample {name} must have shape {shape} and dtype {dtype}')
    if not np.isfinite(sample[name]).all(): raise ValueError(f'Sample {name} contains nonfinite values')
  tensors = {name: Tensor(sample[name], device=device).realize() for name, (_, _, device) in input_specs.items()}
  output_buffers = allocate_inputs(output_specs)

  @TinyJit(prune=True)
  def run(output_buffers, **kwargs):
    if args.pre:
      prepared = pre(kwargs)
      packed = Tensor.empty(sum(native_sizes.values()), dtype='float32', device='QNN')
      result = Tensor.custom_kernel(packed, *(prepared[n].to('QNN').contiguous() for n in inputs),
                                    fxn=functools.partial(qnn_kernel, binary))[0].realize()
      transferred = result.to(Device.DEFAULT).realize()
      heads, offset = {}, 0
      for name in outputs:
        heads[name] = transferred[offset:offset+native_sizes[name]].reshape(native_specs[name][0])
        offset += native_sizes[name]
      assembled = prepared | post(kwargs | heads)
      Tensor.realize(*(output_buffers[k].assign(assembled[k]) for k in output_specs))
    else:
      Tensor.realize(*Tensor.custom_kernel(*(output_buffers[n] for n in outputs), *(kwargs[n] for n in inputs),
                                          fxn=functools.partial(qnn_kernel, binary))[:len(outputs)])

  for _ in range(3): run(output_buffers=output_buffers, **tensors)
  expected = {k:v.numpy().copy() for k,v in output_buffers.items()}
  if not all(np.isfinite(v).all() for v in expected.values()): raise ValueError('QNN outputs contain nonfinite values')
  provenance = {'context_sha256': hashlib.sha256(binary).hexdigest(), 'onnx_sha256': hashlib.sha256(Path(args.onnx).read_bytes()).hexdigest()}
  if args.pre: provenance.update({k+'_sha256': hashlib.sha256(Path(v).read_bytes()).hexdigest() for k,v in [('pre', args.pre), ('post', args.post)]})
  dump_pickle({'metadata': metadata, 'run': run, 'input_specs': input_specs, 'output_specs': output_specs, 'qnn': provenance}, args.output)
  loaded = load_pickle(args.output, out_of_band=True)
  fresh_outputs = allocate_inputs(output_specs)
  fresh_inputs = {name: Tensor(sample[name].copy(), device=device).realize() for name, (_, _, device) in input_specs.items()}
  loaded['run'](output_buffers=fresh_outputs, **fresh_inputs)
  for name in output_specs: np.testing.assert_array_equal(fresh_outputs[name].numpy(), expected[name])
  changed = {name: np.flip(sample[name]).copy() for name in input_specs}
  changed_inputs = {name: Tensor(changed[name], device=device).realize() for name, (_, _, device) in input_specs.items()}
  run(output_buffers=output_buffers, **changed_inputs)
  loaded['run'](output_buffers=fresh_outputs, **changed_inputs)
  for name in output_specs: np.testing.assert_array_equal(fresh_outputs[name].numpy(), output_buffers[name].numpy())
  print('PASS: QNN TinyJit capture, serialization, and input/output buffer rebinding', input_specs, output_specs)
