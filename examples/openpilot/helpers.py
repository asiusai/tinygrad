"""Shared compilation and input allocation for model and warp artifacts."""
import io, json, pathlib, pickle, shutil, struct, time, tempfile
import numpy as np
from typing import Callable, Sequence
from tinygrad import Tensor, Device, Context, dtypes
from tinygrad.device import Buffer
from tinygrad.uop.ops import PatternMatcher, UPat, Ops, graph_rewrite, AxisType
from tinygrad.codegen.opt import Opt, OptOps, KERNEL_OPTS
from tinygrad.nn.state import get_parameters

def allocate_inputs(input_specs, initialize=None):
  """Initialize inputs before copying to devices."""
  arrays = {name: np.zeros(shape, dtype=dtype) for name, (shape, dtype, _) in input_specs.items()}
  if initialize is not None: initialize(arrays)
  return {name: Tensor(arrays[name], device=device).realize() for name, (_, _, device) in input_specs.items()}

def _pending_reduce(t:Tensor) -> bool:
  """True if computing t runs a reduction that is not stored in its own buffer yet."""
  gate = lambda u: u.op is not Ops.STAGE and not u.has_buffer_identity()
  return any(u.op is Ops.REDUCE for u in t.uop.toposort(gate=gate))

def openpilot_onnx_ops(onnx_ops:dict) -> dict:
  """ONNX ops that schedule faster kernels for the openpilot models. Only the float rounding order changes."""
  def Concat(*xs:Tensor, axis:int):
    # a concat of several pending reductions fuses into one kernel that runs every reduction in every thread.
    # store each reduction first, so each one runs in its own kernel.
    if sum(map(_pending_reduce, xs)) >= 2: xs = tuple(x.contiguous() if _pending_reduce(x) else x for x in xs)
    return Tensor.cat(*xs, dim=axis)
  def ReduceMean(data:Tensor, axes:Sequence[int]|None=None, keepdims:int=1, noop_with_empty_axes:int=0):
    # a mean of a pending reduction (e.g. a conv) fuses into one reduction with few threads: store the reduction first.
    # only float32 data, a half buffer would round it
    if data.dtype == dtypes.float32 and _pending_reduce(data): data = data.contiguous()
    return onnx_ops['ReduceMean'](data, axes, keepdims, noop_with_empty_axes)
  return {**onnx_ops, 'Concat': Concat, 'ReduceMean': ReduceMean}

def load_kernel_opts(path) -> int:
  """Load tuned kernel opts: {"<kernel AST key hex>": [[op, axis, arg], ...]}, a SPLIT arg is [amount, axis type, top]."""
  def opt(op, axis, arg): return Opt(OptOps[op], axis, (arg[0], AxisType[arg[1]], *arg[2:]) if isinstance(arg, list) else arg)
  table = json.load(open(path))
  KERNEL_OPTS.update({bytes.fromhex(k): tuple(opt(*o) for o in v["opts"]) for k, v in table.items()})
  return len(table)

def dump_pickle(obj, path):
  with tempfile.TemporaryFile(dir=".") as buffers:
    opcodes, offset = io.BytesIO(), 0

    def persistent_id(b):
      nonlocal offset
      if not isinstance(b, Buffer) or b.device != Device.DEFAULT: return None
      offset += buffers.write(bytes((-offset) % 256))
      offset = (off:=offset) + buffers.write(b.as_memoryview())
      return b.nbytes, off

    p = pickle.Pickler(opcodes)
    p.persistent_id = persistent_id
    p.dump(obj)
    with open(path, "wb") as f:
      f.write(struct.pack('<q', len(opcodes_b:=opcodes.getvalue())))
      f.write(opcodes_b)
      buffers.seek(0)
      shutil.copyfileobj(buffers, f)

def load_pickle(path, *, out_of_band=False):
  with open(path, "rb") as f:
    if not out_of_band: return pickle.load(f)
    opcodes, buffers = f.read(struct.unpack('<q', f.read(8))[0]), Tensor(pathlib.Path(path))[f.tell():].to(Device.DEFAULT).realize().uop.buffer

  def persistent_load(pid): return buffers.view(*pid)

  u = pickle.Unpickler(io.BytesIO(opcodes))
  u.persistent_load = persistent_load
  return u.load()

@Context(OPENPILOT_HACKS=1, **{'AMD': {'TC_OPT': 2, 'TC_MIN_GLOBALS': 32}}.get(Device.DEFAULT, {}))
def benchmark(fxn:Callable, cb=None, **kwargs):
  Device.default.synchronize()
  start = time.perf_counter()
  if (output := fxn(**kwargs)) is not None: output.realize()
  Device.default.synchronize()
  end = time.perf_counter()
  if cb: cb(end-start)
  return [t.numpy().copy() for t in get_parameters(kwargs.get('output_buffers', output))]

pm_retargetable = PatternMatcher([
  (UPat(Ops.PROGRAM, src=(UPat(), UPat(), UPat(), UPat()), name="p"), lambda p: p.replace(src=p.src[:-1]) if p.arg.target.device == "CPU" else None)
])

def make_retargetable(jit): jit.captured._linear = graph_rewrite(jit.captured._linear, pm_retargetable, walk=True, enter_calls=True)
