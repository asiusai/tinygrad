from typing import TypeVar, Generic, Callable, cast, Any
import functools, collections
from tinygrad.tensor import Tensor
from tinygrad.helpers import flatten, merge_dicts, DEBUG, Context, BEAM, getenv, colored, JIT, JIT_BATCH_SIZE, dedup, unwrap, pluralize, PROFILE, all_int
from tinygrad.device import Buffer, Compiled, Device, MultiBuffer
from tinygrad.dtype import DType
from tinygrad.uop.ops import UOp, Variable, sym_infer, Ops, buffers, track_rewrites
from tinygrad.engine.realize import ExecItem, capturing, ViewOp, BufferCopy, BufferXfer, EncDec, CompiledRunner, Runner, Estimates
from tinygrad.engine.memory import memory_plan_rewrite, _collect_bufs
from tinygrad.engine.schedule import linear_to_schedule
from tinygrad.nn.state import get_parameters
from tinygrad.schedule.rangeify import mop_cleanup
from dataclasses import dataclass

def prune_linear(linear:UOp, needed:set[UOp]) -> tuple[UOp, UOp]:
  kept, onetime = [], []
  for si in linear.src:
    si_bufs = {b for src in si.src[1:] for b in _collect_bufs(src)}
    if not si_bufs.isdisjoint(needed):
      kept.append(si)
      needed |= si_bufs
    else: onetime.append(si)
  return linear.replace(src=tuple(kept)), linear.replace(src=tuple(onetime))

@track_rewrites(lambda linear,held_bufs,ret: f"JIT {pluralize('Kernel', len(ret))}")
def jit_lower(linear:UOp, held_bufs:set[UOp]) -> list[ExecItem]:
  return [ei.lower() for ei in linear_to_schedule(memory_plan_rewrite(linear, held_bufs))]

class GraphException(Exception): pass
class JitError(Exception): pass

def _check_no_non_tensor_return(ret):
  if ret is None or isinstance(ret, Tensor): return
  if isinstance(ret, (tuple, list, dict)):
    for item in (ret.values() if isinstance(ret, dict) else ret): _check_no_non_tensor_return(item)
    return
  raise JitError(f"JIT return contains non-Tensor value of type {type(ret).__name__}")

def graph_class(dev): return dev.graph.func if isinstance(dev.graph, functools.partial) else dev.graph

def apply_graph_to_jit(jit_cache: list[ExecItem], input_buffers: list[Buffer], var_vals: dict[str, int],
                       orig_valid_positions: dict[int, set[int]]|None = None, max_batch_size=0) -> list[ExecItem]:
  # Split JIT cache into batches for faster graph execution.
  # This allows the accelerator to run some batches while subsequent graphs are still being updated.
  graphed_jit_cache: list[ExecItem] = []
  current_batch: list[ExecItem] = []
  current_batch_devs: list[Compiled] = []

  def flush_batch():
    nonlocal current_batch, current_batch_devs, max_batch_size
    try:
      if len(current_batch_devs) == 0: raise GraphException("no device for graph")
      if len(current_batch) <= 1 and not getenv("GRAPH_ONE_KERNEL"): raise GraphException("only one kernel doesn't graph")
      graph_runner = current_batch_devs[0].graph(current_batch, input_buffers, var_vals, orig_valid_positions=orig_valid_positions)
      # clear jit inputs to allow their memory to be freed/reused
      for (j,i) in graph_runner.input_replace.keys(): graph_runner.jit_cache[j].bufs[i] = None
      graphed_jit_cache.append(ExecItem(UOp(Ops.NOOP), cast(list[Buffer|None], input_buffers), prg=graph_runner))
      max_batch_size *= 2
      if DEBUG >= 2: print(f"JIT GRAPHing batch with {len(current_batch)} kernels on device {current_batch_devs[0]}")
    except GraphException as e:
      graphed_jit_cache.extend(current_batch)
      if DEBUG >= 2: print(f"JIT GRAPHing failed batch with {len(current_batch)} kernels on device {current_batch_devs[0]}: {e}")
    current_batch = []
    current_batch_devs = []

  for ji in jit_cache:
    match ji.prg:
      case CompiledRunner(): ji_graph_dev = ji.prg.dev
      case BufferXfer(): ji_graph_dev = Device[unwrap(ji.bufs[0]).device]
      case BufferCopy(): ji_graph_dev = next((Device[unwrap(b).device] for b in ji.bufs if unwrap(b).device != "CPU"), None)
      case ViewOp(): continue # ViewOps are just ignored
      case _: ji_graph_dev = None # Everything else is not graphed and flushes existing graph if it's being constructed

    # Check if this jit item can be graphed at all, so check if a new graph supports the current item.
    can_be_graphed = ji_graph_dev is not None and ji_graph_dev.graph is not None and graph_class(ji_graph_dev).supports_exec_item([ji_graph_dev], ji)

    # Check if the current batch can be extended with this item.
    can_share_graph = can_be_graphed and len(current_batch_devs) > 0 and \
                      graph_class(current_batch_devs[0]).supports_exec_item(dedup(current_batch_devs + [ji_graph_dev]), ji)
    can_extend_graph_batch = can_share_graph and (max_batch_size == 0 or len(current_batch) < max_batch_size)

    # Flush the current batch if any, since it can't be extended or is full.
    if not can_extend_graph_batch and len(current_batch) > 0: flush_batch()
    (current_batch if can_be_graphed else graphed_jit_cache).append(ji)
    current_batch_devs = dedup(current_batch_devs + [ji_graph_dev]) if can_be_graphed else []

  if len(current_batch) > 0: flush_batch()
  return graphed_jit_cache

def get_input_replace(jit_cache: list[ExecItem], input_buffers:list[Buffer],
                      orig_valid_positions: dict[int, set[int]]|None = None) -> dict[tuple[int, int], int]:
  input_replace: dict[tuple[int, int], int] = {}
  for j,ji in enumerate(jit_cache):
    for i,a in enumerate(ji.bufs):
      if a in input_buffers:
        # filter out positions that weren't valid inputs in the original capture (prevents aliasing bugs)
        if orig_valid_positions is not None and i not in orig_valid_positions.get(id(ji), set()): continue
        input_replace[(j,i)] = input_buffers.index(a)
  return input_replace

class GraphRunner(Runner):
  def __init__(self, jit_cache: list[ExecItem], input_buffers: list[Buffer], var_vals: dict[str, int],
               orig_valid_positions: dict[int, set[int]]|None = None):
    self.jit_cache = jit_cache  # NOTE: this is not used, but you have to keep these objects alive for the Graph
    self.input_replace:dict[tuple[int, int], int] = get_input_replace(jit_cache, input_buffers, orig_valid_positions)
    self.var_vals_replace:dict[int, list[tuple[int, int]]] = {}
    self.launch_dims_replace:dict[int, tuple[int|None, int|None]] = {}
    self.launch_dims_base:dict[int, tuple[tuple[int, ...], tuple[int, ...]]] = {}

    def is_sym_dim(dim) -> bool: return not all(isinstance(d, (int, float)) for d in dim)

    self.vars = sorted(var_vals.keys())
    self.symbolic_dims = dedup([tuple(d) for ji in jit_cache if isinstance(ji.prg, CompiledRunner) and (d:=ji.prg.p.local_size) and is_sym_dim(d)] +
                               [tuple(d) for ji in jit_cache if isinstance(ji.prg, CompiledRunner) and (d:=ji.prg.p.global_size) and is_sym_dim(d)])
    def find_symbolic_dim(dim): return self.symbolic_dims.index(tuple(dim)) if dim is not None and tuple(dim) in self.symbolic_dims else None

    estimates = Estimates()
    for j,ji in enumerate(jit_cache):
      assert ji.prg is not None
      estimates += ji.prg.estimates
      if isinstance(ji.prg, CompiledRunner):
        if (replace:=[(i, self.vars.index(v.expr)) for i, v in enumerate(ji.prg.p.vars) if v.expr not in ji.fixedvars | ji.prg.p.runtimevars]):
          self.var_vals_replace[j] = replace

        global_dim_idx, local_dim_idx = find_symbolic_dim(ji.prg.p.global_size), find_symbolic_dim(ji.prg.p.local_size)
        if global_dim_idx is not None or local_dim_idx is not None:
          self.launch_dims_replace[j] = (global_dim_idx, local_dim_idx)
          assert ji.prg.p.local_size is not None
          self.launch_dims_base[j] = (tuple(ji.prg.p.global_size), tuple(ji.prg.p.local_size))

    # used in MultiGraphRunner. tracks (offset, end, dep) ranges per base buffer id to handle suballocated buffers correctly.
    self.w_dependency_map: dict[int, list[tuple[int, int, Any]]] = collections.defaultdict(list)
    self.r_dependency_map: dict[int, list[tuple[int, int, Any]]] = collections.defaultdict(list)

    assert jit_cache[0].prg is not None
    super().__init__(colored(f"<batched {len(jit_cache)}>", "cyan"), jit_cache[0].prg.device.split(":")[0], estimates.simplify())

  def updated_vars(self, var_vals: dict[str, int]):
    vals = [var_vals[v] for v in self.vars]
    for j, vidxs in self.var_vals_replace.items():
      for i, v in vidxs: yield j, i, vals[v]

  def updated_launch_dims(self, var_vals: dict[str, int]):
    dims = [tuple(sym_infer(s, var_vals) for s in dim) for dim in self.symbolic_dims]
    for j, (gl, lc) in self.launch_dims_replace.items():
      yield j, (dims[gl] if gl is not None else self.launch_dims_base[j][0]), (dims[lc] if lc is not None else self.launch_dims_base[j][1])

  def _access_resources(self, bufs:list[Buffer], write:list[int], new_dependency:Any):
    wait_nodes = []
    for i,buf in enumerate(bufs):
      key, s, e = id(buf.base._buf), buf.offset, buf.offset + buf.nbytes
      wait_nodes += [dep for st,en,dep in self.w_dependency_map[key] if st < e and s < en]
      if i in write: wait_nodes += [dep for st,en,dep in self.r_dependency_map[key] if st < e and s < en]
    for i,buf in enumerate(bufs):
      key, s, e = id(buf.base._buf), buf.offset, buf.offset + buf.nbytes
      if i in write:
        for dmap in [self.w_dependency_map, self.r_dependency_map]:
          kept = []
          for st,en,dep in dmap[key]:
            if st < min(s, en): kept.append((st, min(s, en), dep))
            if max(e, st) < en: kept.append((max(e, st), en, dep))
          dmap[key] = kept
        self.w_dependency_map[key].append((s, e, new_dependency))
      else: self.r_dependency_map[key].append((s, e, new_dependency))
    return list({id(x):x for x in wait_nodes}.values())

  @staticmethod
  def supports_exec_item(devs:list[Compiled], ei:ExecItem) -> bool: return isinstance(ei.prg, CompiledRunner) and len(dedup(devs)) == 1

# a marker for your graph supporting multiple devices of the same type
class MultiGraphRunner(GraphRunner):
  @staticmethod
  def supports_exec_item(devs:list[Compiled], ei:ExecItem) -> bool:
    # Devices must be the same type
    return isinstance(ei.prg, (CompiledRunner, BufferXfer)) and len(dedup([type(Device[b.device]) for b in ei.bufs if b]+[type(d) for d in devs]))==1

def get_out_buffers_for_ei(ei:ExecItem) -> list[Buffer]:
  if isinstance(ei.prg, CompiledRunner): return [cast(Buffer, ei.bufs[out]) for out in ei.prg.p.outs if out not in ei.prg.p.ins]
  if isinstance(ei.prg, (BufferCopy, BufferXfer, EncDec)): return [cast(Buffer, ei.bufs[0])]
  return []

def update_depends(depends:set[Buffer|None], jit_cache:list[ExecItem]):
  for ei in jit_cache:
    if any(b in depends for b in ei.bufs): depends.update(get_out_buffers_for_ei(ei))

ReturnType = TypeVar('ReturnType')
@dataclass
class CapturedJit(Generic[ReturnType]):
  ret: Any  # includes the Tensors or any other returned object
  jit_cache: list[ExecItem]
  input_replace: dict[tuple[int, int], int]
  extra_view_inputs: list[tuple[int, int, str, int, DType]]
  expected_names: list[int|str]
  expected_input_info: list[tuple[UOp, tuple[Variable, ...], DType, str]]  # (view, variables, dtype, device) per input

  def __reduce__(self):
    # TODO: free_intermediates here?
    return self.__class__, (self.ret, self.jit_cache, self.input_replace, self.extra_view_inputs, self.expected_names, self.expected_input_info)

  def __post_init__(self):
    self._jit_cache: list[ExecItem] = self.jit_cache
    self._input_replace: dict[tuple[int, int], int] = self.input_replace
    self._first_run = True
    self._fast_jit_data: list|None = None  # pre-computed fast dispatch data
    self._alias_scratch: dict[int, Buffer] = {}
    # precompute read-after-write hazard detection
    self._output_to_writer = {b: j for j, ei in enumerate(self.jit_cache) for b in get_out_buffers_for_ei(ei)}
    self._input_to_max_reader: dict[int, int] = {}
    for (j, i), idx in self.input_replace.items():
      # only buffers that were different during capture but alias at jit time (e.g. feeding output back as input) need the copy.
      if self.jit_cache[j].bufs[i] not in get_out_buffers_for_ei(self.jit_cache[j]):
        self._input_to_max_reader[idx] = max(self._input_to_max_reader.get(idx, -1), j)
    self._clear_inputs()

  def _clear_inputs(self):
    for (j,i) in self._input_replace.keys(): self._jit_cache[j].bufs[i] = None

  def free_intermediates(self):
    depends: set[Buffer|None] = set([None])
    update_depends(depends, self.jit_cache)
    arenas = {b._base for b in depends if b is not None and b._base is not None}
    to_free = {b for b in depends if b is not None} | {b for ei in self.jit_cache for b in ei.bufs if b is not None and b._base in arenas}
    for b in to_free:
      if hasattr(b, '_buf'): b.deallocate()
    for a in arenas:
      if a.allocated_views == 0 and a.is_allocated(): a.deallocate()
    self.__post_init__()

  # jit exec
  def __call__(self, input_buffers:list[Buffer], var_vals:dict[str, int]) -> ReturnType:
    # assign inputs
    if not hasattr(self, '_view_cache'): self._view_cache: dict[int, Buffer] = {}
    for vi, (idx, offset, device, size, dtype) in enumerate(self.extra_view_inputs):
      cached = self._view_cache.get(vi)
      if cached is not None and cached._base is input_buffers[idx]:
        input_buffers.append(cached)
      else:
        buf = Buffer(device, size, dtype, base=input_buffers[idx], offset=offset).ensure_allocated()
        self._view_cache[vi] = buf
        input_buffers.append(buf)

    # copy aliased inputs to prevent read-after-write hazard
    for i, ib in enumerate(input_buffers):
      if (writer := self._output_to_writer.get(ib)) is not None and self._input_to_max_reader.get(i, -1) >= writer:
        scratch = self._alias_scratch.get(i)
        if scratch is None or scratch.size != ib.size or scratch.dtype != ib.dtype:
          scratch = Buffer(ib.device, ib.size, ib.dtype).ensure_allocated()
          self._alias_scratch[i] = scratch
        try:
          from tinygrad.runtime.autogen import opencl as _cl
          _cl.clEnqueueCopyBuffer(Device[ib.device].queue, ib._buf[0], scratch._buf[0], 0, 0, ib.nbytes, 0, None, None)
        except Exception:
          scratch.copyin(ib.as_memoryview())
        input_buffers[i] = scratch
    for (j,i),input_idx in self._input_replace.items(): self._jit_cache[j].bufs[i] = input_buffers[input_idx]

    # Condense the items into a graph executor.
    if self._first_run:
      # allocate intermediates if freed
      for ji in self.jit_cache:
        for b in ji.bufs:
          if b is not None: b.ensure_allocated()
      # create graph if needed
      if JIT < 2:
        # build a map from ExecItem object to the buffer positions that are valid inputs (from original input_replace)
        orig_valid_positions: dict[int, set[int]] = {}  # id(ExecItem) -> set of valid buffer indices
        for (j, i) in self.input_replace: orig_valid_positions.setdefault(id(self.jit_cache[j]), set()).add(i)
        self._jit_cache = apply_graph_to_jit(self.jit_cache, input_buffers, var_vals, orig_valid_positions, max_batch_size=JIT_BATCH_SIZE.value)
        # recompute input_replace: GraphRunner items have all positions valid, non-GraphRunner items use orig_valid_positions
        valid_positions = {id(ji): set(range(len(ji.bufs))) if isinstance(ji.prg, GraphRunner) else orig_valid_positions.get(id(ji), set())
                          for ji in self._jit_cache}
        self._input_replace = get_input_replace(self._jit_cache, input_buffers, valid_positions)
      self._first_run = False

    if DEBUG >= 1 and len(self._jit_cache) >= 10: print(f"jit execs {len(self._jit_cache)} kernels")
    if self._fast_jit_data is None and not (DEBUG >= 2) and not PROFILE and not getenv("NO_FAST_JIT"):
      self._fast_jit_data = self._build_fast_dispatch(var_vals)
    if self._fast_jit_data is not None:
      self._run_fast_dispatch(var_vals)
    else:
      for ei in self._jit_cache: ei.run(var_vals, jit=True)
    self._clear_inputs()
    return self.ret

  def _build_fast_dispatch(self, var_vals: dict[str, int]) -> list:
    """Build data for two-phase C dispatch: first frame sets all args, subsequent frames set only changed args."""
    import ctypes, os
    from tinygrad.dtype import ImageDType
    from tinygrad.runtime.autogen import opencl as cl
    lib_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'cl_dispatch.so')
    if not os.path.exists(lib_path): lib_path = '/data/openpilot/tinygrad_repo/cl_dispatch.so'
    try: clib = ctypes.CDLL(lib_path)
    except OSError: return None
    clib.dispatch_batch.restype = ctypes.c_int
    clib.dispatch_batch.argtypes = [
      ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_uint),
      ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_int),
      ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_void_p),
      ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
    clib.dispatch_fast.restype = ctypes.c_int
    clib.dispatch_fast.argtypes = [
      ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_uint),
      ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_int),
      ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_void_p)]
    queue = None
    kernels_list, ndims_list, gs_flat, ls_flat, has_local_list = [], [], [], [], []
    buf_counts_list, buf_indices_flat, buf_ei_map_flat = [], [], []
    val_counts_list, val_indices_flat, val_values_flat = [], [], []
    non_kernel_items = []
    kernel_ei_refs = []
    buf_offset, val_offset = 0, 0
    buf_offsets_list, val_offsets_list = [], []
    image_args: dict[int, tuple[ImageDType, Any]] = {}
    image_cache: dict[tuple[int, tuple[int, ...], int], Any] = {}

    def get_image_arg(buf, dtype:ImageDType, dev):
      key = (ctypes.cast(buf._buf, ctypes.c_void_p).value, dtype.shape, dtype.itemsize)
      if (img := image_cache.get(key)) is None:
        fmt = cl.cl_image_format(cl.CL_RGBA, {2:cl.CL_HALF_FLOAT, 4:cl.CL_FLOAT}[dtype.itemsize])
        desc = cl.cl_image_desc(cl.CL_MEM_OBJECT_IMAGE2D, dtype.shape[1], dtype.shape[0], image_row_pitch=dtype.pitch, buffer=buf._buf)
        img = cl.clCreateImage(dev.context, cl.CL_MEM_READ_WRITE, fmt, desc, None, status:=ctypes.c_int32())
        if status.value != 0: raise RuntimeError(f"OpenCL image arg creation failed: {status.value}")
        image_cache[key] = img
      return img

    for ei in self._jit_cache:
      if isinstance(ei.prg, CompiledRunner) and ei.prg.p.local_size is None and Device[ei.prg.p.device].renderer.has_local and all_int(ei.prg.p.global_size):
        try: ei.run(var_vals, wait=True, jit=True, do_update_stats=False)
        except Exception as e:
          if DEBUG >= 2: print(f"fast dispatch local-size warmup failed for {ei.prg.display_name}: {type(e).__name__}: {e}")
    for j, ei in enumerate(self._jit_cache):
      if not isinstance(ei.prg, CompiledRunner):
        non_kernel_items.append((j, ei))
        continue
      cl_prg = ei.prg._prg
      if queue is None: queue = cl_prg.dev.queue
      merged = var_vals | ei.fixedvars if ei.fixedvars else var_vals
      gs, ls = ei.prg.p.launch_dims(merged)
      if ls is not None: gs = [int(g*l) for g,l in zip(gs, ls)]
      ndim = len(gs)
      kernels_list.append(ctypes.cast(cl_prg.kernel, ctypes.c_void_p).value)
      ndims_list.append(ndim)
      gs_flat.extend([gs[0], gs[1] if ndim > 1 else 1, gs[2] if ndim > 2 else 1])
      if ls: ls_flat.extend([ls[0], ls[1] if ndim > 1 else 0, ls[2] if ndim > 2 else 0])
      else: ls_flat.extend([0, 0, 0])
      has_local_list.append(1 if ls else 0)
      glob_idx = ei.prg.p.globals
      bi, bm = [], []
      for k, ei_bufs_idx in enumerate(glob_idx):
        for real_i, dt in cl_prg.arg_dtypes[k]:
          bi.append(real_i)
          bm.append(ei_bufs_idx)
          if isinstance(dt, ImageDType): image_args[buf_offset + len(bi) - 1] = (dt, cl_prg.dev)
      buf_offsets_list.append(buf_offset)
      buf_counts_list.append(len(bi))
      buf_indices_flat.extend(bi)
      buf_ei_map_flat.extend(bm)
      buf_offset += len(bi)
      vi, vv = [], []
      if ei.prg.p.vars:
        for vii, v in enumerate(ei.prg.p.vars):
          vi.append(len(glob_idx) + vii)
          vv.append(merged.get(v.expr, v.vmin))
      val_offsets_list.append(val_offset)
      val_counts_list.append(len(vi))
      val_indices_flat.extend(vi)
      val_values_flat.extend(vv)
      val_offset += len(vi)
      kernel_ei_refs.append(ei)
    n_k = len(kernels_list)
    n_b = len(buf_indices_flat)
    n_v = len(val_indices_flat)
    c_kernels = (ctypes.c_void_p * n_k)(*kernels_list)
    c_ndims = (ctypes.c_uint * n_k)(*ndims_list)
    c_gs = (ctypes.c_size_t * (n_k*3))(*gs_flat)
    c_ls = (ctypes.c_size_t * (n_k*3))(*ls_flat)
    c_has_local = (ctypes.c_int * n_k)(*has_local_list)
    c_buf_counts = (ctypes.c_int * n_k)(*buf_counts_list)
    c_buf_offsets = (ctypes.c_int * n_k)(*buf_offsets_list)
    c_buf_indices = (ctypes.c_int * max(n_b, 1))(*buf_indices_flat)
    c_buf_values = (ctypes.c_void_p * max(n_b, 1))()
    for bp in range(n_b):
      k_idx = 0
      while k_idx < n_k - 1 and buf_offsets_list[k_idx+1] <= bp: k_idx += 1
      b = kernel_ei_refs[k_idx].bufs[buf_ei_map_flat[bp]]
      if b is None: c_buf_values[bp] = 0
      elif bp in image_args:
        dtype, dev = image_args[bp]
        c_buf_values[bp] = ctypes.cast(get_image_arg(b, dtype, dev), ctypes.c_void_p).value
      else:
        c_buf_values[bp] = ctypes.cast(b._buf, ctypes.c_void_p).value
    changing = set(self._input_replace.keys())
    jit_to_kernel = {}
    ki = 0
    for j, ei in enumerate(self._jit_cache):
      if isinstance(ei.prg, CompiledRunner):
        jit_to_kernel[j] = ki
        ki += 1
    # build arrays for dispatch_fast: (kernel_index_in_c, cl_arg_index, ei_ref_k_idx, ei_bufs_idx, flat buf arg index)
    changing_entries = []
    for bp in range(n_b):
      k_idx = 0
      while k_idx < n_k - 1 and buf_offsets_list[k_idx+1] <= bp: k_idx += 1
      ei_bufs_idx = buf_ei_map_flat[bp]
      for jj, kk in jit_to_kernel.items():
        if kk == k_idx:
          if (jj, ei_bufs_idx) in changing:
            changing_entries.append((k_idx, buf_indices_flat[bp], k_idx, ei_bufs_idx, bp))
          break
    n_ch = len(changing_entries)
    c_ch_kernel_indices = (ctypes.c_int * max(n_ch, 1))(*[e[0] for e in changing_entries])
    c_ch_arg_indices = (ctypes.c_int * max(n_ch, 1))(*[e[1] for e in changing_entries])
    c_ch_values = (ctypes.c_void_p * max(n_ch, 1))()
    c_val_counts = (ctypes.c_int * n_k)(*val_counts_list)
    c_val_offsets = (ctypes.c_int * n_k)(*val_offsets_list)
    c_val_indices = (ctypes.c_int * max(n_v, 1))(*val_indices_flat)
    c_val_values = (ctypes.c_int * max(n_v, 1))(*val_values_flat)
    queue_ptr = ctypes.cast(queue, ctypes.c_void_p).value
    # store ei_ref info for changing entries
    ch_ei_info = [(e[2], e[3], e[4]) for e in changing_entries]
    return [clib, queue_ptr, n_k, c_kernels, c_ndims, c_gs, c_ls, c_has_local,
            c_buf_counts, c_buf_offsets, c_buf_indices, c_buf_values, kernel_ei_refs,
            c_val_counts, c_val_offsets, c_val_indices, c_val_values, non_kernel_items,
            n_ch, c_ch_kernel_indices, c_ch_arg_indices, c_ch_values, ch_ei_info,
            image_args, image_cache, True]  # last element: needs_full_init

  def _run_fast_dispatch(self, var_vals: dict[str, int]):
    import ctypes
    data = self._fast_jit_data
    (clib, queue_ptr, n_k, c_kernels, c_ndims, c_gs, c_ls, c_has_local,
     c_buf_counts, c_buf_offsets, c_buf_indices, c_buf_values, kernel_ei_refs,
     c_val_counts, c_val_offsets, c_val_indices, c_val_values, non_kernel_items,
     n_ch, c_ch_kernel_indices, c_ch_arg_indices, c_ch_values, ch_ei_info,
     image_args, image_cache, needs_full_init) = data
    for j, ei in non_kernel_items:
      ei.run(var_vals, jit=True)
    _cast = ctypes.cast
    _cvp = ctypes.c_void_p
    def get_image_arg(buf, dtype, dev):
      key = (_cast(buf._buf, _cvp).value, dtype.shape, dtype.itemsize)
      if (img := image_cache.get(key)) is None:
        from tinygrad.runtime.autogen import opencl as cl
        fmt = cl.cl_image_format(cl.CL_RGBA, {2:cl.CL_HALF_FLOAT, 4:cl.CL_FLOAT}[dtype.itemsize])
        desc = cl.cl_image_desc(cl.CL_MEM_OBJECT_IMAGE2D, dtype.shape[1], dtype.shape[0], image_row_pitch=dtype.pitch, buffer=buf._buf)
        img = cl.clCreateImage(Device["CL"].context, cl.CL_MEM_READ_WRITE, fmt, desc, None, status:=ctypes.c_int32())
        if status.value != 0: raise RuntimeError(f"OpenCL image arg creation failed: {status.value}")
        image_cache[key] = img
      return img

    for i, (k_idx, ei_bufs_idx, bp) in enumerate(ch_ei_info):
      b = kernel_ei_refs[k_idx].bufs[ei_bufs_idx]
      if b is None: c_buf_values[bp] = 0
      elif bp in image_args:
        dtype, dev = image_args[bp]
        c_buf_values[bp] = _cast(get_image_arg(b, dtype, dev), _cvp).value
      else:
        c_buf_values[bp] = _cast(b._buf, _cvp).value
    if not hasattr(clib, '_smart_setup'):
      clib.dispatch_smart.restype = ctypes.c_int
      clib.dispatch_smart.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_uint),
        ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
      clib._smart_setup = True
    err = clib.dispatch_smart(queue_ptr, n_k, c_kernels, c_ndims, c_gs, c_ls, c_has_local,
                              c_buf_counts, c_buf_offsets, c_buf_indices, c_buf_values,
                              None,
                              c_val_counts, c_val_offsets, c_val_indices, c_val_values)
    if err != 0: raise RuntimeError(f"dispatch_smart error: {err}")
    if Device.DEFAULT == "CL":
      from tinygrad.runtime.autogen import opencl as cl
      cl.clFlush(Device["CL"].queue)


def _prepare_jit_inputs(args, kwargs):
  input_tensors: list[tuple[int|str, Tensor]] = [(name,t) for name,t in list(enumerate(args))+sorted(kwargs.items()) if t.__class__ is Tensor]
  names, tensors = [name for name,_ in input_tensors], [t for _,t in input_tensors]
  # extract tensors from containers (shallow, not recursive to avoid grabbing model weights)
  for x in args + tuple(kwargs.values()):
    it = x if isinstance(x, (tuple,list)) else x.values() if isinstance(x, dict) else []
    tensors += [t for t in it if t.__class__ is Tensor and not any(t is y for y in tensors)]
  if len(unrealized_tensors := [x for x in tensors if not x.uop.is_realized]): Tensor.realize(*unrealized_tensors)
  input_uops: list[UOp] = flatten([t.uop.src if t.uop.op is Ops.MULTI else [t.uop] for t in tensors])
  if any(u.base.op is Ops.CONST for u in input_uops):
    raise JitError("JIT inputs cannot be const, create a buffer with .contiguous()")
  input_buffers: list[Buffer] = flatten([b.bufs if isinstance(b, MultiBuffer) else [b] for u in input_uops if (b:=u.base.realized) is not None])
  if len(set(input_buffers)) != len(input_buffers): raise JitError("duplicate inputs to JIT")
  inputs = [(*(u.substitute({u.base:UOp(Ops.NOOP)}, extra_pm=mop_cleanup).unbind_all()), u.dtype, u.device) for u in input_uops]
  _var_vals = merge_dicts([x[1] for x in inputs] + [dict(v.unbind() for v in (args + tuple(kwargs.values())) if isinstance(v, UOp))])
  var_vals = {k.expr:v for k,v in _var_vals.items()}
  expected_input_info = [(x[0], tuple(sorted(x[1].keys(), key=lambda v: v.expr)), x[2], x[3]) for x in inputs]
  return input_buffers, var_vals, names, expected_input_info

class TinyJit(Generic[ReturnType]):
  def __init__(self, fxn:Callable[..., ReturnType]|None, captured:CapturedJit|None=None, prune=False):
    assert fxn or captured, "need either a function or a CapturedJit"
    self.fxn = fxn
    self.captured: CapturedJit|None = captured
    self.cnt: int = 2 if self.fxn is None else 0
    self.prune = prune

  def add_linear(self, linear:UOp, var_vals:dict[str, int]): self._linears.append(linear)

  def reset(self):
    assert self.fxn is not None, "can't reset without function"
    self.cnt = 0
    self.captured = None

  def __reduce__(self):
    assert self.captured is not None, "can't pickle an uncaptured JIT"
    return self.__class__, (None, self.captured)

  # keep legacy code working
  @property
  def jit_cache(self) -> list[ExecItem]: return self.captured._jit_cache if self.captured is not None else []
  @property
  def input_replace(self) -> dict[tuple[int, int], int]: return self.captured._input_replace if self.captured is not None else {}

  def __get__(self, obj, objtype): return functools.partial(self.__call__, obj) # add support for instance methods

  def _fast_prepare_inputs(self, args, kwargs):
    """Fast path for JIT exec: extract buffers directly, skip UOp operations."""
    assert self.captured is not None
    # extract tensors in same order as _prepare_jit_inputs
    input_tensors = [(name,t) for name,t in list(enumerate(args))+sorted(kwargs.items()) if t.__class__ is Tensor]
    names = [name for name,_ in input_tensors]
    tensors = [t for _,t in input_tensors]
    for x in args + tuple(kwargs.values()):
      it = x if isinstance(x, (tuple,list)) else x.values() if isinstance(x, dict) else []
      tensors += [t for t in it if t.__class__ is Tensor and not any(t is y for y in tensors)]
    # extract buffers directly (skip UOp substitute/unbind which is expensive)
    input_buffers = []
    for t in tensors:
      if t.uop.op is Ops.MULTI:
        for u in t.uop.src:
          if u.base.realized is not None:
            b = u.base.realized
            if isinstance(b, MultiBuffer): input_buffers.extend(b.bufs)
            else: input_buffers.append(b)
      else:
        if t.uop.base.realized is not None:
          b = t.uop.base.realized
          if isinstance(b, MultiBuffer): input_buffers.extend(b.bufs)
          else: input_buffers.append(b)
    # return cached expected_input_info (it doesn't change between calls)
    return input_buffers, {}, names, self.captured.expected_input_info

  def __call__(self, *args, **kwargs) -> ReturnType:
    input_buffers, var_vals, names, expected_input_info = _prepare_jit_inputs(args, kwargs)
    if not JIT or self.cnt == 0:
      # jit ignore
      assert self.fxn is not None
      with Context(BEAM=0 if getenv("IGNORE_JIT_FIRST_BEAM") else BEAM.value):
        ret = self.fxn(*args, **kwargs)
        if len(params:=get_parameters(ret)): Tensor.realize(*params)
    elif self.cnt == 1:
      # jit capture
      assert self.fxn is not None
      if capturing: raise RuntimeError(f"having TinyJit inside another TinyJit is not supported {len(capturing)=} {capturing=}")
      self._linears: list[UOp] = []
      capturing.append(self)
      try:
        ret = self.fxn(*args, **kwargs)
        if len(params:=get_parameters(ret)): Tensor.realize(*params)
      finally: capturing.clear()
      if not len(self._linears): raise JitError("didn't JIT anything!")
      _check_no_non_tensor_return(ret)
      if DEBUG >= 1: print(f"JIT captured {len(self._linears)} linears with {len(input_buffers)} inputs")

      # combine all captured linears into one, memory plan, and convert to ExecItems
      big_linear = UOp(Ops.LINEAR, src=tuple(flatten([l.src for l in self._linears])))
      del self._linears

      if self.prune:
        big_linear, onetime_linear = prune_linear(big_linear, {k for k,v in buffers.items() if isinstance(v, Buffer) and v in set(input_buffers)})
        if DEBUG >= 1: print(f"pruned from {len(big_linear.src) + len(onetime_linear.src)} -> {len(big_linear.src)} kernels")
        for ei in (si.lower() for si in linear_to_schedule(onetime_linear)):
          for b in ei.bufs: cast(Buffer, b).ensure_allocated()
          ei.run(var_vals, jit=True)

      held_bufs = set(buffers) | {t.uop.buf_uop for t in get_parameters(ret) if t.uop.buf_uop.op is Ops.BUFFER}
      with Context(BEAM=getenv("JITBEAM", BEAM.value)):
        jit_cache = jit_lower(big_linear, held_bufs)

      # track inputs that are views of buffers
      # TODO: eventually expected_buffers should live in ExecItem
      extra_view_inputs: list[tuple[int, int, str, int, DType]] = []
      for item in jit_cache:
        for b in item.bufs:
          if b is not None and b._base is not None and b._base in input_buffers:
            input_buffers.append(b)
            extra_view_inputs.append((input_buffers.index(b.base), b.offset, b.device, b.size, b.dtype))

      input_replace = get_input_replace(jit_cache, input_buffers)
      if DEBUG >= 1 and len(set(input_replace.values())) != len(input_buffers): print("WARNING: some input tensors not found")

      # exec
      for ei in jit_cache: ei.run(var_vals)

      self.captured = CapturedJit(ret, jit_cache, input_replace, extra_view_inputs, names, expected_input_info)
    elif self.cnt >= 2:
      # jit exec
      assert self.captured is not None
      if self.captured.expected_names != names: raise JitError(f"args mismatch in JIT: {self.captured.expected_names=} != {names}")
      if self.captured.expected_input_info != expected_input_info:
        raise JitError(f"args mismatch in JIT: {self.captured.expected_input_info=} != {expected_input_info=}")
      ret = self.captured(input_buffers, var_vals)

    self.cnt += 1
    return ret
