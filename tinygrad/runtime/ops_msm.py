from __future__ import annotations
import os, ctypes, functools, mmap, struct, array, math, time, glob
assert os.name == 'posix'
from dataclasses import dataclass
from typing import Any, cast
from tinygrad.device import Buffer, BufferSpec, Compiled, CompilerSet, LRUAllocator
from tinygrad.runtime.support.hcq import FileIOInterface
from tinygrad.runtime.autogen import msm_drm, mesa
from tinygrad.renderer.nir import IR3Renderer
from tinygrad.engine.jit import GraphRunner
from tinygrad.engine.realize import CompiledRunner
from tinygrad.helpers import mv_address, to_mv, round_up, data64_le, next_power2, flatten

# PM4 packet helpers (shared with ops_qcom.py)
def _parity(val: int):
  for i in range(4, 1, -1): val ^= val >> (1 << i)
  return (~0x6996 >> (val & 0xf)) & 1

def _pkt7(opcode: int, cnt: int): return mesa.CP_TYPE7_PKT | cnt & 0x3FFF | _parity(cnt) << 15 | (opcode & 0x7F) << 16 | _parity(opcode) << 23
def _pkt4(reg: int, cnt: int): return mesa.CP_TYPE4_PKT | cnt & 0x7F | _parity(cnt) << 7 | (reg & 0x3FFFF) << 8 | _parity(reg) << 27

def _qreg_exec(__reg, __val=0, **kwargs):
  for k, v in kwargs.items():
    reg_name = f"{__reg[4:]}_{k.removeprefix('_').upper()}"
    __val |= (getattr(mesa, reg_name) if v else 0) if type(v) is bool else (v << getattr(mesa, f'{reg_name}__SHIFT'))
  return __val
qreg: Any = type("QREG", (object,), {name[4:].lower(): functools.partial(_qreg_exec, name) for name in mesa.__dict__.keys() if name[:4] == 'REG_'})

def _ctz(v): return (v & -v).bit_length() - 1

@dataclass
class MSMBuffer:
  handle: int     # GEM handle
  size: int       # allocation size
  iova: int       # GPU virtual address
  cpu_addr: int   # CPU mmap address
  mmap_size: int  # mmap'd size

class MSMAllocator(LRUAllocator['MSMDevice']):
  def __init__(self, dev: 'MSMDevice', **kwargs):
    super().__init__(dev, **kwargs)
    self._all_buffers: list[MSMBuffer] = []  # track all GEM BOs, only close on finalize

  def _alloc(self, size: int, options: BufferSpec) -> MSMBuffer:
    alloc_size = round_up(size, 0x1000)
    flags = msm_drm.MSM_BO_WC
    if options.cpu_access: flags = msm_drm.MSM_BO_CACHED_COHERENT
    gem = msm_drm.DRM_IOCTL_MSM_GEM_NEW(self.dev.fd, size=alloc_size, flags=flags)
    handle = gem.handle
    info = msm_drm.DRM_IOCTL_MSM_GEM_INFO(self.dev.fd, handle=handle, info=msm_drm.MSM_INFO_GET_OFFSET)
    cpu_addr = self.dev.drm_fd.mmap(0, alloc_size, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, info.value)
    info2 = msm_drm.DRM_IOCTL_MSM_GEM_INFO(self.dev.fd, handle=handle, info=msm_drm.MSM_INFO_GET_IOVA)
    buf = MSMBuffer(handle=handle, size=size, iova=info2.value, cpu_addr=cpu_addr, mmap_size=alloc_size)
    self._all_buffers.append(buf)
    return buf

  def _free(self, opaque: MSMBuffer, options: BufferSpec):
    # never close GEM handles during runtime, IOMMU unmapping races cause GPU translation faults
    # GEM BOs are closed in finalize() or when the fd is closed at process exit
    pass

  def finalize(self):
    for buf in self._all_buffers:
      if buf.mmap_size > 0: FileIOInterface.munmap(buf.cpu_addr, buf.mmap_size)
      msm_drm.DRM_IOCTL_GEM_CLOSE(self.dev.fd, handle=buf.handle)
    self._all_buffers.clear()

  def _copyin(self, dest: MSMBuffer, src: memoryview):
    ctypes.memmove(dest.cpu_addr, mv_address(src), src.nbytes)

  def _copyout(self, dest: memoryview, src: MSMBuffer):
    self.dev.synchronize()
    ctypes.memmove(mv_address(dest), src.cpu_addr, dest.nbytes)

  def _as_buffer(self, src: MSMBuffer) -> memoryview:
    self.dev.synchronize()
    return to_mv(src.cpu_addr, src.size)

  def _offset(self, buf: MSMBuffer, size: int, offset: int) -> MSMBuffer:
    return MSMBuffer(handle=buf.handle, size=size, iova=buf.iova + offset, cpu_addr=buf.cpu_addr + offset, mmap_size=0)

def _build_pm4(prg: MSMProgram, args_buf: MSMBuffer, global_size, local_size) -> list[int]:
  """Build PM4 command stream for a compute dispatch. Same register programming as QCOMComputeQueue.exec()."""
  q: list[int] = []
  def cmd(opcode, *vals): q.extend([_pkt7(opcode, len(vals)), *vals])
  def reg(r, *vals): q.extend([_pkt4(r, len(vals)), *vals])

  def cast_int(x, ceil=False): return (math.ceil(x) if ceil else int(x)) if isinstance(x, float) else x
  global_size_mp = [cast_int(g * l) for g, l in zip(global_size, local_size)]

  # invalidate caches so GPU sees fresh data from CPU writes
  cmd(mesa.CP_WAIT_FOR_IDLE)
  cmd(mesa.CP_EVENT_WRITE, mesa.CACHE_INVALIDATE)
  cmd(mesa.CP_WAIT_MEM_WRITES)
  cmd(mesa.CP_WAIT_FOR_IDLE)

  # set compute mode
  cmd(mesa.CP_SET_MARKER, qreg.a6xx_cp_set_marker_0(mode=mesa.RM6_COMPUTE))
  reg(mesa.REG_A6XX_SP_UPDATE_CNTL, qreg.a6xx_sp_update_cntl(cs_state=True, cs_uav=True))
  reg(mesa.REG_A6XX_SP_UPDATE_CNTL, 0x0)
  reg(mesa.REG_A6XX_SP_CS_TSIZE, qreg.a6xx_sp_cs_tsize(0x80))
  reg(mesa.REG_A6XX_SP_CS_USIZE, qreg.a6xx_sp_cs_usize(0x40))
  reg(mesa.REG_A6XX_SP_MODE_CNTL, qreg.a6xx_sp_mode_cntl(isammode=mesa.ISAMMODE_GL))
  reg(mesa.REG_A6XX_SP_PERFCTR_SHADER_MASK, qreg.a6xx_sp_perfctr_shader_mask(cs=True))
  reg(mesa.REG_A6XX_TPL1_MODE_CNTL, qreg.a6xx_tpl1_mode_cntl(isammode=mesa.ISAMMODE_GL))
  reg(mesa.REG_A6XX_TPL1_DBG_ECO_CNTL, 0)
  cmd(mesa.CP_WAIT_FOR_IDLE)

  # work dimensions
  reg(mesa.REG_A6XX_SP_CS_NDRANGE_0,
      qreg.a6xx_sp_cs_ndrange_0(kerneldim=3, localsizex=local_size[0] - 1, localsizey=local_size[1] - 1, localsizez=local_size[2] - 1),
      global_size_mp[0], 0, global_size_mp[1], 0, global_size_mp[2], 0, 0xccc0cf, 0xfc | qreg.a6xx_sp_cs_wge_cntl(threadsize=mesa.THREAD64),
      cast_int(global_size[0], ceil=True), cast_int(global_size[1], ceil=True), cast_int(global_size[2], ceil=True))

  # shader program config
  reg(mesa.REG_A6XX_SP_CS_CNTL_0,
      qreg.a6xx_sp_cs_cntl_0(threadsize=mesa.THREAD64, halfregfootprint=prg.hregs, fullregfootprint=prg.fregs, branchstack=prg.brnchstck),
      qreg.a6xx_sp_cs_cntl_1(constantrammode=mesa.CONSTLEN_256, shared_size=prg.shared_size), 0, prg.prg_offset,
      *data64_le(prg.lib_buf.iova),
      qreg.a6xx_sp_cs_pvt_mem_param(memsizeperitem=prg.pvtmem_size_per_item), *data64_le(prg.dev._stack.iova),
      qreg.a6xx_sp_cs_pvt_mem_size(totalpvtmemsize=prg.pvtmem_size_total))

  # write workgroup size to args buffer if needed
  if prg.wgsz != 0xfc:
    to_mv(args_buf.cpu_addr + prg.wgsz * 4, 12)[:] = struct.pack("III", *local_size)

  # load shader constants and code
  cmd(mesa.CP_LOAD_STATE6_FRAG, qreg.cp_load_state6_0(state_type=mesa.ST_CONSTANTS, state_src=mesa.SS6_INDIRECT,
                                                       state_block=mesa.SB6_CS_SHADER, num_unit=1024 // 4),
      *data64_le(args_buf.iova))
  cmd(mesa.CP_LOAD_STATE6_FRAG, qreg.cp_load_state6_0(state_type=mesa.ST_SHADER, state_src=mesa.SS6_INDIRECT,
                                                       state_block=mesa.SB6_CS_SHADER, num_unit=round_up(prg.image_size, 128) // 128),
      *data64_le(prg.lib_buf.iova))

  reg(mesa.REG_A6XX_SP_REG_PROG_ID_0, 0xfcfcfcfc, 0xfcfcfcfc, 0xfcfcfcfc, 0xfc, qreg.a6xx_sp_cs_const_config(constlen=1024 // 4, enabled=True))
  reg(mesa.REG_A6XX_SP_CS_PVT_MEM_STACK_OFFSET, qreg.a6xx_sp_cs_pvt_mem_stack_offset(prg.hw_stack_offset))
  reg(mesa.REG_A6XX_SP_CS_INSTR_SIZE, qreg.a6xx_sp_cs_instr_size(prg.image_size // 4))

  # sampler state
  if prg.samp_cnt > 0:
    cmd(mesa.CP_LOAD_STATE6_FRAG, qreg.cp_load_state6_0(state_type=mesa.ST_SHADER, state_src=mesa.SS6_INDIRECT,
                                                         state_block=mesa.SB6_CS_TEX, num_unit=prg.samp_cnt),
        *data64_le(args_buf.iova + prg.samp_off))
    reg(mesa.REG_A6XX_SP_CS_SAMPLER_BASE, *data64_le(args_buf.iova + prg.samp_off))
    reg(mesa.REG_A6XX_TPL1_CS_BORDER_COLOR_BASE, *data64_le(prg.dev.border_color_buf.iova))

  # texture state
  if prg.tex_cnt > 0:
    cmd(mesa.CP_LOAD_STATE6_FRAG, qreg.cp_load_state6_0(state_type=mesa.ST_CONSTANTS, state_src=mesa.SS6_INDIRECT,
                                                         state_block=mesa.SB6_CS_TEX, num_unit=min(16, prg.tex_cnt)),
        *data64_le(args_buf.iova + prg.tex_off))
    reg(mesa.REG_A6XX_SP_CS_TEXMEMOBJ_BASE, *data64_le(args_buf.iova + prg.tex_off))

  # UAV/image state
  if prg.ibo_cnt > 0:
    cmd(mesa.CP_LOAD_STATE6_FRAG, qreg.cp_load_state6_0(state_type=mesa.ST6_UAV, state_src=mesa.SS6_INDIRECT,
                                                         state_block=mesa.SB6_CS_SHADER, num_unit=prg.ibo_cnt),
        *data64_le(args_buf.iova + prg.ibo_off))
    reg(mesa.REG_A6XX_SP_CS_UAV_BASE, *data64_le(args_buf.iova + prg.ibo_off))

  # compute config
  reg(mesa.REG_A6XX_SP_CS_CONFIG,
      qreg.a6xx_sp_cs_config(enabled=True, nsamp=prg.samp_cnt, ntex=prg.tex_cnt, nuav=prg.ibo_cnt))

  # dispatch (NIR path only)
  reg(mesa.REG_A6XX_SP_CS_CONST_CONFIG_0,
      qreg.a6xx_sp_cs_const_config_0(wgidconstid=prg.wgid, wgsizeconstid=prg.wgsz, wgoffsetconstid=0xfc, localidregid=prg.lid),
      qreg.a6xx_sp_cs_wge_cntl(linearlocalidregid=0xfc, threadsize=mesa.THREAD64))
  cmd(mesa.CP_EXEC_CS, 0,
      qreg.cp_exec_cs_1(ngroups_x=global_size[0]), qreg.cp_exec_cs_2(ngroups_y=global_size[1]), qreg.cp_exec_cs_3(_ngroups_z=global_size[2]))

  # full cache flush: write-back, invalidate, wait for memory writes, wait for idle
  cmd(mesa.CP_EVENT_WRITE, mesa.CACHE_FLUSH_TS, *data64_le(prg.dev.dummy_buf.iova), 0)
  cmd(mesa.CP_EVENT_WRITE, mesa.CACHE_INVALIDATE)
  cmd(mesa.CP_WAIT_MEM_WRITES)
  cmd(mesa.CP_WAIT_FOR_IDLE)

  return q

class MSMProgram:
  def __init__(self, dev: MSMDevice, name: str, lib: bytes, buf_dtypes=[], **kwargs):
    from tinygrad.runtime.support.compiler_mesa import IR3Compiler

    self.dev, self.name, self.buf_dtypes = dev, name, buf_dtypes

    # unpack IR3-compiled shader (same as QCOMProgram NIR path)
    v, cs, imm_vals, self.image = IR3Compiler.unpack_lib(lib)
    self.prg_offset, self.brnchstck, self.image_size, self.pvtmem, self.shmem = 0, v.branchstack, v.info.size, v.pvtmem_size, v.shared_size
    self.wgsz = alloc.offset_vec4 * 4 + 8 if (alloc := cs.allocs.consts[mesa.IR3_CONST_ALLOC_DRIVER_PARAMS]).size_vec4 else 0xfc
    self.wgid, self.lid = v.cs.work_group_id, v.cs.local_invocation_id
    self.buf_off, imm_off = cs.ubo_state.range[0].offset, cs.allocs.max_const_offset_vec4 * 16
    self.consts_info = [(struct.unpack_from("<I", imm_vals, i)[0], imm_off + i, 4) for i in range(0, len(imm_vals), 4)]

    self.samp_cnt, self.tex_cnt, self.ibo_cnt = (nt := v.image_mapping.num_tex), nt, v.num_uavs - nt
    self.tex_to_image = v.image_mapping.tex_to_image[:]
    self.samplers = [qreg.a6xx_tex_samp_0(wrap_s=(clamp_mode := mesa.A6XX_TEX_CLAMP_TO_BORDER), wrap_t=clamp_mode, wrap_r=clamp_mode),
                     qreg.a6xx_tex_samp_1(unnorm_coords=True, cubemapseamlessfiltoff=True), 0, 0] * self.samp_cnt

    self.tex_off, self.ibo_off, self.samp_off = 2048, 2048 + 0x40 * self.tex_cnt, 2048 + 0x40 * (self.tex_cnt + self.ibo_cnt)
    self.fregs, self.hregs = v.info.max_reg + 1, v.info.max_half_reg + 1

    # allocate GEM BO for shader code
    self.lib_buf: MSMBuffer = dev.allocator.alloc(self.image_size)
    to_mv(self.lib_buf.cpu_addr, self.image_size)[:] = self.image

    # compute derived sizes
    self.pvtmem_size_per_item: int = round_up(self.pvtmem, 512) >> 9
    self.pvtmem_size_total: int = self.pvtmem_size_per_item * 128 * 2
    self.hw_stack_offset: int = round_up(next_power2(round_up(self.pvtmem, 512)) * 128 * 16, 0x1000)
    self.shared_size: int = max(1, (self.shmem - 1) // 1024)
    self.max_threads = min(1024, ((384 * 32) // (max(1, (self.fregs + round_up(self.hregs, 2) // 2)) * 128)) * 128)
    dev._ensure_stack_size(self.hw_stack_offset * 4)

    self.kernargs_alloc_size = round_up(2048 + (self.tex_cnt + self.ibo_cnt) * 0x40 + len(self.samplers) * 4, 0x100)

  def __del__(self):
    if hasattr(self, 'lib_buf'): self.dev.allocator.free(self.lib_buf, self.lib_buf.size)

  def __call__(self, *bufs: MSMBuffer, global_size: tuple[int, int, int] = (1, 1, 1), local_size: tuple[int, int, int] = (1, 1, 1),
               vals: tuple[int, ...] = (), wait=False, **kw) -> float | None:
    from tinygrad.dtype import ImageDType
    from tinygrad.helpers import prod

    if self.max_threads < prod(local_size): raise RuntimeError("Too many resources requested for launch")

    # allocate args buffer and fill it
    args_buf: MSMBuffer = self.dev.allocator.alloc(self.kernargs_alloc_size)
    ctypes.memset(args_buf.cpu_addr, 0, self.kernargs_alloc_size)

    ubos = [b for i, b in enumerate(bufs) for _, dt in self.buf_dtypes[i] if not isinstance(dt, ImageDType)]
    uavs = [(dt, b) for i, b in enumerate(bufs) for _, dt in self.buf_dtypes[i] if isinstance(dt, ImageDType)]
    ibos, texs = uavs[:self.ibo_cnt], [uavs[self.ibo_cnt + self.tex_to_image[i]] for i in range(self.tex_cnt)]

    # write constants
    for cnst_val, cnst_off, cnst_sz in self.consts_info:
      to_mv(args_buf.cpu_addr + cnst_off, cnst_sz)[:] = cnst_val.to_bytes(cnst_sz, byteorder='little')

    # write sampler descriptors
    if self.samp_cnt > 0: to_mv(args_buf.cpu_addr + self.samp_off, len(self.samplers) * 4).cast('I')[:] = array.array('I', self.samplers)

    # pre-fill UBO region with dummy IOVA to prevent null-pointer GPU faults from unused slots
    dummy_iova = self.dev.dummy_buf.iova
    for i in range(len(bufs) + len(vals)):
      struct.pack_into("<Q", to_mv(args_buf.cpu_addr + self.buf_off + i * 8, 8), 0, dummy_iova)

    # write actual buffer addresses and scalar values (overwrites used slots)
    buf_data = struct.pack(f"<{len(ubos)}Q", *[b.iova for b in ubos])
    to_mv(args_buf.cpu_addr + self.buf_off, len(buf_data))[:] = buf_data
    val_data = struct.pack(f"<{len(vals)}I", *vals)
    to_mv(args_buf.cpu_addr + self.buf_off + len(ubos) * 8, len(val_data))[:] = val_data

    # write texture/image descriptors
    def _tex(b, ibo=False):
      imgdt, buf = b
      fmt = mesa.FMT6_32_32_32_32_FLOAT if imgdt.itemsize == 4 else mesa.FMT6_16_16_16_16_FLOAT
      return [qreg.a6xx_tex_const_0(fmt=fmt) if ibo else qreg.a6xx_tex_const_0(0x8, swiz_x=0, swiz_y=1, swiz_z=2, swiz_w=3, fmt=fmt),
              qreg.a6xx_tex_const_1(width=imgdt.shape[1], height=imgdt.shape[0]),
              qreg.a6xx_tex_const_2(type=mesa.A6XX_TEX_2D, pitch=imgdt.pitch, pitchalign=_ctz(imgdt.pitch) - 6), 0, *data64_le(buf.iova),
              qreg.a6xx_tex_const_6(plane_pitch=0x400000), qreg.a6xx_tex_const_7(13), 0, 0, 0, 0, 0, 0, 0, 0]

    if texs:
      tex_data = array.array('I', flatten(map(_tex, texs)))
      to_mv(args_buf.cpu_addr + self.tex_off, len(tex_data) * 4).cast('I')[:] = tex_data
    if ibos:
      ibo_data = array.array('I', flatten(map(functools.partial(_tex, ibo=True), ibos)))
      to_mv(args_buf.cpu_addr + self.ibo_off, len(ibo_data) * 4).cast('I')[:] = ibo_data

    # build PM4 command stream
    pm4 = _build_pm4(self, args_buf, global_size, local_size)

    # write PM4 into command buffer
    cmd_buf: MSMBuffer = self.dev.allocator.alloc(len(pm4) * 4)
    to_mv(cmd_buf.cpu_addr, len(pm4) * 4).cast('I')[:] = array.array('I', pm4)

    # collect all referenced BOs for the submit
    bo_handles = {self.dev.dummy_buf.handle, self.dev._stack.handle, self.dev.border_color_buf.handle,
                  cmd_buf.handle, args_buf.handle, self.lib_buf.handle}
    for b in bufs: bo_handles.add(b.handle)
    bo_list = list(bo_handles)

    # build submit_bo array
    submit_bos = (msm_drm.struct_drm_msm_gem_submit_bo * len(bo_list))()
    for i, h in enumerate(bo_list):
      submit_bos[i].flags = msm_drm.MSM_SUBMIT_BO_READ | msm_drm.MSM_SUBMIT_BO_WRITE
      submit_bos[i].handle = h
      submit_bos[i].presumed = 0

    # build submit_cmd (single command buffer)
    cmd_idx = bo_list.index(cmd_buf.handle)
    submit_cmds = (msm_drm.struct_drm_msm_gem_submit_cmd * 1)()
    submit_cmds[0].type = msm_drm.MSM_SUBMIT_CMD_BUF
    submit_cmds[0].submit_idx = cmd_idx
    submit_cmds[0].submit_offset = 0
    submit_cmds[0].size = len(pm4) * 4
    submit_cmds[0].pad = 0
    submit_cmds[0].nr_relocs = 0
    submit_cmds[0].iova = cmd_buf.iova

    # submit (async, no fence wait - GPU handles ordering via implicit sync)
    st = time.perf_counter_ns() if wait else 0
    submit = msm_drm.DRM_IOCTL_MSM_GEM_SUBMIT(self.dev.fd, flags=msm_drm.MSM_PIPE_3D0,
                                                nr_bos=len(bo_list), nr_cmds=1,
                                                bos=ctypes.addressof(submit_bos), cmds=ctypes.addressof(submit_cmds),
                                                queueid=self.dev.queue_id)
    self.dev.last_fence = submit.fence

    if wait:
      self.dev.synchronize()
      return float(time.perf_counter_ns() - st) * 1e-9
    return None

class MSMGraph(GraphRunner):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.dev: MSMDevice = cast(MSMDevice, cast(CompiledRunner, self.jit_cache[0].prg).dev)

    # pre-build PM4 for each kernel in the JIT cache
    self.pm4_per_kernel: list[tuple[MSMProgram, list[int]]] = []
    for ji in self.jit_cache:
      if not isinstance(ji.prg, CompiledRunner): continue
      prg: MSMProgram = ji.prg._prg
      bufs = [cast(Buffer, b)._buf for b in ji.bufs]
      gs, ls = tuple(ji.prg.p.global_size or (1, 1, 1)), tuple(ji.prg.p.local_size or (1, 1, 1))
      self.pm4_per_kernel.append((prg, self._build_kernel_pm4(prg, bufs, gs, ls)))

    # pre-allocate one big command buffer for the concatenated PM4
    total_dwords = sum(len(pm4) for _, pm4 in self.pm4_per_kernel)
    self.cmd_buf: MSMBuffer = self.dev.allocator.alloc(total_dwords * 4)

    # collect all BO handles referenced across all kernels
    self.all_bo_handles: set[int] = {self.dev.dummy_buf.handle, self.dev._stack.handle, self.dev.border_color_buf.handle, self.cmd_buf.handle}
    for ji in self.jit_cache:
      if not isinstance(ji.prg, CompiledRunner): continue
      for b in ji.bufs:
        if b is not None: self.all_bo_handles.add(cast(Buffer, b)._buf.handle)
      self.all_bo_handles.add(ji.prg._prg.lib_buf.handle)

  def _build_kernel_pm4(self, prg: MSMProgram, bufs: list[MSMBuffer], global_size, local_size) -> list[int]:
    from tinygrad.dtype import ImageDType
    args_buf: MSMBuffer = self.dev.allocator.alloc(prg.kernargs_alloc_size)
    ctypes.memset(args_buf.cpu_addr, 0, prg.kernargs_alloc_size)
    self.all_bo_handles = getattr(self, 'all_bo_handles', set())
    self.all_bo_handles.add(args_buf.handle)

    ubos = [b for i, b in enumerate(bufs) for _, dt in prg.buf_dtypes[i] if not isinstance(dt, ImageDType)]
    uavs = [(dt, b) for i, b in enumerate(bufs) for _, dt in prg.buf_dtypes[i] if isinstance(dt, ImageDType)]
    ibos, texs = uavs[:prg.ibo_cnt], [uavs[prg.ibo_cnt + prg.tex_to_image[i]] for i in range(prg.tex_cnt)]

    for cnst_val, cnst_off, cnst_sz in prg.consts_info:
      to_mv(args_buf.cpu_addr + cnst_off, cnst_sz)[:] = cnst_val.to_bytes(cnst_sz, byteorder='little')
    if prg.samp_cnt > 0: to_mv(args_buf.cpu_addr + prg.samp_off, len(prg.samplers) * 4).cast('I')[:] = array.array('I', prg.samplers)

    dummy_iova = self.dev.dummy_buf.iova
    for i in range(len(bufs) + len(())):
      struct.pack_into("<Q", to_mv(args_buf.cpu_addr + prg.buf_off + i * 8, 8), 0, dummy_iova)
    buf_data = struct.pack(f"<{len(ubos)}Q", *[b.iova for b in ubos])
    to_mv(args_buf.cpu_addr + prg.buf_off, len(buf_data))[:] = buf_data

    def _tex(b, ibo=False):
      imgdt, buf = b
      fmt = mesa.FMT6_32_32_32_32_FLOAT if imgdt.itemsize == 4 else mesa.FMT6_16_16_16_16_FLOAT
      return [qreg.a6xx_tex_const_0(fmt=fmt) if ibo else qreg.a6xx_tex_const_0(0x8, swiz_x=0, swiz_y=1, swiz_z=2, swiz_w=3, fmt=fmt),
              qreg.a6xx_tex_const_1(width=imgdt.shape[1], height=imgdt.shape[0]),
              qreg.a6xx_tex_const_2(type=mesa.A6XX_TEX_2D, pitch=imgdt.pitch, pitchalign=_ctz(imgdt.pitch) - 6), 0, *data64_le(buf.iova),
              qreg.a6xx_tex_const_6(plane_pitch=0x400000), qreg.a6xx_tex_const_7(13), 0, 0, 0, 0, 0, 0, 0, 0]
    if texs:
      to_mv(args_buf.cpu_addr + prg.tex_off, len(texs) * 0x40).cast('I')[:] = array.array('I', flatten(map(_tex, texs)))
    if ibos:
      ibo_data = array.array('I', flatten(map(functools.partial(_tex, ibo=True), ibos)))
      to_mv(args_buf.cpu_addr + prg.ibo_off, len(ibo_data) * 4).cast('I')[:] = ibo_data

    return _build_pm4(prg, args_buf, global_size, local_size)

  def __call__(self, input_buffers: list[Buffer], var_vals: dict[str, int], wait=False) -> float | None:
    # update input buffer handles in BO set
    for input_idx in self.input_replace.values():
      self.all_bo_handles.add(input_buffers[input_idx]._buf.handle)

    # concatenate all PM4 into one command buffer
    offset = 0
    for _, pm4 in self.pm4_per_kernel:
      to_mv(self.cmd_buf.cpu_addr + offset, len(pm4) * 4).cast('I')[:] = array.array('I', pm4)
      offset += len(pm4) * 4

    # build BO table
    bo_list = list(self.all_bo_handles)
    submit_bos = (msm_drm.struct_drm_msm_gem_submit_bo * len(bo_list))()
    for i, h in enumerate(bo_list):
      submit_bos[i].flags = msm_drm.MSM_SUBMIT_BO_READ | msm_drm.MSM_SUBMIT_BO_WRITE
      submit_bos[i].handle = h
      submit_bos[i].presumed = 0

    # single command buffer covering all kernels
    cmd_idx = bo_list.index(self.cmd_buf.handle)
    submit_cmds = (msm_drm.struct_drm_msm_gem_submit_cmd * 1)()
    submit_cmds[0].type = msm_drm.MSM_SUBMIT_CMD_BUF
    submit_cmds[0].submit_idx = cmd_idx
    submit_cmds[0].submit_offset = 0
    submit_cmds[0].size = offset
    submit_cmds[0].pad = 0
    submit_cmds[0].nr_relocs = 0
    submit_cmds[0].iova = self.cmd_buf.iova

    # single submit for all kernels
    st = time.perf_counter_ns() if wait else 0
    submit = msm_drm.DRM_IOCTL_MSM_GEM_SUBMIT(self.dev.fd, flags=msm_drm.MSM_PIPE_3D0,
                                                nr_bos=len(bo_list), nr_cmds=1,
                                                bos=ctypes.addressof(submit_bos), cmds=ctypes.addressof(submit_cmds),
                                                queueid=self.dev.queue_id)
    self.dev.last_fence = submit.fence

    if wait:
      self.dev.synchronize()
      return float(time.perf_counter_ns() - st) * 1e-9
    return None

def _find_msm_device() -> str:
  for path in sorted(glob.glob("/dev/dri/renderD*")):
    try:
      fd = os.open(path, os.O_RDWR)
      try:
        msm_drm.DRM_IOCTL_MSM_GET_PARAM(fd, pipe=msm_drm.MSM_PIPE_3D0, param=msm_drm.MSM_PARAM_GPU_ID)
        os.close(fd)
        return path
      except OSError:
        os.close(fd)
    except OSError: pass
  raise RuntimeError("No MSM DRM device found at /dev/dri/renderD*")

class MSMDevice(Compiled):
  devices: list[str] | None = None

  def __init__(self, device: str = ""):
    if MSMDevice.devices is None:
      MSMDevice.devices = []
      for path in sorted(glob.glob("/dev/dri/renderD*")):
        try:
          fd = os.open(path, os.O_RDWR)
          try:
            msm_drm.DRM_IOCTL_MSM_GET_PARAM(fd, pipe=msm_drm.MSM_PIPE_3D0, param=msm_drm.MSM_PARAM_GPU_ID)
            MSMDevice.devices.append(path)
          except OSError: pass
          os.close(fd)
        except OSError: pass
      if not MSMDevice.devices: raise RuntimeError("No MSM DRM device found")

    device_id = 0 if ":" not in device else int(device.split(":")[1])
    if device_id >= len(MSMDevice.devices): raise RuntimeError(f"MSM device {device_id} not found, have {len(MSMDevice.devices)}")

    self.drm_fd = FileIOInterface(MSMDevice.devices[device_id], os.O_RDWR)
    self.fd = self.drm_fd.fd

    # query GPU info
    gpu_id_resp = msm_drm.DRM_IOCTL_MSM_GET_PARAM(self.fd, pipe=msm_drm.MSM_PIPE_3D0, param=msm_drm.MSM_PARAM_GPU_ID)
    chip_id_resp = msm_drm.DRM_IOCTL_MSM_GET_PARAM(self.fd, pipe=msm_drm.MSM_PIPE_3D0, param=msm_drm.MSM_PARAM_CHIP_ID)
    self.gpu_id_val = gpu_id_resp.value
    self.chip_id = chip_id_resp.value
    self.gpu_id = (self.chip_id >> 24, (self.chip_id >> 16) & 0xFF, (self.chip_id >> 8) & 0xFF)

    if self.gpu_id[:2] >= (7, 3): raise RuntimeError(f"Unsupported GPU: chip_id={self.chip_id:#x}")

    # create submit queue
    sq = msm_drm.DRM_IOCTL_MSM_SUBMITQUEUE_NEW(self.fd, flags=0, prio=1)
    self.queue_id = sq.id
    self.last_fence = 0

    # init allocator first (needed for internal buffer allocations)
    allocator = MSMAllocator(self)
    self.allocator = allocator

    # allocate device-internal buffers
    self.dummy_buf: MSMBuffer = allocator.alloc(0x1000)
    ctypes.memset(self.dummy_buf.cpu_addr, 0, 0x1000)
    self.border_color_buf: MSMBuffer = allocator.alloc(0x1000)
    ctypes.memset(self.border_color_buf.cpu_addr, 0, 0x1000)

    # compilers: IR3 only (no QCOMCLRenderer, that needs KGSL)
    compilers = CompilerSet(cset=[(functools.partial(IR3Renderer, self.chip_id), None)])

    super().__init__(device, allocator, compilers, functools.partial(MSMProgram, self), graph=MSMGraph)

  def _ensure_stack_size(self, sz):
    if not hasattr(self, '_stack'): self._stack = self.allocator.alloc(sz)
    elif self._stack.size < sz:
      self.synchronize()
      self.allocator.free(self._stack, self._stack.size)
      self._stack = self.allocator.alloc(sz)

  def synchronize(self):
    if self.last_fence == 0: return
    req = msm_drm.struct_drm_msm_wait_fence(fence=self.last_fence, flags=msm_drm.MSM_WAIT_FENCE_BOOST, queueid=self.queue_id,
                                              timeout=msm_drm.struct_drm_msm_timespec(tv_sec=int(time.time()) + 10, tv_nsec=0))
    try: msm_drm.DRM_IOCTL_MSM_WAIT_FENCE(self.fd, __payload=req)
    except OSError as e:
      if e.errno != 62: raise  # ETIME is ok (already completed)

  def finalize(self):
    self.synchronize()
    self.allocator.free_cache()
    self.allocator.finalize()
