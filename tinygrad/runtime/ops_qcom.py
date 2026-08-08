from __future__ import annotations
import os, ctypes, errno, functools, glob, mmap, struct, array, math, sys, time, weakref, contextlib, select
assert sys.platform != 'win32'
from dataclasses import dataclass
from typing import Any
from tinygrad.device import BufferSpec, Device, TinyELF
from tinygrad.runtime.support.hcq import HCQBuffer, HWQueue, HCQProgram, HCQCompiled, HCQAllocatorBase, HCQSignal, HCQArgsState, BumpAllocator
from tinygrad.runtime.support.hcq import FileIOInterface, MMIOInterface
from tinygrad.runtime.support.memory import TLSFAllocator
from tinygrad.runtime.autogen import kgsl, mesa, msm_drm
from tinygrad.renderer.cstyle import QCOMCLRenderer
from tinygrad.renderer.nir import IR3Renderer
from tinygrad.helpers import getenv, mv_address, to_mv, round_up, data64_le, ceildiv, prod, cpu_profile, lo32, suppress_finalizing
from tinygrad.helpers import is_image_shape, next_power2, flatten, PROFILE, IMAGE
from tinygrad.dtype import dtypes, AddrSpace
from tinygrad.runtime.support.system import System
if getenv("IOCTL"): import extra.qcom_gpu_driver.opencl_ioctl  # noqa: F401  # pylint: disable=unused-import

BUFTYPE_BUF, BUFTYPE_TEX, BUFTYPE_IBO = 0, 1, 2
MSM_WAIT_SLICE_NS = 1_000_000

@functools.cache
def dcache_flush():
  from tinygrad.uop.ops import UOp, Ops, KernelInfo
  from tinygrad.codegen import to_program
  buf, n = UOp.param(0, dtypes.uint8, shape=(1,)), UOp.param(1, dtypes.int, shape=(), name="n", addrspace=AddrSpace.ALU)
  i = UOp.range(n, 0, dtype=dtypes.int)
  flush = UOp(Ops.CUSTOM, src=(buf.index(i * 64),), arg='__asm__ volatile("dc cvac, %0" :: "r"({0}) : "memory");')
  sink = UOp.sink(flush.end(i), UOp(Ops.CUSTOM, arg='__asm__ volatile("dsb sy" ::: "memory");'), arg=KernelInfo(name="dcache_flush"))
  prg = to_program(UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=tuple(sink.toposort())))), Device["CPU"].renderer)
  return Device["CPU"].runtime(prg.to_elf())

#Parse C-style defines: <regname>_<field_x>__SHIFT and <regname>_<field_y>__MASK from the adreno module into the following format:
# qreg.<regname>(<field_x>=..., <field_y>=..., ..., <field_n>=...)
def _qreg_exec(__reg, __val=0, **kwargs):
  for k, v in kwargs.items():
    reg_name = f"{__reg[4:]}_{k.removeprefix('_').upper()}"
    __val |= (getattr(mesa, reg_name) if v else 0) if type(v) is bool else (v << getattr(mesa, f'{reg_name}__SHIFT'))
  return __val
qreg: Any = type("QREG", (object,), {name[4:].lower(): functools.partial(_qreg_exec, name) for name in mesa.__dict__.keys() if name[:4] == 'REG_'})

def ctz(v): return (v & -v).bit_length() - 1

def parity(val: int):
  for i in range(4,1,-1): val ^= val >> (1 << i)
  return (~0x6996 >> (val & 0xf)) & 1

def pkt7_hdr(opcode: int, cnt: int): return mesa.CP_TYPE7_PKT | cnt & 0x3FFF | parity(cnt) << 15 | (opcode & 0x7F) << 16 | parity(opcode) << 23

def pkt4_hdr(reg: int, cnt: int): return mesa.CP_TYPE4_PKT | cnt & 0x7F | parity(cnt) << 7 | (reg & 0x3FFFF) << 8 | parity(reg) << 27

def _read_lib(lib, off) -> int: return struct.unpack("I", lib[off:off+4])[0]

def _qcom_pvtmem_sizes(pvtmem:int, fibers_per_sp:int, num_sp_cores:int) -> tuple[int, int, int, int]:
  if pvtmem == 0: return 0, 0, 0, 0x1000
  per_fiber_size = next_power2(round_up(pvtmem, 512))
  per_sp_size = round_up(per_fiber_size * fibers_per_sp, 0x1000)
  return per_fiber_size >> 9, per_sp_size >> 12, per_sp_size >> 11, per_sp_size * num_sp_cores

class QCOMSignal(HCQSignal):
  def __init__(self, *args, **kwargs): super().__init__(*args, **{**kwargs, 'timestamp_divider': 19.2})

  def _sleep(self, time_spent_since_last_sleep_ms:int):
    # Sleep only for timeline signals. Do it immediately to free cpu.
    if self.is_timeline and self.owner is not None: self.owner.iface.sleep(time_spent_since_last_sleep_ms)

class QCOMComputeQueue(HWQueue):
  def __init__(self, dev:QCOMDevice):
    self.dev = dev
    super().__init__()

  @suppress_finalizing
  def __del__(self):
    if self.binded_device is not None: self.binded_device.allocator.free(self.hw_page, self.hw_page.size, BufferSpec(cpu_access=True, nolru=True))

  def cmd(self, opcode: int, *vals: int): self.q(pkt7_hdr(opcode, len(vals)), *vals)

  def reg(self, reg: int, *vals: int): self.q(pkt4_hdr(reg, len(vals)), *vals)

  def _cache_flush(self, write_back=True, invalidate=False, sync=True, memsync=False):
    # TODO: 7xx support.
    if write_back:
      event = mesa.CACHE_FLUSH_TS | (mesa.CP_EVENT_WRITE_0_IRQ if getattr(self.dev.iface, "event_write_irq", False) else 0)
      self.cmd(mesa.CP_EVENT_WRITE, event, *data64_le(self.dev.dummy_addr), 0) # dirty cache write-back.
    if invalidate: self.cmd(mesa.CP_EVENT_WRITE, mesa.CACHE_INVALIDATE) # invalidate cache lines (following reads from RAM).
    if memsync: self.cmd(mesa.CP_WAIT_MEM_WRITES)
    if sync: self.cmd(mesa.CP_WAIT_FOR_IDLE)

  def memory_barrier(self):
    self._cache_flush(write_back=True, invalidate=True, sync=True, memsync=True)
    return self

  def signal(self, signal:QCOMSignal, value=0):
    self.cmd(mesa.CP_WAIT_FOR_IDLE)
    if self.dev.gpu_id[:2] < (7, 3):
      event = qreg.cp_event_write_0(event=mesa.CACHE_FLUSH_TS) | \
              (mesa.CP_EVENT_WRITE_0_IRQ if getattr(self.dev.iface, "event_write_irq", False) else 0)
      self.cmd(mesa.CP_EVENT_WRITE, event, *data64_le(signal.value_addr), lo32(value))
      self._cache_flush(write_back=True, invalidate=False, sync=False, memsync=False)
    else:
      # TODO: support devices starting with 8 Gen 1. Also, 700th series have convenient CP_GLOBAL_TIMESTAMP and CP_LOCAL_TIMESTAMP
      raise RuntimeError('CP_EVENT_WRITE7 is not supported')
    return self

  def timestamp(self, signal:QCOMSignal):
    self.cmd(mesa.CP_WAIT_FOR_IDLE)
    self.cmd(mesa.CP_REG_TO_MEM, qreg.cp_reg_to_mem_0(reg=mesa.REG_A6XX_CP_ALWAYS_ON_COUNTER, cnt=2, _64b=True),*data64_le(signal.timestamp_addr))
    return self

  def wait(self, signal:QCOMSignal, value=0):
    self.cmd(mesa.CP_WAIT_REG_MEM, qreg.cp_wait_reg_mem_0(function=mesa.WRITE_GE, poll=mesa.POLL_MEMORY),*data64_le(signal.value_addr),
             qreg.cp_wait_reg_mem_3(ref=value&0xFFFFFFFF), qreg.cp_wait_reg_mem_4(mask=0xFFFFFFFF), qreg.cp_wait_reg_mem_5(delay_loop_cycles=32))
    return self

  def _build_gpu_command(self, dev:QCOMDevice, hw_page:HCQBuffer|None=None) -> HCQBuffer:
    if hw_page is None:
      hw_page_addr = dev.cmd_buf_allocator.alloc(len(self._q) * 4)
      hw_page = dev.cmd_buf.offset(offset=int(hw_page_addr - dev.cmd_buf.va_addr), size=len(self._q) * 4)
    hw_page.cpu_view().view(fmt='I')[:] = array.array('I', self._q)
    return hw_page

  def bind(self, dev:QCOMDevice):
    self.binded_device = dev
    self.hw_page = self._build_gpu_command(dev, dev.allocator.alloc(len(self._q) * 4, BufferSpec(cpu_access=True, nolru=True)))
    # From now on, the queue is on the device for faster submission.
    self._q = self.hw_page.cpu_view().view(fmt='I')

  def _submit(self, dev:QCOMDevice):
    command = self.hw_page if self.binded_device == dev else self._build_gpu_command(dev)
    dev.last_cmd = dev.iface.submit(command, len(self._q) * 4)

  def exec(self, prg:QCOMProgram, args_state:QCOMArgsState, global_size, local_size):
    self.bind_args_state(args_state)

    def cast_int(x, ceil=False): return (math.ceil(x) if ceil else int(x)) if isinstance(x, float) else x
    global_size_mp = [cast_int(g*l) for g,l in zip(global_size, local_size)]

    threadsize = mesa.THREAD128 if prg.double_threadsize else mesa.THREAD64
    supports_double_threadsize = self.dev.dev_info.a6xx.supports_double_threadsize
    wge_threadsize = threadsize if supports_double_threadsize else mesa.THREAD128
    const_ram_mode = mesa.CONSTLEN_512 if prg.constlen > 256 else \
                     mesa.CONSTLEN_256 if prg.constlen > 192 else \
                     mesa.CONSTLEN_192 if prg.constlen > 128 else mesa.CONSTLEN_128

    self.cmd(mesa.CP_SET_MARKER, qreg.a6xx_cp_set_marker_0(mode=mesa.RM6_COMPUTE))
    self.reg(mesa.REG_A6XX_SP_UPDATE_CNTL, qreg.a6xx_sp_update_cntl(vs_state=True, hs_state=True, ds_state=True, gs_state=True,
                                                                   fs_state=True, cs_state=True, cs_uav=True, gfx_uav=True))
    self.reg(mesa.REG_A6XX_SP_CS_TSIZE, qreg.a6xx_sp_cs_tsize(0x80)) # is this right? mesa uses 1
    self.reg(mesa.REG_A6XX_SP_CS_USIZE, qreg.a6xx_sp_cs_usize(0x40)) # mesa also uses 1
    self.reg(mesa.REG_A6XX_SP_MODE_CNTL, qreg.a6xx_sp_mode_cntl(isammode=mesa.ISAMMODE_GL if prg.NIR else mesa.ISAMMODE_CL,
                                                                constant_demotion_enable=prg.NIR))
    self.reg(mesa.REG_A6XX_SP_PERFCTR_SHADER_MASK, qreg.a6xx_sp_perfctr_shader_mask(cs=True))
    self.reg(mesa.REG_A6XX_TPL1_MODE_CNTL, qreg.a6xx_tpl1_mode_cntl(isammode=mesa.ISAMMODE_GL if prg.NIR else mesa.ISAMMODE_CL))
    self.reg(mesa.REG_A6XX_TPL1_DBG_ECO_CNTL, 0)
    self.cmd(mesa.CP_WAIT_FOR_IDLE)

    self.reg(mesa.REG_A6XX_SP_CS_NDRANGE_0,
             qreg.a6xx_sp_cs_ndrange_0(kerneldim=3, localsizex=local_size[0] - 1, localsizey=local_size[1] - 1, localsizez=local_size[2] - 1),
             global_size_mp[0], 0, global_size_mp[1], 0, global_size_mp[2], 0, 0xccc0cf,
             0xfc | qreg.a6xx_sp_cs_wge_cntl(threadsize=wge_threadsize),
             cast_int(global_size[0], ceil=True), cast_int(global_size[1], ceil=True), cast_int(global_size[2], ceil=True))

    self.reg(mesa.REG_A6XX_SP_CS_CNTL_0,
             qreg.a6xx_sp_cs_cntl_0(threadsize=threadsize, halfregfootprint=prg.hregs, fullregfootprint=prg.fregs,
                                    branchstack=prg.brnchstck, earlypreamble=prg.early_preamble, mergedregs=prg.mergedregs),
             qreg.a6xx_sp_cs_cntl_1(constantrammode=const_ram_mode, shared_size=prg.shared_size),
             0, prg.prg_offset, *data64_le(prg.lib_gpu.va_addr),
             qreg.a6xx_sp_cs_pvt_mem_param(memsizeperitem=prg.pvtmem_size_per_item), *data64_le(prg.dev._stack.va_addr),
             qreg.a6xx_sp_cs_pvt_mem_size(totalpvtmemsize=prg.pvtmem_size_total,
                                          perwavememlayout=prg.pvtmem_per_wave))
    if self.dev.dev_info.a6xx.has_lpac:
      self.reg(mesa.REG_A6XX_HLSQ_CS_CTRL_REG1,
               qreg.a6xx_hlsq_cs_ctrl_reg1(constantrammode=const_ram_mode, shared_size=prg.shared_size))
    if not supports_double_threadsize:
      self.reg(mesa.REG_A6XX_SP_PS_WAVE_CNTL, qreg.a6xx_sp_ps_wave_cntl(threadsize=threadsize))

    if prg.NIR and prg.wgsz != 0xfc:
      args_state.buf.cpu_view().view(offset=prg.wgsz * 4, size=12, fmt='B')[:] = struct.pack("III", *local_size)
    self.cmd(mesa.CP_LOAD_STATE6_FRAG, qreg.cp_load_state6_0(state_type=mesa.ST_CONSTANTS, state_src=mesa.SS6_INDIRECT,
                                                             state_block=mesa.SB6_CS_SHADER, num_unit=1024 // 4),
             *data64_le(args_state.buf.va_addr))
    self.cmd(mesa.CP_LOAD_STATE6_FRAG, qreg.cp_load_state6_0(state_type=mesa.ST_SHADER, state_src=mesa.SS6_INDIRECT,
                                                             state_block=mesa.SB6_CS_SHADER, num_unit=ceildiv(prg.image_size, 128)),
             *data64_le(prg.lib_gpu.va_addr))

    self.reg(mesa.REG_A6XX_SP_REG_PROG_ID_0, 0xfcfcfcfc, 0xfcfcfcfc, 0xfcfcfcfc, 0xfc,
             qreg.a6xx_sp_cs_const_config(constlen=ceildiv(prg.constlen, 4), enabled=True))

    self.reg(mesa.REG_A6XX_SP_CS_PVT_MEM_STACK_OFFSET, qreg.a6xx_sp_cs_pvt_mem_stack_offset(prg.hw_stack_offset))
    # image_size is in bytes, but INSTR_SIZE is measured in units of instruction groups (16 instructions, 8 bytes each)
    # https://elixir.bootlin.com/mesa/mesa-26.1.5/source/src/freedreno/ir3/ir3_shader.h#L719-L723
    self.reg(mesa.REG_A6XX_SP_CS_INSTR_SIZE, qreg.a6xx_sp_cs_instr_size(ceildiv(prg.image_size, 128)))

    if prg.samp_cnt > 0:
      self.cmd(mesa.CP_LOAD_STATE6_FRAG, qreg.cp_load_state6_0(state_type=mesa.ST_SHADER, state_src=mesa.SS6_INDIRECT,
                                                               state_block=mesa.SB6_CS_TEX, num_unit=args_state.prg.samp_cnt),
               *data64_le(args_state.buf.va_addr + args_state.prg.samp_off))
      self.reg(mesa.REG_A6XX_SP_CS_SAMPLER_BASE, *data64_le(args_state.buf.va_addr + args_state.prg.samp_off))
      self.reg(mesa.REG_A6XX_TPL1_CS_BORDER_COLOR_BASE, *data64_le(prg.dev.border_color_buf.va_addr))

    if prg.tex_cnt > 0:
      self.cmd(mesa.CP_LOAD_STATE6_FRAG, qreg.cp_load_state6_0(state_type=mesa.ST_CONSTANTS, state_src=mesa.SS6_INDIRECT,
                                                               state_block=mesa.SB6_CS_TEX, num_unit=min(16, args_state.prg.tex_cnt)),
               *data64_le(args_state.buf.va_addr + args_state.prg.tex_off))
      self.reg(mesa.REG_A6XX_SP_CS_TEXMEMOBJ_BASE, *data64_le(args_state.buf.va_addr + args_state.prg.tex_off))

    if prg.ibo_cnt > 0:
      self.cmd(mesa.CP_LOAD_STATE6_FRAG, qreg.cp_load_state6_0(state_type=mesa.ST6_UAV, state_src=mesa.SS6_INDIRECT,
                                                               state_block=mesa.SB6_CS_SHADER, num_unit=args_state.prg.ibo_cnt),
               *data64_le(args_state.buf.va_addr + args_state.prg.ibo_off))
      self.reg(mesa.REG_A6XX_SP_CS_UAV_BASE, *data64_le(args_state.buf.va_addr + args_state.prg.ibo_off))

    self.reg(mesa.REG_A6XX_SP_CS_CONFIG,
             qreg.a6xx_sp_cs_config(enabled=True, nsamp=args_state.prg.samp_cnt, ntex=args_state.prg.tex_cnt, nuav=args_state.prg.ibo_cnt))

    if prg.NIR:
      self.reg(mesa.REG_A6XX_SP_CS_CONST_CONFIG_0,
               qreg.a6xx_sp_cs_const_config_0(wgidconstid=prg.wgid, wgsizeconstid=prg.wgsz, wgoffsetconstid=0xfc, localidregid=prg.lid),
               qreg.a6xx_sp_cs_wge_cntl(linearlocalidregid=0xfc, threadsize=wge_threadsize))
      if self.dev.dev_info.a6xx.has_lpac:
        self.reg(mesa.REG_A6XX_SP_CS_WIE_CNTL_0,
                 qreg.a6xx_sp_cs_wie_cntl_0(wgidconstid=prg.wgid, wgsizeconstid=prg.wgsz,
                                            wgoffsetconstid=0xfc, localidregid=prg.lid),
                 qreg.a6xx_sp_cs_wie_cntl_1(linearlocalidregid=0xfc, threadsize=threadsize))
      self.cmd(mesa.CP_EXEC_CS, 0,
               qreg.cp_exec_cs_1(ngroups_x=cast_int(global_size[0], ceil=True)),
               qreg.cp_exec_cs_2(ngroups_y=cast_int(global_size[1], ceil=True)),
               qreg.cp_exec_cs_3(_ngroups_z=cast_int(global_size[2], ceil=True)))
    else: self.cmd(mesa.CP_RUN_OPENCL, 0)

    self._cache_flush(write_back=True, invalidate=False, sync=False, memsync=False)
    return self

class QCOMArgsState(HCQArgsState):
  def __init__(self, buf:HCQBuffer, prg:QCOMProgram, bufs:tuple[HCQBuffer, ...], vals:tuple[int, ...]=()):
    super().__init__(buf, prg, bufs, vals=vals)
    self.buf.cpu_view().view(size=prg.kernargs_alloc_size, fmt='B')[:] = bytes(prg.kernargs_alloc_size)

    ubos = [bufs[slot] for _,slot,_,shape in prg.signature if slot < len(bufs) and not is_image_shape(shape)]
    uavs = [(dt,shape,bufs[slot]) for _,slot,dt,shape in prg.signature if slot < len(bufs) and is_image_shape(shape)]
    # NIR can reorder images to different texture slots
    ibos, texs = uavs[:prg.ibo_cnt], [uavs[prg.ibo_cnt + (prg.tex_to_image[i] if prg.NIR else i)] for i in range(prg.tex_cnt)]
    for cnst_val,cnst_off,cnst_sz in prg.consts_info:
      self.buf.cpu_view().view(offset=cnst_off, size=cnst_sz, fmt='B')[:] = cnst_val.to_bytes(cnst_sz, byteorder='little')

    if prg.samp_cnt > 0:
      self.buf.cpu_view().view(offset=prg.samp_off, size=len(prg.samplers) * 4, fmt='I')[:] = array.array('I', prg.samplers)
    if prg.NIR:
      self.bind_sints_to_buf(*[b.va_addr for b in ubos], buf=self.buf, fmt='Q', offset=prg.buf_off)
      for v,(o,dt) in zip(vals, TinyELF.iter_sig(prg.signature[len(bufs):], len(ubos)*8)):
        self.bind_sints_to_buf(v, buf=self.buf, fmt=dt.fmt, offset=prg.buf_off + o)
    else:
      for i, b in enumerate(ubos): self.bind_sints_to_buf(b.va_addr, buf=self.buf, fmt='Q', offset=prg.buf_offs[i])
      for i,(v,(_,_,dt,_)) in enumerate(zip(vals, prg.signature[len(bufs):])):
        self.bind_sints_to_buf(v, buf=self.buf, fmt=dt.fmt, offset=prg.buf_offs[i+len(ubos)])

    def _tex(b, ibo=False):
      imgdt, shape, buf = b
      pitch = shape[1] * 4 * imgdt.itemsize
      fmt = mesa.FMT6_32_32_32_32_FLOAT if imgdt.itemsize == 4 else mesa.FMT6_16_16_16_16_FLOAT
      return [qreg.a6xx_tex_const_0(fmt=fmt) if ibo else qreg.a6xx_tex_const_0(0x8, swiz_x=0, swiz_y=1, swiz_z=2, swiz_w=3, fmt=fmt),
              qreg.a6xx_tex_const_1(width=shape[1], height=shape[0]),
              qreg.a6xx_tex_const_2(type=mesa.A6XX_TEX_2D, pitch=pitch, pitchalign=ctz(pitch)-6), 0, *data64_le(buf.va_addr),
              qreg.a6xx_tex_const_6(plane_pitch=0x400000), qreg.a6xx_tex_const_7(13), 0, 0, 0, 0, 0, 0, 0, 0]

    self.bind_sints_to_buf(*flatten(map(_tex, texs)), buf=self.buf, fmt='I', offset=prg.tex_off)
    self.bind_sints_to_buf(*flatten(map(functools.partial(_tex, ibo=True), ibos)), buf=self.buf, fmt='I', offset=prg.ibo_off)

class QCOMProgram(HCQProgram['QCOMDevice']):
  def __init__(self, dev: QCOMDevice, obj: TinyELF):
    self.dev: QCOMDevice = dev
    self.signature, self.name, self.NIR = obj.signature, obj.name, isinstance(dev.renderer, IR3Renderer)

    if self.NIR:
      from tinygrad.runtime.support.compiler_mesa import IR3Compiler
      v, cs, imm_vals, self.image = IR3Compiler.unpack_lib(obj.lib)
      self.prg_offset, self.brnchstck, self.image_size, self.pvtmem, self.shmem = 0, v.branchstack, v.info.size, v.pvtmem_size, v.shared_size
      self.constlen, self.pvtmem_per_wave = v.constlen, v.pvtmem_per_wave
      self.double_threadsize, self.early_preamble, self.mergedregs = v.info.double_threadsize, v.early_preamble, v.mergedregs
      self.wgsz = alloc.offset_vec4 * 4 + 8 if (alloc:=cs.allocs.consts[mesa.IR3_CONST_ALLOC_DRIVER_PARAMS]).size_vec4 else 0xfc

      self.wgid, self.lid = v.cs.work_group_id, v.cs.local_invocation_id # register ids
      self.buf_off, imm_off = cs.ubo_state.range[0].offset, cs.allocs.max_const_offset_vec4 * 16
      self.consts_info = [(struct.unpack_from("<I", imm_vals, i)[0], imm_off + i, 4) for i in range(0, len(imm_vals), 4)]

      # see https://elixir.bootlin.com/mesa/mesa-25.3.0/source/src/freedreno/ir3/ir3_shader.h#L525
      # and https://elixir.bootlin.com/mesa/mesa-25.3.0/source/src/freedreno/ir3/ir3_compiler_nir.c#L5389
      self.samp_cnt, self.tex_cnt, self.ibo_cnt = (nt:=v.image_mapping.num_tex), nt, v.num_uavs - nt
      self.tex_to_image = v.image_mapping.tex_to_image[:]
      # IR3 outputs a sampler for every texture (https://elixir.bootlin.com/mesa/mesa-25.3.0/source/src/freedreno/ir3/ir3_compiler_nir.c#L1714)
      self.samplers = [qreg.a6xx_tex_samp_0(wrap_s=(clamp_mode:=mesa.A6XX_TEX_CLAMP_TO_BORDER), wrap_t=clamp_mode, wrap_r=clamp_mode),
                       qreg.a6xx_tex_samp_1(unnorm_coords=True, cubemapseamlessfiltoff=True), 0, 0] * self.samp_cnt

      self.tex_off, self.ibo_off, self.samp_off = 2048, 2048 + 0x40 * self.tex_cnt, 2048 + 0x40 * (self.tex_cnt + self.ibo_cnt)
      self.fregs, self.hregs = v.info.max_reg + 1, v.info.max_half_reg + 1
    else:
      self._parse_lib(obj.lib)
      self.constlen, self.pvtmem_per_wave = 256, False
      self.double_threadsize = self.early_preamble = self.mergedregs = False

    self.lib_gpu: HCQBuffer = self.dev.allocator.alloc(self.image_size, buf_spec:=BufferSpec(cpu_access=True, nolru=True))
    self.lib_gpu.cpu_view().view(size=self.image_size, fmt='B')[:] = self.image

    self.pvtmem_size_per_item, self.pvtmem_size_total, self.hw_stack_offset, stack_size = \
      _qcom_pvtmem_sizes(self.pvtmem, self.dev.dev_info.fibers_per_sp, self.dev.dev_info.num_sp_cores)
    self.shared_size: int = max(1, (self.shmem - 1) // 1024)
    self.max_threads = min(1024, ((384 * 32) // (max(1, (self.fregs + round_up(self.hregs, 2) // 2)) * 128)) * 128)
    dev._ensure_stack_size(stack_size)

    kernargs_alloc_size = round_up(2048 + (self.tex_cnt + self.ibo_cnt) * 0x40 + len(self.samplers) * 4, 0x100)
    super().__init__(QCOMArgsState, self.dev, obj, kernargs_alloc_size=kernargs_alloc_size)
    weakref.finalize(self, self._fini, self.dev, self.lib_gpu, buf_spec)

  def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1),
               vals:tuple[int|None, ...]=(), wait=False, **kw):
    if self.max_threads < prod(local_size): raise RuntimeError("Too many resources requested for launch")
    if any(g*l>mx for g,l,mx in zip(global_size, local_size, [65536, 65536, 65536])) and any(l>mx for l,mx in zip(local_size, [1024, 1024, 1024])):
      raise RuntimeError(f"Invalid global/local dims {global_size=}, {local_size=}")
    return super().__call__(*bufs, global_size=global_size, local_size=local_size, vals=vals, wait=wait)

  def _parse_lib(self, lib):
    # Extract image binary
    self.image_size = _read_lib(lib, 0x100)
    self.image = lib[(image_offset:=_read_lib(lib, 0xc0)):image_offset+self.image_size]

    # Parse image descriptors
    image_desc_off = _read_lib(lib, 0x110)
    self.prg_offset, self.brnchstck = _read_lib(lib, image_desc_off+0xc4), _read_lib(lib, image_desc_off+0x108) // 2
    self.pvtmem, self.shmem = _read_lib(lib, image_desc_off+0xc8), _read_lib(lib, image_desc_off+0xd8)

    # Fill up constants and buffers info
    self.consts_info = []

    # Collect sampler info.
    self.samp_cnt = samp_cnt_in_file = _read_lib(lib, image_desc_off + 0xdc)
    assert self.samp_cnt <= 1, "Up to one sampler supported"
    if self.samp_cnt:
      self.samp_cnt += 1
      self.samplers = [qreg.a6xx_tex_samp_0(wrap_s=(clamp_mode:=mesa.A6XX_TEX_CLAMP_TO_BORDER), wrap_t=clamp_mode, wrap_r=clamp_mode),
                       qreg.a6xx_tex_samp_1(unnorm_coords=True, cubemapseamlessfiltoff=True), 0, 0, 0, 0, 0, 0]
    else: self.samplers = []

    # Collect kernel arguments (buffers) info.
    bdoff, binfos = round_up(image_desc_off + 0x158 + len(self.name), 4) + 8 * samp_cnt_in_file, []
    while bdoff + 32 <= len(lib):
      length, _, _, offset_words, _, _, _, typ = struct.unpack("8I", lib[bdoff:bdoff+32])
      if length == 0: break
      binfos.append((offset_words * 4, typ))
      bdoff += length
    self.buf_offs = [off for off,typ in binfos if typ not in {BUFTYPE_TEX, BUFTYPE_IBO}]

    # Setting correct offsets to textures/ibos.
    self.tex_cnt, self.ibo_cnt = sum(typ is BUFTYPE_TEX for _,typ in binfos), sum(typ is BUFTYPE_IBO for _,typ in binfos)
    self.ibo_off, self.tex_off, self.samp_off = 2048, 2048 + 0x40 * self.ibo_cnt, 2048 + 0x40 * self.tex_cnt + 0x40 * self.ibo_cnt

    if _read_lib(lib, 0xb0) != 0: # check if we have constants.
      cdoff = _read_lib(lib, 0xac)
      while cdoff + 40 <= image_offset:
        cnst, offset_words, _, is32 = struct.unpack("I", lib[cdoff:cdoff+4])[0], *struct.unpack("III", lib[cdoff+16:cdoff+28])
        self.consts_info.append((cnst, offset_words * (sz_bytes:=(2 << is32)), sz_bytes))
        cdoff += 40

    # Registers info
    reg_desc_off = _read_lib(lib, 0x34)
    self.fregs, self.hregs = _read_lib(lib, reg_desc_off + 0x14), _read_lib(lib, reg_desc_off + 0x18)

class QCOMAllocator(HCQAllocatorBase):
  def _alloc(self, size:int, opts:BufferSpec) -> HCQBuffer:
    if opts.external_ptr is not None: return self.dev.iface.map(opts.external_ptr, size)
    return self.dev.iface.alloc(size, uncached=opts.uncached)

  def _do_copy(self, src_addr, dest_addr, size, prof_text):
    self.dev.synchronize()
    with cpu_profile(prof_text, f"{self.dev.device}:COPY"): ctypes.memmove(dest_addr, src_addr, size)

  def _copyin(self, dest:HCQBuffer, src:memoryview): self._do_copy(mv_address(src), dest.cpu_view().addr, src.nbytes, f"TINY -> {self.dev.device}")
  def _copyout(self, dest:memoryview, src:HCQBuffer): self._do_copy(src.cpu_view().addr, mv_address(dest), src.size, f"{self.dev.device} -> TINY")

  def _as_buffer(self, src:HCQBuffer) -> memoryview: return to_mv(src.cpu_view().addr, src.size)

  def _do_free(self, opaque, options:BufferSpec): self.dev.iface.free(opaque)

def flag(nm, val): return (val << getattr(kgsl, f"{nm}_SHIFT")) & getattr(kgsl, f"{nm}_MASK")

class KGSLIface:
  count = 1
  renderers = [QCOMCLRenderer, IR3Renderer]

  def __init__(self, dev:QCOMDevice, device_id:int):
    if device_id != 0: raise RuntimeError(f"QCOM:{device_id} does not exist (1 device available)")
    self.dev = dev
    self.fd = FileIOInterface('/dev/kgsl-3d0', os.O_RDWR)

    flags = kgsl.KGSL_CONTEXT_PREAMBLE | kgsl.KGSL_CONTEXT_PWR_CONSTRAINT | kgsl.KGSL_CONTEXT_NO_FAULT_TOLERANCE | kgsl.KGSL_CONTEXT_NO_GMEM_ALLOC \
      | flag("KGSL_CONTEXT_PRIORITY", getenv("QCOM_PRIORITY", 8)) | flag("KGSL_CONTEXT_PREEMPT_STYLE", kgsl.KGSL_CONTEXT_PREEMPT_STYLE_FINEGRAIN)
    self.ctx = kgsl.IOCTL_KGSL_DRAWCTXT_CREATE(self.fd, flags=flags).drawctxt_id

    # Set max power
    struct.pack_into('IIQQ', pwr:=memoryview(bytearray(0x18)), 0, 1, self.ctx, mv_address(_:=memoryview(array.array('I', [1]))), 4)
    kgsl.IOCTL_KGSL_SETPROPERTY(self.fd, type=kgsl.KGSL_PROP_PWR_CONSTRAINT, value=mv_address(pwr), sizebytes=pwr.nbytes)

    # Load info about qcom device
    info = kgsl.struct_kgsl_devinfo()
    kgsl.IOCTL_KGSL_DEVICE_GETPROPERTY(self.fd, type=kgsl.KGSL_PROP_DEVICE_INFO, value=ctypes.addressof(info), sizebytes=ctypes.sizeof(info))
    self.chip_id = info.chip_id
    self.gpu_id = (self.chip_id >> 24, (self.chip_id >> 16) & 0xFF, (self.chip_id >> 8) & 0xFF)

    if PROFILE and self.gpu_id[:2] < (7, 3):
      System.write_sysfs("/sys/class/kgsl/kgsl-3d0/idle_timer", value="4000000000", msg="Failed to disable suspend mode", expected="4294967276")

  def alloc(self, size:int, uncached=False, fill_zeroes=False) -> HCQBuffer:
    flags = flag("KGSL_MEMALIGN", alignment_hint:=12) | kgsl.KGSL_MEMFLAGS_USE_CPU_MAP
    if uncached: flags |= flag("KGSL_CACHEMODE", kgsl.KGSL_CACHEMODE_UNCACHED)

    alloc = kgsl.IOCTL_KGSL_GPUOBJ_ALLOC(self.fd, size=(bosz:=round_up(size, 1<<alignment_hint)), flags=flags, mmapsize=bosz)
    va_addr = self.fd.mmap(0, bosz, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, alloc.id * 0x1000)

    if fill_zeroes: ctypes.memset(va_addr, 0, size)
    return HCQBuffer(va_addr=va_addr, size=size, meta=(alloc, True), view=MMIOInterface(va_addr, size, fmt='B'), owner=self.dev)

  def map(self, ptr:int, size:int, _fd:int|None=None) -> HCQBuffer:
    ptr_aligned, size_aligned = (ptr & ~0xfff), round_up(size + (ptr & 0xfff), 0x1000)
    dcache_flush().fxn(ctypes.c_uint64(ptr_line_aligned:=ptr & ~63), ceildiv(ptr + size - ptr_line_aligned, 64))
    try:
      mi = kgsl.IOCTL_KGSL_MAP_USER_MEM(self.fd, hostptr=ptr_aligned, len=size_aligned, memtype=kgsl.KGSL_USER_MEM_TYPE_ADDR)
      return HCQBuffer(mi.gpuaddr + (ptr - ptr_aligned), size=size, meta=(mi, False), view=MMIOInterface(ptr, size, fmt='B'), owner=self.dev)
    except OSError as e:
      if e.errno == 14: return HCQBuffer(va_addr=ptr, size=size, meta=(None, False), view=MMIOInterface(ptr, size, fmt='B'), owner=self.dev)
      raise RuntimeError("Failed to map external pointer to GPU memory") from e

  def free(self, mem:HCQBuffer):
    if mem.meta[0] is None: return # external (gpu) ptr
    if not mem.meta[1]: kgsl.IOCTL_KGSL_SHAREDMEM_FREE(self.fd, gpuaddr=mem.meta[0].gpuaddr) # external (cpu) ptr
    else:
      kgsl.IOCTL_KGSL_GPUOBJ_FREE(self.fd, id=mem.meta[0].id)
      FileIOInterface.munmap(mem.cpu_view().addr, mem.meta[0].mmapsize)

  def submit(self, command:HCQBuffer, size:int) -> int:
    obj = kgsl.struct_kgsl_command_object(gpuaddr=command.va_addr, size=size, flags=kgsl.KGSL_CMDLIST_IB)
    req = kgsl.struct_kgsl_gpu_command(cmdlist=ctypes.addressof(obj), numcmds=1, context_id=self.ctx,
                                       cmdsize=ctypes.sizeof(kgsl.struct_kgsl_command_object))
    return kgsl.IOCTL_KGSL_GPU_COMMAND(self.fd, __payload=req).timestamp

  def sleep(self, time_spent_since_last_sleep_ms:int):
    kgsl.IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID(self.fd, context_id=self.ctx, timestamp=self.dev.last_cmd, timeout=0xffffffff)

  def profile_finalize(self):
    with contextlib.suppress(RuntimeError): System.write_sysfs("/sys/class/kgsl/kgsl-3d0/idle_timer", "10", "Failed to reenable suspend mode")

@dataclass
class MSMAllocation:
  handle: int
  iova: int
  size: int
  cpu_addr: int|None = None
  refcount: int = 1

def _open_msm_render_node(path:str) -> FileIOInterface|None:
  try: fd = FileIOInterface(path, os.O_RDWR)
  except OSError: return None
  name = (ctypes.c_ubyte * 16)()
  try: version = msm_drm.DRM_IOCTL_VERSION(fd, name_len=len(name), name=ctypes.cast(name, ctypes.POINTER(ctypes.c_ubyte)))
  except OSError: return None
  return fd if bytes(name[:version.name_len]) == b"msm" else None

class MSMIface:
  count = 1
  renderers = [IR3Renderer]
  event_write_irq = True

  def __init__(self, dev:QCOMDevice, device_id:int):
    if device_id != 0: raise RuntimeError(f"QCOM:{device_id} does not exist (1 MSM DRM device available)")
    self.dev = dev

    for path in sorted(glob.glob("/dev/dri/renderD*")):
      if (fd:=_open_msm_render_node(path)) is not None:
        self.fd = fd
        break
    else: raise RuntimeError("No MSM DRM render node found")

    msm_drm.DRM_IOCTL_MSM_SET_PARAM(self.fd, pipe=msm_drm.MSM_PIPE_3D0, param=msm_drm.MSM_PARAM_EN_VM_BIND, value=1)
    self.chip_id = msm_drm.DRM_IOCTL_MSM_GET_PARAM(self.fd, pipe=msm_drm.MSM_PIPE_3D0, param=msm_drm.MSM_PARAM_CHIP_ID).value
    self.mesa_gpu_id = msm_drm.DRM_IOCTL_MSM_GET_PARAM(self.fd, pipe=msm_drm.MSM_PIPE_3D0, param=msm_drm.MSM_PARAM_GPU_ID).value
    chip_id = self.chip_id & 0xffffffff
    self.gpu_id = (chip_id >> 24, (chip_id >> 16) & 0xff, (chip_id >> 8) & 0xff)
    if self.gpu_id not in {(6, 3, 0), (6, 3, 5)}:
      raise RuntimeError(f"MSM DRM requires a validated Adreno 630/635, got chip_id={self.chip_id:#x}")
    self.submit_flags = msm_drm.MSM_PIPE_3D0 | (msm_drm.MSM_SUBMIT_SUDO if self.gpu_id == (6, 3, 5) else 0)
    va_start = msm_drm.DRM_IOCTL_MSM_GET_PARAM(self.fd, pipe=msm_drm.MSM_PIPE_3D0, param=msm_drm.MSM_PARAM_VA_START).value
    va_size = msm_drm.DRM_IOCTL_MSM_GET_PARAM(self.fd, pipe=msm_drm.MSM_PIPE_3D0, param=msm_drm.MSM_PARAM_VA_SIZE).value
    self.va_allocator = TLSFAllocator(va_size, base=va_start, block_size=mmap.PAGESIZE)
    self.vm_bind_queue_id = msm_drm.DRM_IOCTL_MSM_SUBMITQUEUE_NEW(self.fd, flags=msm_drm.MSM_SUBMITQUEUE_VM_BIND, prio=0).id
    self.queue_id = msm_drm.DRM_IOCTL_MSM_SUBMITQUEUE_NEW(self.fd, flags=0, prio=0).id
    self.allocations: dict[int, MSMAllocation] = {}

  def _vm_bind(self, op:int, allocation:MSMAllocation):
    bind_op = msm_drm.struct_drm_msm_vm_bind_op(op=op, handle=allocation.handle if op == msm_drm.MSM_VM_BIND_OP_MAP else 0,
                                                iova=allocation.iova, range=allocation.size)
    bind = msm_drm.struct_drm_msm_vm_bind(flags=msm_drm.MSM_VM_BIND_FENCE_FD_OUT, nr_ops=1, fence_fd=-1,
                                          queue_id=self.vm_bind_queue_id, op_stride=ctypes.sizeof(bind_op), op=bind_op)
    msm_drm.DRM_IOCTL_MSM_VM_BIND(self.fd, __payload=bind)
    try: select.select([bind.fence_fd], [], [])
    finally: os.close(bind.fence_fd)

  def _new_allocation(self, handle:int, size:int) -> MSMAllocation:
    try: return MSMAllocation(handle, self.va_allocator.alloc(size, mmap.PAGESIZE), size)
    except Exception:
      msm_drm.DRM_IOCTL_GEM_CLOSE(self.fd, handle=handle)
      raise

  def _release(self, allocation:MSMAllocation):
    if allocation.cpu_addr is not None: self.fd.munmap(allocation.cpu_addr, allocation.size)
    msm_drm.DRM_IOCTL_GEM_CLOSE(self.fd, handle=allocation.handle)
    self.va_allocator.free(allocation.iova)

  def alloc(self, size:int, uncached=False, fill_zeroes=False) -> HCQBuffer:
    if size <= 0: raise ValueError(f"MSM allocation size must be positive, got {size}")
    mapped_size = round_up(size, mmap.PAGESIZE)
    gem = msm_drm.DRM_IOCTL_MSM_GEM_NEW(self.fd, size=mapped_size, flags=msm_drm.MSM_BO_WC)
    allocation = self._new_allocation(gem.handle, mapped_size)
    try:
      offset = msm_drm.DRM_IOCTL_MSM_GEM_INFO(self.fd, handle=gem.handle, info=msm_drm.MSM_INFO_GET_OFFSET).value
      allocation.cpu_addr = self.fd.mmap(0, mapped_size, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, offset)
      if fill_zeroes: ctypes.memset(allocation.cpu_addr, 0, size)
      self._vm_bind(msm_drm.MSM_VM_BIND_OP_MAP, allocation)
    except Exception:
      self._release(allocation)
      raise
    self.allocations[gem.handle] = allocation
    return HCQBuffer(allocation.iova, size, meta=allocation, view=MMIOInterface(allocation.cpu_addr, size), owner=self.dev)

  def map(self, ptr:int, size:int, fd:int|None=None) -> HCQBuffer:
    if fd is None: raise ValueError("MSM DRM external pointers require a DMA-BUF fd")
    if size <= 0: raise ValueError(f"MSM mapping size must be positive, got {size}")
    if size > (dma_buf_size:=os.fstat(fd).st_size): raise ValueError(f"Mapping size {size} exceeds DMA-BUF size {dma_buf_size}")
    imported = msm_drm.DRM_IOCTL_PRIME_FD_TO_HANDLE(self.fd, fd=fd)
    if (allocation:=self.allocations.get(imported.handle)) is None:
      allocation = self._new_allocation(imported.handle, round_up(dma_buf_size, mmap.PAGESIZE))
      try:
        self._vm_bind(msm_drm.MSM_VM_BIND_OP_MAP, allocation)
      except Exception:
        self._release(allocation)
        raise
      self.allocations[imported.handle] = allocation
    else: allocation.refcount += 1
    return HCQBuffer(allocation.iova, size, meta=allocation, view=MMIOInterface(ptr, size), owner=self.dev)

  @staticmethod
  def _allocation(mem:HCQBuffer) -> MSMAllocation:
    if not isinstance(allocation:=mem.base.meta, MSMAllocation): raise RuntimeError("MSM buffer was not allocated by the MSM DRM interface")
    return allocation

  def free(self, mem:HCQBuffer):
    allocation = self._allocation(mem)
    if allocation.refcount > 1:
      allocation.refcount -= 1
      return
    self._vm_bind(msm_drm.MSM_VM_BIND_OP_UNMAP, allocation)
    self._release(allocation)
    del self.allocations[allocation.handle]

  def submit(self, command:HCQBuffer, size:int) -> int:
    cmd = msm_drm.struct_drm_msm_gem_submit_cmd(type=msm_drm.MSM_SUBMIT_CMD_BUF, size=size, iova=int(command.va_addr))
    submit = msm_drm.struct_drm_msm_gem_submit(flags=self.submit_flags, nr_cmds=1, cmds=ctypes.addressof(cmd), queueid=self.queue_id)
    msm_drm.DRM_IOCTL_MSM_GEM_SUBMIT(self.fd, __payload=submit)
    return submit.fence

  def sleep(self, _time_spent_since_last_sleep_ms:int):
    if self.dev.last_cmd == 0: return
    tv_sec, tv_nsec = divmod(time.monotonic_ns() + MSM_WAIT_SLICE_NS, 1_000_000_000)
    timeout = msm_drm.struct_drm_msm_timespec(tv_sec=tv_sec, tv_nsec=tv_nsec)
    try: msm_drm.DRM_IOCTL_MSM_WAIT_FENCE(self.fd, fence=self.dev.last_cmd, flags=0, timeout=timeout, queueid=self.queue_id)
    except OSError as e:
      if e.errno not in {errno.EINTR, errno.ETIMEDOUT}: raise RuntimeError("MSM fence wait failed") from e

  def device_fini(self):
    msm_drm.DRM_IOCTL_MSM_SUBMITQUEUE_CLOSE(self.fd, self.queue_id)
    msm_drm.DRM_IOCTL_MSM_SUBMITQUEUE_CLOSE(self.fd, self.vm_bind_queue_id)

class QCOMDevice(HCQCompiled):
  ifaces = [KGSLIface, MSMIface]

  def __init__(self, device:str=""):
    self.device_id = int(device.split(":")[1]) if ":" in device else 0
    self.iface = self._select_iface()
    self.gpu_id = self.iface.gpu_id

    # a7xx start with 730x or 'Cxxx', a8xx starts 'Exxx'
    if self.gpu_id[:2] >= (7, 3): raise RuntimeError(f"Unsupported GPU: chip_id={self.iface.chip_id:#x}")

    self.dummy_buf = self.iface.alloc(0x1000)
    self.dummy_addr = int(self.dummy_buf.va_addr)
    self.cmd_buf = self.iface.alloc(16 << 20)
    self.cmd_buf_allocator = BumpAllocator(size=self.cmd_buf.size, base=int(self.cmd_buf.va_addr), wrap=True)
    self.border_color_buf = self.iface.alloc(0x1000, fill_zeroes=True)
    self.last_cmd:int = 0

    mesa_gpu_id = getattr(self.iface, "mesa_gpu_id", self.gpu_id[0] * 100 + self.gpu_id[1] * 10 + self.gpu_id[2])
    self.dev_info = mesa.fd_dev_info(mesa.struct_fd_dev_id(mesa_gpu_id, self.iface.chip_id))
    arch = ("a%d%d%d,GPU_ID=%d,CHIP_ID=%#x" % (*self.gpu_id, mesa_gpu_id, self.iface.chip_id)) + \
           (",IMAGE_PITCH_ALIGNMENT=64" if IMAGE else "")
    super().__init__(device, QCOMAllocator(self), self.iface.renderers, QCOMProgram, QCOMSignal, functools.partial(QCOMComputeQueue, self), arch=arch)

  def _ensure_stack_size(self, sz):
    if not hasattr(self, '_stack'): self._stack = self.iface.alloc(sz)
    elif self._stack.size < sz:
      self.synchronize()
      self.iface.free(self._stack)
      self._stack = self.iface.alloc(sz)

  def _at_profile_finalize(self):
    super()._at_profile_finalize()
    if hasattr(self.iface, "profile_finalize"): self.iface.profile_finalize()
