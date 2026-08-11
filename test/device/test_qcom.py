import ctypes, itertools, math, platform, struct, unittest
from types import SimpleNamespace
from unittest.mock import patch

from tinygrad import Device
from tinygrad.device import BufferSpec, TinyELF
from tinygrad.dtype import dtypes
from tinygrad.helpers import Target, mv_address, round_up
from tinygrad.renderer.cstyle import ClangRenderer
from tinygrad.runtime.support.hcq import HCQBuffer, HWQueue, MMIOInterface


class FakeAllocator:
  def __init__(self): self.allocations = []

  def alloc(self, size, options=None):
    gpu_memory, cpu_memory = bytearray([0xaa] * size), bytearray([0xaa] * size)
    buf = HCQBuffer(mv_address(memoryview(gpu_memory)), size, view=MMIOInterface(mv_address(memoryview(cpu_memory)), size))
    self.allocations.append((gpu_memory, cpu_memory, buf))
    return buf

  def free(self, *args): pass


def make_compute_state(image_size=0x3280):
  args_gpu, args_cpu = bytearray([0xaa] * 32), bytearray([0xaa] * 32)
  args = HCQBuffer(mv_address(memoryview(args_gpu)), len(args_gpu),
                   view=MMIOInterface(mv_address(memoryview(args_cpu)), len(args_cpu)))
  dev = SimpleNamespace(
    gpu_id=(6, 3, 5), iface=SimpleNamespace(event_write_irq=True),
    dev_info=SimpleNamespace(a6xx=SimpleNamespace(supports_double_threadsize=False, has_lpac=True)),
    dummy_addr=0x300000, _stack=HCQBuffer(0x400000, 4096), border_color_buf=HCQBuffer(0x500000, 4096),
  )
  prg = SimpleNamespace(
    dev=dev, NIR=True, wgsz=1, hregs=0, fregs=0, brnchstck=0, shared_size=1, prg_offset=0,
    lib_gpu=HCQBuffer(0x200000, image_size), pvtmem_size_per_item=0, pvtmem_size_total=0, pvtmem_per_wave=False,
    hw_stack_offset=0, image_size=image_size, constlen=28, double_threadsize=False, early_preamble=False, mergedregs=False,
    samp_cnt=0, tex_cnt=0, ibo_cnt=0, wgid=0xfc, lid=0xfc,
  )
  return dev, prg, SimpleNamespace(bind_data=[], buf=args, prg=prg, bufs=()), args_gpu, args_cpu


class TestQCOM(unittest.TestCase):
  def test_default_local_size_is_bounded_and_divisible(self):
    from tinygrad.engine.realize import _qcom_default_local_size

    self.assertEqual(_qcom_default_local_size((2500, 5, 5), 1024), (125, 1, 1))
    self.assertEqual(_qcom_default_local_size((5, 12, 32), 256), (5, 12, 2))
    for global_size, max_threads in [((1, 1, 1), 1024), ((17, 19, 23), 64), ((512, 256, 6), 32)]:
      local_size = _qcom_default_local_size(global_size, max_threads)
      self.assertLessEqual(math.prod(local_size), min(128, max_threads))
      self.assertTrue(all(g % l == 0 for g,l in zip(global_size, local_size)))

  def test_private_memory_register_units(self):
    from tinygrad.runtime.ops_qcom import _qcom_pvtmem_sizes

    self.assertEqual(_qcom_pvtmem_sizes(0, 4096, 2), (0, 0, 0, 0x1000))
    self.assertEqual(_qcom_pvtmem_sizes(1, 4096, 2), (1, 512, 1024, 4 << 20))
    self.assertEqual(_qcom_pvtmem_sizes(513, 4096, 2), (2, 1024, 2048, 8 << 20))

  def test_args_use_cpu_view(self):
    from tinygrad.runtime.ops_qcom import QCOMArgsState

    gpu_memory, cpu_memory = bytearray([0xaa] * 64), bytearray([0xaa] * 64)
    args = HCQBuffer(mv_address(memoryview(gpu_memory)), len(gpu_memory),
                     view=MMIOInterface(mv_address(memoryview(cpu_memory)), len(cpu_memory)))
    data = HCQBuffer(0x123456789abcdef0, 16)
    prg = SimpleNamespace(kernargs_alloc_size=64, signature=((None, 0, dtypes.float32, (1,)), (None, 1, dtypes.uint32, ())),
                          ibo_cnt=0, tex_cnt=0, samp_cnt=0, NIR=True, tex_to_image=[], consts_info=[(0x12345678, 24, 4)],
                          buf_off=8, tex_off=64, ibo_off=64, samplers=[])

    state = QCOMArgsState(args, prg, (data,), vals=(0x87654321,))
    HWQueue().bind_args_state(state)

    self.assertEqual(cpu_memory[:8], bytes(8))
    self.assertEqual(int.from_bytes(cpu_memory[8:16], "little"), data.va_addr)
    self.assertEqual(int.from_bytes(cpu_memory[16:20], "little"), 0x87654321)
    self.assertEqual(int.from_bytes(cpu_memory[24:28], "little"), 0x12345678)
    self.assertEqual(cpu_memory[28:], bytes(36))
    self.assertEqual(gpu_memory, bytes([0xaa] * 64))

  def test_program_upload_uses_cpu_view(self):
    from tinygrad.runtime.ops_qcom import QCOMProgram

    lib, image, image_offset, image_desc_offset, reg_desc_offset = bytearray(0x500), bytes(range(129)), 0x400, 0x180, 0x300
    struct.pack_into("I", lib, 0x100, len(image))
    struct.pack_into("I", lib, 0xc0, image_offset)
    struct.pack_into("I", lib, 0x110, image_desc_offset)
    struct.pack_into("I", lib, 0x34, reg_desc_offset)
    struct.pack_into("I", lib, reg_desc_offset + 0x14, 1)
    lib[image_offset:image_offset+len(image)] = image
    allocator = FakeAllocator()
    dev = SimpleNamespace(device="QCOM", renderer=object(), allocator=allocator, prof_prg_counter=itertools.count(),
                          dev_info=SimpleNamespace(fibers_per_sp=4096, num_sp_cores=2), _ensure_stack_size=lambda size: None)

    QCOMProgram(dev, TinyELF(bytes(lib), "test", Target("QCOM"), ()))

    gpu_memory, cpu_memory, _ = allocator.allocations[0]
    self.assertEqual(cpu_memory, image)
    self.assertEqual(gpu_memory, bytes([0xaa] * len(image)))

  def test_a635_compute_registers_and_cpu_view(self):
    from tinygrad.runtime.autogen import mesa
    from tinygrad.runtime.ops_qcom import QCOMComputeQueue, pkt4_hdr, pkt7_hdr

    dev, prg, state, args_gpu, args_cpu = make_compute_state()
    queue = QCOMComputeQueue(dev).exec(prg, state, (1.25, 2.5, 3), (2, 3, 4))

    self.assertEqual(args_cpu[4:16], struct.pack("III", 2, 3, 4))
    self.assertEqual(args_gpu, bytes([0xaa] * len(args_gpu)))
    instr = queue._q.index(pkt4_hdr(mesa.REG_A6XX_SP_CS_INSTR_SIZE, 1))
    const = queue._q.index(pkt4_hdr(mesa.REG_A6XX_SP_REG_PROG_ID_0, 5))
    execute = queue._q.index(pkt7_hdr(mesa.CP_EXEC_CS, 4))
    self.assertEqual(queue._q[instr + 1], round_up(prg.image_size, 128) // 128)
    self.assertEqual(queue._q[const + 5], mesa.A6XX_SP_CS_CONST_CONFIG_ENABLED | 7)
    self.assertEqual(queue._q[execute + 1:execute + 5], [0, 2, 3, 3])
    self.assertIn(pkt4_hdr(mesa.REG_A6XX_HLSQ_CS_CTRL_REG1, 1), queue._q)
    self.assertIn(pkt4_hdr(mesa.REG_A6XX_SP_PS_WAVE_CNTL, 1), queue._q)

  def test_bound_queue_uses_cpu_view_and_interface(self):
    from tinygrad.runtime.ops_qcom import QCOMComputeQueue

    allocator, submissions = FakeAllocator(), []
    iface = SimpleNamespace(submit=lambda command, size: submissions.append((command, size)) or 42)
    dev = SimpleNamespace(allocator=allocator, iface=iface)
    queue = QCOMComputeQueue(dev)
    queue.q(0x12345678, 0x9abcdef0)

    queue.bind(dev)
    queue.submit(dev)

    gpu_memory, cpu_memory, command = allocator.allocations[0]
    self.assertEqual(cpu_memory, struct.pack("II", 0x12345678, 0x9abcdef0))
    self.assertEqual(gpu_memory, bytes([0xaa] * len(gpu_memory)))
    self.assertEqual(submissions, [(command, 8)])
    self.assertEqual(dev.last_cmd, 42)

  def test_allocator_uses_interface(self):
    from tinygrad.runtime.ops_qcom import QCOMAllocator

    allocated, mapped, freed, maps = HCQBuffer(0x1000, 16), HCQBuffer(0x2000, 16), [], []
    iface = SimpleNamespace(alloc=lambda size, uncached=False: allocated,
                            map=lambda ptr, size: maps.append((ptr, size)) or mapped,
                            free=lambda buf: freed.append(buf))
    allocator = object.__new__(QCOMAllocator)
    allocator.dev = SimpleNamespace(iface=iface)

    self.assertIs(allocator._alloc(16, BufferSpec()), allocated)
    self.assertIs(allocator._alloc(16, BufferSpec(external_ptr=0x1234)), mapped)
    self.assertEqual(maps, [(0x1234, 16)])
    allocator._do_free(allocated, BufferSpec())
    self.assertEqual(freed, [allocated])

  def test_ir3_compiler_accepts_a635_device_id(self):
    from tinygrad.runtime.autogen import mesa
    from tinygrad.runtime.support.compiler_mesa import IR3Compiler

    cc, opts = mesa.struct_ir3_compiler(), mesa.struct_nir_shader_compiler_options()
    with patch.object(mesa, "fd_dev_info", return_value=mesa.struct_fd_dev_info()), \
         patch.object(mesa, "ir3_compiler_create", return_value=ctypes.pointer(cc)), \
         patch.object(mesa, "ir3_get_compiler_options", return_value=ctypes.pointer(opts)), \
         patch.object(mesa, "ir3_compiler_destroy"):
      compiler = IR3Compiler("a635,GPU_ID=0,CHIP_ID=0xac06030500")
      self.assertEqual((compiler.dev_id.gpu_id, compiler.dev_id.chip_id), (0, 0xac06030500))
      del compiler

  def test_signal_sleep_uses_interface(self):
    from tinygrad.runtime.ops_qcom import QCOMSignal

    memory, sleeps = bytearray(16), []
    owner = SimpleNamespace(iface=SimpleNamespace(sleep=lambda timeout: sleeps.append(timeout)))
    signal = QCOMSignal(HCQBuffer(0x1000, 16, view=MMIOInterface(mv_address(memoryview(memory)), 16)),
                        owner=owner, is_timeline=True, virt=True)
    signal._sleep(7)
    self.assertEqual(sleeps, [7])

  # although part of the QCOM runtime, this tests flushing the CPU's dcache
  @unittest.skipUnless(isinstance(Device["CPU"].renderer, ClangRenderer) and platform.machine().lower() in {"arm64", "aarch64"},
                       "dcache_flush's inline asm needs ClangRenderer, and runs on arm64")
  def test_dcache_flush(self):
    from tinygrad.runtime.ops_qcom import dcache_flush
    buf = (ctypes.c_uint8 * 64)()
    dcache_flush().fxn(buf, 0)


if __name__ == '__main__':
  unittest.main()
