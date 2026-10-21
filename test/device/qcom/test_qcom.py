import ctypes, platform, struct, unittest
from types import SimpleNamespace
from unittest.mock import patch
from tinygrad import Device
from tinygrad.dtype import dtypes
from tinygrad.helpers import round_up
from tinygrad.renderer.cstyle import ClangRenderer
from tinygrad.runtime.autogen import mesa
from tinygrad.uop.ops import UOp
from tinygrad.uop.spec import spec_full, type_verify


class TestQCOM(unittest.TestCase):
  def test_msm_submit_wait_validates(self):
    from tinygrad.runtime.ops_qcom import QCOMComputeQueue, MSMIface
    queue = object.__new__(QCOMComputeQueue)
    queue.devs = ('QCOM',)
    queue.dev = SimpleNamespace(iface=object.__new__(MSMIface), tag=lambda name: ('QCOM', name))
    cmdbuf = UOp.alloc((128,), dtypes.uint8, device='QCOM')
    type_verify(queue.submit(cmdbuf).sink(), spec_full)

  def test_private_memory_register_units(self):
    from tinygrad.runtime.ops_qcom import _qcom_pvtmem_sizes
    self.assertEqual(_qcom_pvtmem_sizes(0, 4096, 2), (0, 0, 0, 0x1000))
    self.assertEqual(_qcom_pvtmem_sizes(1, 4096, 2), (1, 512, 1024, 4 << 20))
    self.assertEqual(_qcom_pvtmem_sizes(513, 4096, 2), (2, 1024, 2048, 8 << 20))

  def test_a635_compute_registers(self):
    from tinygrad.runtime.ops_qcom import QCOMComputeQueue, MSMIface, pkt4_hdr, pkt7_hdr
    queue = object.__new__(QCOMComputeQueue)
    queue.devs, queue.blob, queue.patches = ('QCOM',), bytearray(), []
    queue.dev = SimpleNamespace(tag=lambda name: ('QCOM', name), gpu_id=(6, 3, 5), iface=object.__new__(MSMIface),
                               dev_info=SimpleNamespace(a6xx=SimpleNamespace(supports_double_threadsize=False, has_lpac=True)))
    data = SimpleNamespace(NIR=True, wgsz=1, hregs=0, fregs=0, brnchstck=0, shared_size=1, prg_offset=0,
                           pvtmem_size_per_item=0, pvtmem_size_total=0, pvtmem_per_wave=False, stack_size=4096,
                           hw_stack_offset=0, image_size=0x3280, constlen=28, double_threadsize=False, early_preamble=False,
                           mergedregs=False, samp_cnt=0, tex_cnt=0, ibo_cnt=0, wgid=0xfc, lid=0xfc, max_threads=1024, kernargs_alloc_size=2048)
    prg = SimpleNamespace(arg=SimpleNamespace(global_size=(2, 3, 3), local_size=(2, 3, 4)))
    buf = UOp.alloc((4096,), dtypes.uint8, device='QCOM')
    with patch('tinygrad.runtime.ops_qcom.qcom_build_program', return_value=(data, buf)), patch.object(queue, 'kernargs', return_value=buf):
      queue.exec(None, prg)
    words = list(struct.unpack(f'{len(queue.blob) // 4}I', queue.blob))
    instr = words.index(pkt4_hdr(mesa.REG_A6XX_SP_CS_INSTR_SIZE, 1))
    const = words.index(pkt4_hdr(mesa.REG_A6XX_SP_REG_PROG_ID_0, 5))
    execute = words.index(pkt7_hdr(mesa.CP_EXEC_CS, 4))
    self.assertEqual(words[instr + 1], round_up(data.image_size, 128) // 128)
    self.assertEqual(words[const + 5], mesa.A6XX_SP_CS_CONST_CONFIG_ENABLED | 7)
    self.assertEqual(words[execute + 1:execute + 5], [0, 2, 3, 3])
    self.assertIn(pkt4_hdr(mesa.REG_A6XX_HLSQ_CS_CTRL_REG1, 1), words)
    self.assertIn(pkt4_hdr(mesa.REG_A6XX_SP_PS_WAVE_CNTL, 1), words)
    self.assertIn(pkt4_hdr(mesa.REG_A6XX_SP_CS_WIE_CNTL_0, 2), words)
    flush = words.index(pkt7_hdr(mesa.CP_EVENT_WRITE, 4))
    self.assertEqual(words[flush + 1], mesa.CACHE_FLUSH_TS)

  def test_msm_barriers_and_signals_do_not_request_irqs(self):
    from tinygrad.runtime.ops_qcom import QCOMComputeQueue, MSMIface
    queue = object.__new__(QCOMComputeQueue)
    queue.devs, queue.blob, queue.patches = ('QCOM',), bytearray(), []
    queue.dev = SimpleNamespace(tag=lambda name: ('QCOM', name), gpu_id=(6, 3, 5), iface=object.__new__(MSMIface))
    queue.memory_barrier()
    queue.signal(UOp.alloc((2,), dtypes.uint64, device='QCOM'), UOp.const(7, dtypes.uint64))
    words = list(struct.unpack(f'{len(queue.blob) // 4}I', queue.blob))
    packets = []
    while words:
      header, *words = words
      count, opcode = header & 0x3fff, (header >> 16) & 0x7f
      packets.append((opcode, words[:count]))
      words = words[count:]
    events = [payload[0] for opcode, payload in packets if opcode == mesa.CP_EVENT_WRITE]
    self.assertEqual(events, [mesa.CACHE_FLUSH_TS, mesa.CACHE_INVALIDATE, mesa.CACHE_FLUSH_TS, mesa.CACHE_FLUSH_TS])
    self.assertIn((mesa.CP_WAIT_MEM_WRITES, []), packets)
    self.assertIn((mesa.CP_WAIT_FOR_IDLE, []), packets)

  # although part of the QCOM runtime, this tests flushing the CPU's dcache
  @unittest.skipUnless(isinstance(Device["CPU"].renderer, ClangRenderer) and platform.machine().lower() in {"arm64", "aarch64"},
                       "dcache_flush's inline asm needs ClangRenderer, and runs on arm64")
  def test_dcache_flush(self):
    from tinygrad.runtime.ops_qcom import dcache_flush
    buf = (ctypes.c_uint8 * 64)()
    dcache_flush().fxn(buf, 0)

if __name__ == '__main__':
  unittest.main()
