# MSM DRM Runtime

The MSM runtime (`tinygrad/runtime/ops_msm.py`) runs tinygrad on Qualcomm Adreno GPUs via the upstream Linux MSM DRM kernel driver (`/dev/dri/renderD*`). This is the mainline Linux path, as opposed to the QCOM runtime which uses the Android-only KGSL driver (`/dev/kgsl-3d0`).

Both backends target the same GPU hardware and use the same IR3 shader compiler from Mesa, but they differ in how they talk to the kernel.

## Architecture

```
                  QCOM (Android)              MSM (mainline Linux)
                  ──────────────              ────────────────────
Shader compiler:  IR3Renderer + IR3Compiler   IR3Renderer + IR3Compiler  (same)
GPU commands:     PM4 packets (a6xx regs)     PM4 packets (a6xx regs)    (same)
Base class:       HCQCompiled                 Compiled
Submission:       KGSL ring buffer + ioctl    DRM_MSM_GEM_SUBMIT ioctl
Memory:           KGSL GPUOBJ_ALLOC           DRM GEM_NEW + GEM_INFO
Sync:             KGSL WAITTIMESTAMP           DRM WAIT_FENCE
Device node:      /dev/kgsl-3d0               /dev/dri/renderD*
```

The key difference: KGSL supports hardware command queues (HCQ) with direct ring buffer access and doorbell writes. MSM DRM is purely ioctl-based, so `MSMDevice` inherits from `Compiled` instead of `HCQCompiled`.

## Components

### MSMAllocator

Allocates GPU memory via `DRM_MSM_GEM_NEW`, maps to CPU via `DRM_MSM_GEM_INFO(GET_OFFSET)` + `mmap`, and gets the GPU virtual address (IOVA) via `DRM_MSM_GEM_INFO(GET_IOVA)`.

Buffer pool design: GEM handles are never closed during runtime. The `_free()` method is a no-op. All GEM BOs are tracked in `_all_buffers` and only closed in `finalize()` (or implicitly when the DRM fd closes at process exit). This prevents GPU IOMMU translation faults caused by IOVA unmapping races when GEM handles are closed while the GPU's TLB still holds stale entries.

The LRU cache from `LRUAllocator` still works for buffer reuse, it just never actually closes anything.

### MSMProgram

Handles a single kernel dispatch. On each `__call__`:

1. Fills an args buffer (UBO pointers, constants, texture/sampler descriptors)
2. Builds a PM4 command stream via `_build_pm4()` (identical register programming to `QCOMComputeQueue.exec()`)
3. Submits via `DRM_MSM_GEM_SUBMIT` with the command buffer and all referenced BOs

The PM4 command stream includes:
- Cache invalidate (so GPU sees fresh CPU writes)
- Compute mode setup (SP_UPDATE_CNTL, mode registers)
- Work dimensions (SP_CS_NDRANGE)
- Shader program config (registers, stack, private memory)
- Constant/shader/texture/sampler/UAV state loads (CP_LOAD_STATE6_FRAG)
- Dispatch (CP_EXEC_CS for NIR shaders)
- Cache flush (write-back + invalidate + wait)

### MSMGraph

Batches all JIT-captured kernels into a single `DRM_MSM_GEM_SUBMIT`. This is the MSM equivalent of `HCQGraph`.

On init, pre-builds all args buffers and PM4 for every kernel in the JIT cache, concatenates the PM4 into one command buffer, and records patch locations where input buffer IOVAs are written.

On each call, only patches the changed input IOVAs (a few `struct.pack_into` calls) and does one ioctl. This brings enqueue overhead from ~135ms (78 separate ioctls) down to ~8ms (single ioctl + IOVA patching).

### Autogen

`tinygrad/runtime/autogen/msm_drm.py` is generated from `extra/qcom_gpu_driver/msm_drm.h` (the kernel UAPI header) plus `/usr/include/drm/drm.h`. Provides ctypes bindings for all MSM DRM ioctls and structs.

## Key ioctls

| ioctl | Purpose |
|-------|---------|
| `DRM_MSM_GET_PARAM` | Query GPU ID, chip ID, GMEM size |
| `DRM_MSM_GEM_NEW` | Allocate GEM buffer |
| `DRM_MSM_GEM_INFO` | Get mmap offset, get/set IOVA |
| `DRM_MSM_GEM_SUBMIT` | Submit command buffer(s) to GPU |
| `DRM_MSM_WAIT_FENCE` | Wait for GPU completion |
| `DRM_MSM_SUBMITQUEUE_NEW` | Create a submission queue |
| `DRM_IOCTL_GEM_CLOSE` | Close GEM handle (only on finalize) |

## Bugs fixed along the way

**cstyle.py aux() IndexError**: The `aux()` function in `OpenCLRenderer` assumed PARAM indices are contiguous. IR3 can produce gaps (e.g., args=[0, 2] skipping 1) when buffers are optimized away. Fixed with `while len(arg_dtypes) <= u.arg: arg_dtypes.append([])`.

**Null-pointer GPU faults**: Unused UBO slots in the args buffer contained zero. The shader would dereference address 0x0, causing a GPU translation fault. Fixed by pre-filling all UBO slots with the dummy buffer's IOVA.

**IOMMU translation faults from LRU eviction**: When `LRUAllocator.free_cache()` called `GEM_CLOSE`, the kernel unmapped the IOVA from the GPU's IOMMU. But the GPU's TLB could still hold stale entries, causing faults on subsequent submits. Fixed by making `_free()` a no-op (buffer pool pattern).

## Performance

compile3.py dmonitoring_model, `FLOAT16=1 IMAGE=1 NOLOCALS=1`:

| Backend | Enqueue (ms) | Total run (ms) |
|---------|-------------|----------------|
| QCOM OpenCL (KGSL, HCQ) | ~5 | ~23 |
| MSM DRM (graph) | ~9 | ~20 |

compile3.py driving_vision:

| Backend | Enqueue (ms) | Total run (ms) |
|---------|-------------|----------------|
| QCOM OpenCL (KGSL, HCQ) | ~2 | ~37 |
| MSM DRM (graph) | ~9 | ~30 |

## Running

```bash
DEV=MSM MESA_PATH=/path/to/libtinymesa.so python3 your_script.py
```

Requires:
- Linux with MSM DRM driver (`CONFIG_DRM_MSM=y`)
- Adreno a6xx GPU (tested on Adreno 630, SDM845)
- `libtinymesa.so` (Mesa's IR3 compiler, built from Mesa source)
- `/usr/include/drm/drm.h` (for autogen regeneration only)
