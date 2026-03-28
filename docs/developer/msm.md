# Adreno GPU on Mainline Linux (MSM DRM)

There are two ways to run tinygrad on Qualcomm Adreno GPUs:

| | `DEV=QCOM` | `DEV=MSM` |
|---|---|---|
| Kernel driver | KGSL (`/dev/kgsl-3d0`) | MSM DRM (`/dev/dri/renderD*`) |
| Found on | Android, Qualcomm vendor kernels | Mainline/upstream Linux |
| Shader compiler | OpenCL (default) or IR3 (`QCOM_IR3=1`) | IR3 only |

Use `DEV=QCOM` if you have `/dev/kgsl-3d0` (Android devices, comma body). Use `DEV=MSM` if you have `/dev/dri/renderD*` with an Adreno GPU (mainline Linux kernel).

## How to run?

```bash
DEV=MSM MESA_PATH=/path/to/libtinymesa.so python3 your_script.py
```

Requirements:

* Linux kernel with `CONFIG_DRM_MSM=y` (GPU support, not just display)
* Adreno a6xx GPU (tested: Adreno 630 / SDM845)
* `libtinymesa.so` built from Mesa (provides the IR3 shader compiler, used by both `QCOM_IR3=1` and `DEV=MSM`)

## Architecture

MSM DRM is ioctl-based (no hardware command queues), so `MSMDevice` inherits from `Compiled` rather than `HCQCompiled`. The shader compiler and PM4 command stream are shared with the QCOM/KGSL backend.

| | QCOM (KGSL) | MSM (DRM) |
|---|---|---|
| Base class | `HCQCompiled` | `Compiled` |
| Submission | Ring buffer + doorbell | `DRM_MSM_GEM_SUBMIT` ioctl |
| Memory | `IOCTL_KGSL_GPUOBJ_ALLOC` | `DRM_MSM_GEM_NEW` + `DRM_MSM_GEM_INFO` |
| Sync | `IOCTL_KGSL_DEVICE_WAITTIMESTAMP` | `DRM_MSM_WAIT_FENCE` |
| Shader compiler | IR3 (Mesa) | IR3 (Mesa) |
| PM4 packets | a6xx registers | a6xx registers (same) |

## MSM DRM Details

### Memory Management

GEM buffers are allocated with `DRM_MSM_GEM_NEW`, CPU-mapped via the offset from `DRM_MSM_GEM_INFO(GET_OFFSET)`, and assigned a GPU virtual address (IOVA) by the kernel via `DRM_MSM_GEM_INFO(GET_IOVA)`.

GEM handles are never closed during runtime. Closing a GEM handle unmaps its IOVA in the GPU's IOMMU, but the GPU's TLB can retain stale entries that cause translation faults on subsequent submits. Instead, all handles are tracked and closed only in `finalize()` (or when the DRM fd closes at process exit).

### Command Submission

Each kernel dispatch builds a PM4 command stream (same register programming as `QCOMComputeQueue.exec()`) and submits it via `DRM_MSM_GEM_SUBMIT` with a BO table listing all referenced GEM handles.

### MSMGraph

`MSMGraph` implements `GraphRunner` for JIT replay. It pre-builds all PM4 command streams and args buffers during init, concatenates them into a single command buffer, and submits everything with one `DRM_MSM_GEM_SUBMIT` ioctl.

On each JIT replay call, only the input buffer IOVAs are patched in the pre-built args buffers (a few `struct.pack_into` writes). This brings enqueue overhead from ~135ms (N separate ioctls) down to ~9ms (IOVA patching + single ioctl).

### UBO Slot Pre-fill

Unused UBO slots in the args buffer are pre-filled with a dummy buffer IOVA. Without this, the shader reads address 0x0 for optimized-away parameters, causing GPU translation faults. KGSL does not fault on null reads, but MSM DRM does.
