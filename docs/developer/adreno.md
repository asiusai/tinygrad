# Adreno GPU

There are two ways to run tinygrad on Qualcomm Adreno GPUs:

| | `DEV=QCOM` | `DEV=MSM` |
|---|---|---|
| Kernel driver | KGSL (`/dev/kgsl-3d0`) | MSM DRM (`/dev/dri/renderD*`) |
| Found on | Android, Qualcomm vendor kernels | Mainline/upstream Linux |
| Shader compiler | OpenCL (default) or IR3 (`QCOM_IR3=1`) | IR3 only |

Use `DEV=QCOM` if you have `/dev/kgsl-3d0` (Android devices, comma body). Use `DEV=MSM` if you have `/dev/dri/renderD*` with an Adreno GPU (mainline Linux kernel).

Both backends share the same IR3 shader compiler (from Mesa) and the same PM4 command stream for a6xx register programming. They differ only in how they talk to the kernel.

## How to run?

```bash
DEV=MSM MESA_PATH=/path/to/libtinymesa.so python3 your_script.py
```

Requirements:

* Linux kernel with `CONFIG_DRM_MSM=y` (GPU support, not just display)
* Adreno a6xx GPU (tested: Adreno 630 / SDM845)
* `libtinymesa.so` built from Mesa (provides the IR3 shader compiler, used by both `QCOM_IR3=1` and `DEV=MSM`)

## MSM Driver Details

### Memory Management

GEM buffers are allocated via `DRM_MSM_GEM_NEW`, CPU-mapped via `DRM_MSM_GEM_INFO`, and assigned a GPU virtual address (IOVA) by the kernel. GEM handles are never closed during runtime. The GPU's IOMMU TLB can retain stale entries after a handle is closed, causing translation faults on subsequent submits. All handles are tracked and closed only at process exit.

### Command Submission

MSM DRM is ioctl-based: each dispatch builds a PM4 command stream and submits via `DRM_MSM_GEM_SUBMIT`. There are no hardware command queues or doorbells.

### Graph

`MSMGraph` batches all JIT-captured kernels into a single `DRM_MSM_GEM_SUBMIT`. PM4 and args buffers are pre-built once; on each replay only the input buffer IOVAs are patched.
