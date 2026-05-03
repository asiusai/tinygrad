#include <CL/cl.h>
#include <string.h>

typedef struct {
    cl_command_queue queue;
    cl_kernel kernel;
    cl_uint ndim;
    size_t global_size[3];
    size_t local_size[3];
    int has_local;
} KernelCmd;

// Batch enqueue multiple kernels with no arg changes
cl_int batch_enqueue(KernelCmd* cmds, int count) {
    for (int i = 0; i < count; i++) {
        cl_int err = clEnqueueNDRangeKernel(
            cmds[i].queue, cmds[i].kernel, cmds[i].ndim,
            NULL, cmds[i].global_size,
            cmds[i].has_local ? cmds[i].local_size : NULL,
            0, NULL, NULL);
        if (err != 0) return err;
    }
    return 0;
}

// Set a kernel arg and enqueue
cl_int set_arg_and_enqueue(
    cl_command_queue queue, cl_kernel kernel,
    cl_uint ndim, size_t* global_size, size_t* local_size,
    int num_args, int* arg_indices, cl_mem* arg_values) {
    for (int i = 0; i < num_args; i++) {
        cl_int err = clSetKernelArg(kernel, arg_indices[i], sizeof(cl_mem), &arg_values[i]);
        if (err != 0) return err;
    }
    return clEnqueueNDRangeKernel(queue, kernel, ndim, NULL, global_size, local_size, 0, NULL, NULL);
}
