// compile in Linux with gcc:
// g++ hello_world.cpp -lOpenCL

#include "matmult.hpp"
#include <assert.h>
#include <omp.h>  // for wall-clock timing
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "CL/cl.h"

#define DATA_SIZE MAT_SIZE* MAT_SIZE
#define MEM_SIZE DATA_SIZE * sizeof(float)

#ifndef KERNEL
#define KERNEL 1
#endif

#if KERNEL == 0

const char* KernelName = "Initial Version";
const char* KernelSource = "#define DIM " MAT_SIZE_STR
                           "\n"
                           "__kernel void mult(__global float *A,"
                           "                   __global float *B,"
                           "                   __global float *C) {"
                           "   int i, j, k;"
                           "   i = get_global_id(0);"
                           "   for (j = 0; j < DIM; ++j) {"
                           "       for (k = 0; k < DIM; ++k) {"
                           "           C[i*DIM+j] += A[i*DIM+k] * B[k*DIM+j];"
                           "       }"
                           "   }"
                           "}";

#elif KERNEL == 1
//------------------------------------------------------------------------
// task 1: Reduction of field access

const char* KernelName = "Reduction of field access";
const char* KernelSource = "#define DIM " MAT_SIZE_STR
                           "\n"
                           "__kernel void mult(__global float *A,"
                           "                   __global float *B,"
                           "                   __global float *C) {"
                           "   int i, j, k;"
                           "   i = get_global_id(0);"
                           "   for (j = 0; j < DIM; ++j) {"
                           "       float tmp = .0f;"
                           "       for (k = 0; k < DIM; ++k) {"
                           "            tmp += A[i*DIM+k] * B[k*DIM+j];"
                           "       }"
                           "       C[i*DIM+j] = tmp;"
                           "   }"
                           "}";

#elif KERNEL == 2
//------------------------------------------------------------------------
// task 2: Loop swapping

const char* KernelName = "Loop swapping";
const char* KernelSource = "#define DIM " MAT_SIZE_STR
                           "\n"
                           "__kernel void mult(__global float *A,"
                           "                   __global float *B,"
                           "                   __global float *C) {"
                           "   int i, j, k;"
                           "   j = get_global_id(0);"
                           "   for (i = 0; i < DIM; ++i) {"
                           "       float tmp = .0f;"
                           "       for (k = 0; k < DIM; ++k) {"
                           "            tmp += A[i*DIM+k] * B[k*DIM+j];"
                           "       }"
                           "       C[i*DIM+j] = tmp;"
                           "   }"
                           "}";

#elif KERNEL == 3
//------------------------------------------------------------------------
// task 3: Memory optimization
// TODO: Shouldn't we also do the loop swapping of kernel 2 here?

const char* KernelName = "Memory optimization";
const char* KernelSource = "#define DIM " MAT_SIZE_STR
                           "\n"
                           "__kernel void mult(__global float *A,"
                           "                   __global float *B,"
                           "                   __global float *C) {"
                           "   int i, j, k;"
                           "   i = get_global_id(0);"
                           "   float A1[DIM];"
                           "   for (k = 0; k < DIM; ++k) {"
                           "       A1[k] = A[i*DIM+k];"
                           "   }"
                           "   for (j = 0; j < DIM; ++j) {"
                           "       float tmp = .0f;"
                           "       for (k = 0; k < DIM; ++k) {"
                           "           tmp += A1[k] * B[k*DIM+j];"
                           "       }"
                           "       C[i*DIM+j] = tmp;"
                           "   }"
                           "}";

#elif KERNEL == 4
//------------------------------------------------------------------------
// task 4: Distributed storage optimization in workgroups
// TODO (COPIED from task 3)

const char* KernelName = "Distributed storage optimization in workgroups";
const char* KernelSource = "#define DIM " MAT_SIZE_STR
                           "\n"
                           "__kernel void mult(__global float *A,"
                           "                   __global float *B,"
                           "                   __global float *C) {"
                           "   int i, j, k;"
                           "   int id = get_local_id(0);"
                           "   int size = get_local_size(0);"
                           "   i = get_global_id(0);"
                           "   float A1[DIM];"
                           "   for (k = 0; k < DIM; ++k) {"
                           "       A1[k] = A[i*DIM+k];"
                           "   }"
                           "   for (j = 0; j < DIM; ++j) {"
                           "       for (k = id; k < DIM; k += size) {"
                           "           B1[k] = B[k*DIM+j];}"
                           "       float tmp = .0f;"
                           "       for (k = 0; k < DIM; ++k) {"
                           "           tmp += A1[k] * B1[k];"
                           "       }"
                           "       C[i*DIM+j] = tmp;"
                           "   }"
                           "}";
#endif

/** **/
int main(void) {
    printf("Running with kernel #%d (%s).\n", KERNEL, KernelName);

    cl_int err;
    cl_platform_id* platforms = NULL;
    char platform_name[1024];
    cl_device_id device_id = NULL;
    cl_uint num_of_platforms = 0, num_of_devices = 0;
    cl_context context;
    cl_kernel kernel;
    cl_command_queue command_queue;
    cl_program program;
    cl_mem buf_A, buf_B, output;

    float** A = alloc_mat(MAT_SIZE, MAT_SIZE);
    init_mat(A, MAT_SIZE, MAT_SIZE);

    float** B = alloc_mat(MAT_SIZE, MAT_SIZE);
    init_mat(B, MAT_SIZE, MAT_SIZE);

    float** C = alloc_mat(MAT_SIZE, MAT_SIZE);
    float** C_serial = alloc_mat(MAT_SIZE, MAT_SIZE);

    size_t global[1] = {MAT_SIZE};

    double t_start_par = omp_get_wtime();
    /* 1) */
    err = clGetPlatformIDs(0, NULL, &num_of_platforms);
    if (err != CL_SUCCESS) {
        printf("No platforms found. Error: %d\n", err);
        return 0;
    }

    platforms = (cl_platform_id*)malloc(num_of_platforms);
    err = clGetPlatformIDs(num_of_platforms, platforms, NULL);
    if (err != CL_SUCCESS) {
        printf("No platforms found. Error: %d\n", err);
        return 0;
    } else {
        int nvidia_platform = 0;

        for (unsigned int i = 0; i < num_of_platforms; i++) {
            clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME,
                              sizeof(platform_name), platform_name, NULL);
            if (err != CL_SUCCESS) {
                printf("Could not get information about platform. Error: %d\n",
                       err);
                return 0;
            }

            if (strstr(platform_name, "NVIDIA") != NULL) {
                nvidia_platform = i;
                break;
            }
        }

        err = clGetDeviceIDs(platforms[nvidia_platform], CL_DEVICE_TYPE_GPU, 1,
                             &device_id, &num_of_devices);
        if (err != CL_SUCCESS) {
            printf("Could not get device in platform. Error: %d\n", err);
            return 0;
        }
    }

    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("Unable to create context. Error: %d\n", err);
        return 0;
    }

    command_queue = clCreateCommandQueue(context, device_id, 0, &err);
    if (err != CL_SUCCESS) {
        printf("Unable to create command queue. Error: %d\n", err);
        return 0;
    }

    program = clCreateProgramWithSource(context, 1, (const char**)&KernelSource,
                                        NULL, &err);
    if (err != CL_SUCCESS) {
        printf("Unable to create program. Error: %d\n", err);
        return 0;
    }

    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error building program. Error: %d\n", err);
        return 0;
    }

    kernel = clCreateKernel(program, "mult", &err);
    if (err != CL_SUCCESS) {
        printf("Error setting kernel. Error: %d\n", err);
        return 0;
    }

    /* 2) */

    buf_A = clCreateBuffer(context, CL_MEM_READ_ONLY, MEM_SIZE, NULL, &err);
    buf_B = clCreateBuffer(context, CL_MEM_READ_ONLY, MEM_SIZE, NULL, &err);
    output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, MEM_SIZE, NULL, &err);

    clEnqueueWriteBuffer(command_queue, buf_A, CL_TRUE, 0, MEM_SIZE, A[0], 0,
                         NULL, NULL);
    clEnqueueWriteBuffer(command_queue, buf_B, CL_TRUE, 0, MEM_SIZE, B[0], 0,
                         NULL, NULL);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_A);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_B);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &output);

    /* 3)  */

    clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, global, NULL, 0,
                           NULL, NULL);

    clFinish(command_queue);

    clEnqueueReadBuffer(command_queue, output, CL_TRUE, 0, MEM_SIZE, C[0], 0,
                        NULL, NULL);

    /* 4) */
    clReleaseMemObject(buf_A);
    clReleaseMemObject(buf_B);
    clReleaseMemObject(output);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);

    double t_end_par = omp_get_wtime();

    matmult_serial(A, B, C_serial);
    double t_end_serial = omp_get_wtime();

    assert(mat_equal(C, C_serial, MAT_SIZE, MAT_SIZE));

    printf("Results are correct.\n");
    double t_serial = t_end_serial - t_end_par;
    double t_parallel = t_end_par - t_start_par;
    printf("Serial took %.5f seconds.\n", t_serial);
    printf("Parallel took %.5f seconds.\n", t_parallel);
    printf("That's %.2f times faster!\n", t_serial / t_parallel);

    return 0;
}
