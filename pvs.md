---
titlepage: true
title: "Parallele und Verteilte Systeme: WS20/21 -- Übung 5 -- Gruppe B"
author: Lisa Piekarski, Klara Wichmann, Jakob Béla Ruckel
date: "2021-02-04"
listings-no-page-break: true
...

# Initial Kernel

```cpp
__kernel void mult(__global float *A,
                   __global float *B,
                   __global float *C) {
   int i, j, k;
   i = get_global_id(0);
   for (j = 0; j < DIM; ++j) {
       for (k = 0; k < DIM; ++k) {
           C[i*DIM+j] += A[i*DIM+k] * B[k*DIM+j];
       }
   }
}
```

TODO:  Discuss.


# Task 1: Reduction of Field Access

```cpp
__kernel void mult(__global float *A,
                   __global float *B,
                   __global float *C) {
   int i, j, k;
   i = get_global_id(0);
   for (j = 0; j < DIM; ++j) {
       float tmp = .0f;
       for (k = 0; k < DIM; ++k) {
            tmp += A[i*DIM+k] * B[k*DIM+j];
       }
       C[i*DIM+j] = tmp;
   }
}
```

TODO:  Discuss.


# Task 2: Loop Swapping

```cpp
__kernel void mult(__global float *A,
                   __global float *B,
                   __global float *C) {
   int i, j, k;
   j = get_global_id(0);
   for (i = 0; i < DIM; ++i) {
       float tmp = .0f;
       for (k = 0; k < DIM; ++k) {
            tmp += A[i*DIM+k] * B[k*DIM+j];
       }
       C[i*DIM+j] = tmp;
   }
}
```

TODO:  Discuss.


# Task 3: Memory Optimization

```cpp
__kernel void mult(__global float *A,
                   __global float *B,
                   __global float *C) {
   int i, j, k;
   i = get_global_id(0);
   float A1[DIM];
   for (k = 0; k < DIM; ++k) {
       A1[k] = A[i*DIM+k];
   }
   for (j = 0; j < DIM; ++j) {
       float tmp = .0f;
       for (k = 0; k < DIM; ++k) {
           tmp += A1[k] * B[k*DIM+j];
       }
       C[i*DIM+j] = tmp;
   }
}
```

TODO:  Discuss.


# Task 4: Distributed Storage Optimization in Workgroups

TODO:  Implement, add kernel here.

TODO:  Discuss.
