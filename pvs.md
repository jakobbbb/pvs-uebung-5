---
titlepage: true
title: "Parallele und Verteilte Systeme: WS20/21 -- Übung 5 -- Gruppe B"
author: Lisa Piekarski, Klara Wichmann, Jakob Béla Ruckel
date: "2021-02-04"
listings-no-page-break: true
...

# Compiling

Run `make test KERNEL=n` to compile and test the kernel with id `n`,
where `id` ranges from `0` to `4`:

| `n` | Kernel | Speedup |
|-----|--------|---------|
| `0` | Initial | - |
| `1` | Reduction of Field Access | 12.39 |
| `2` | Swapped Loops | 45.84 |
| `3` | Memory Optimization | 53.46 |
| `4` | More Memory Optimization | TODO |

# Initial Kernel (0)

```
[user@computer pvs-uebung-5]$ make test KERNEL=0
Running with kernel #0 (Initial Version).
Matrices differ at [0][0], where 61788.00 != 20596.00
matmult: matmult.cpp:267: int main(): Assertion `mat_equal(C, C_serial, MAT_SIZE, MAT_SIZE)' failed.
make: *** [Makefile:24: test] Aborted (core dumped)
```

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

```
[user@computer pvs-uebung-5]$ make test KERNEL=1
Running with kernel #1 (Reduction of field access).
Results are correct.
Serial took 8.95086 seconds.
Parallel took 0.72261 seconds.
That's 12.39 times faster!
```

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

```
[user@computer pvs-uebung-5]$ make test KERNEL=2
Running with kernel #2 (Loop swapping).
Results are correct.
Serial took 8.65357 seconds.
Parallel took 0.18878 seconds.
That's 45.84 times faster!
```


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

```
[user@computer pvs-uebung-5]$ make test KERNEL=3
Running with kernel #3 (Memory optimization).
Results are correct.
Serial took 9.02747 seconds.
Parallel took 0.16888 seconds.
That's 53.46 times faster!
```

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
