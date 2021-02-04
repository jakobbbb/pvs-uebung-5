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

```
[user@computer pvs-uebung-5]$ make test KERNEL=0
Running with kernel #0 (Initial Version).
Matrices differ at [0][0], where 61788.00 != 20596.00
matmult: matmult.cpp:267: int main(): Assertion `mat_equal(C, C_serial, MAT_SIZE, MAT_SIZE)' failed.
make: *** [Makefile:24: test] Aborted (core dumped)
```

Turns out that this kernel does not produce correct results.  This may
be caused by a race condition where write accesses to `C` collide
because they are happening too quickly.


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

```
[user@computer pvs-uebung-5]$ make test KERNEL=1
Running with kernel #1 (Reduction of field access).
Results are correct.
Serial took 8.95086 seconds.
Parallel took 0.72261 seconds.
That's 12.39 times faster!
```

By using a temporary variable to accumulate the result for `C[i*DIM+j]`,
we're avoiding the race condition of kernel `0`.

Furthermore, writing to a local variable like `tmp` is much faster than
writing to a variable in global memory, like `C`.

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

```
[user@computer pvs-uebung-5]$ make test KERNEL=2
Running with kernel #2 (Loop swapping).
Results are correct.
Serial took 8.65357 seconds.
Parallel took 0.18878 seconds.
That's 45.84 times faster!
```

By swapping the loops we're speeding up access to `A`: The matrix `A`
(which is a large contiguous array in memory) is accessed at index
`i*DIM+k`.  Swapping the loops changes the loop variables to `i` in
the outer loop and `k` in the inner loop.  This means that with each
iteration of the inner loop, were always going to the very next element
of `A`.  That is, `A` is read sequentially, which allows the computer to
make more efficient use of its cache.

The previous iteration of the kernel jumps around more while reading
`A`, causing more cache misses which costs time.

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

```
[user@computer pvs-uebung-5]$ make test KERNEL=3
Running with kernel #3 (Memory optimization).
Results are correct.
Serial took 9.02747 seconds.
Parallel took 0.16888 seconds.
That's 53.46 times faster!
```

TODO:  Discuss.


# Task 4: Distributed Storage Optimization in Workgroups

TODO:  Implement, add kernel here.

```
[user@computer pvs-uebung-5]$ make test KERNEL=4
Running with kernel #4 (Distributed storage optimization in workgroups).
Results are correct.
Serial took 8.81930 seconds.
Parallel took 0.41513 seconds.
That's 21.24 times faster!
```

TODO:  Discuss.
