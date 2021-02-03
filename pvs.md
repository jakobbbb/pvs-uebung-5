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
