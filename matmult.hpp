#ifndef MATMULT_HPP
#define MATMULT_HPP

#include <stdio.h>
#include <stdlib.h>

#define MAT_SIZE 1000
#define MAT_SIZE_STR "1000"
#define EPSILON 0.0001f

// ---------------------------------------------------------------------------
// allocate space for empty matrix A[row][col]
// access to matrix elements possible with:
// - A[row][col]
// - A[0][row*col]

float** alloc_mat(int row, int col) {
    float **A1, *A2;

    A1 = (float**)calloc(row, sizeof(float*));      // pointer on rows
    A2 = (float*)calloc(row * col, sizeof(float));  // all matrix elements
    for (int i = 0; i < row; i++)
        A1[i] = A2 + i * col;

    return A1;
}

// ---------------------------------------------------------------------------
// random initialisation of matrix with values [0..9]

void init_mat(float** A, int row, int col) {
    for (int i = 0; i < row * col; i++)
        A[0][i] = (float)(rand() % 10);
}

// ---------------------------------------------------------------------------
// DEBUG FUNCTION: printout of all matrix elements

void print_mat(float** A, int row, int col, char const* tag) {
    int i, j;

    printf("Matrix %s:\n", tag);
    for (i = 0; i < row; i++) {
        for (j = 0; j < col; j++)
            printf("%6.1f   ", A[i][j]);
        printf("\n");
    }
}

// ---------------------------------------------------------------------------
// free dynamically allocated memory, which was used to store a 2D matrix
void free_mat(float** A, int num_rows) {
    free(A[0]);  // free contiguous block of float elements (row*col floats)
    free(A);  // free memory for pointers pointing to the beginning of each row
}

// ---------------------------------------------------------------------------
// Check two matrices for equality
bool mat_equal(float** mat1, float** mat2, int m, int n) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (abs(mat1[i][j] - mat2[i][j]) > EPSILON) {
                return false;
            }
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Number of rows of A that will be transferred to the worker.
// Will not always be even, e.g. 333, 333 and 334 for 1000x1000
// matrices and 3 workers.
int calc_num_rows_part(int worker_id, int num_workers) {
    int num_rows_part = MAT_SIZE / num_workers;
    if (worker_id == num_workers) {
        num_rows_part += MAT_SIZE % num_workers;
    }
    return num_rows_part;
}

// ---------------------------------------------------------------------------
// Serial matrix multiplication
void matmult_serial(float** A, float** B, float** C) {
    for (int i = 0; i < MAT_SIZE; i++) {
        for (int j = 0; j < MAT_SIZE; j++) {
            for (int k = 0; k < MAT_SIZE; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

#endif  // MATMULT_HPP
