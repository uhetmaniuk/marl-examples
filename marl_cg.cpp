///
/// clang++ -O3 -I/Users/ulrich/TPL/marl/include/ marl_cg.cpp -std=c++17 -L/Users/ulrich/TPL/marl/build/ -lmarl
///

#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <numeric>
#include <vector>

// Google Marl headers
#include "marl/defer.h"
#include "marl/event.h"
#include "marl/scheduler.h"
#include "marl/waitgroup.h"

// Represents a sparse matrix in Compressed Sparse Row (CSR) format.
struct SparseMatrix {
    std::vector<double> values;
    std::vector<int> col_indices;
    std::vector<int> row_ptr;
    int num_rows;
    int num_cols;
};

// Vector operations parallelized with Marl

// Computes the dot product of two vectors in parallel.
double dot(const std::vector<double>& a, const std::vector<double>& b) {
auto start = std::chrono::high_resolution_clock::now();
    auto *sch = marl::Scheduler::get();
    const int num_tasks = sch->config().workerThread.count;
    std::vector<double> partial_sums(num_tasks, 0.0);
    marl::WaitGroup wg(num_tasks);

    int chunk_size = a.size() / num_tasks;
    for (int i = 0; i < num_tasks; ++i) {
        marl::schedule([=, &partial_sums, &a, &b] {
            defer(wg.done());
            int start = i * chunk_size;
            int end = (i == num_tasks - 1) ? a.size() : start + chunk_size;
            for (int j = start; j < end; ++j) {
                partial_sums[i] += a[j] * b[j];
            }
        });
    }
    wg.wait();
  auto stop = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = stop - start;
  printf(" %e ", elapsed.count());
    return std::accumulate(partial_sums.begin(), partial_sums.end(), 0.0);
}

// Computes v_out = v_in1 + scalar * v_in2 in parallel.
void axpy(std::vector<double>& v_out, const std::vector<double>& v_in1, const std::vector<double>& v_in2, double scalar) {
auto start = std::chrono::high_resolution_clock::now();
    auto *sch = marl::Scheduler::get();
    const int num_tasks = sch->config().workerThread.count;
    marl::WaitGroup wg(num_tasks);

    int chunk_size = v_out.size() / num_tasks;
    for (int i = 0; i < num_tasks; ++i) {
        marl::schedule([=, &v_out, &v_in1, &v_in2] {
            defer(wg.done());
            int start = i * chunk_size;
            int end = (i == num_tasks - 1) ? v_out.size() : start + chunk_size;
            for (int j = start; j < end; ++j) {
                v_out[j] = v_in1[j] + scalar * v_in2[j];
            }
        });
    }
  auto stop = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = stop - start;
  printf(" %e ", elapsed.count());

  wg.wait();
}

// Computes matrix-vector product y = Ax in parallel.
void spmv(std::vector<double>& y, const SparseMatrix& A, const std::vector<double>& x) {
auto start = std::chrono::high_resolution_clock::now();
    auto *sch = marl::Scheduler::get();
    const int num_tasks = sch->config().workerThread.count;
    marl::WaitGroup wg(num_tasks);

    int chunk_size = A.num_rows / num_tasks;
    for (int i = 0; i < num_tasks; ++i) {
        marl::schedule([=, &x, &A, &y] {
            defer(wg.done());
            int start_row = i * chunk_size;
            int end_row = (i == num_tasks - 1) ? A.num_rows : start_row + chunk_size;
            for (int row = start_row; row < end_row; ++row) {
                double sum = 0.0;
                for (int j = A.row_ptr[row]; j < A.row_ptr[row + 1]; ++j) {
                    sum += A.values[j] * x[A.col_indices[j]];
                }
                y[row] = sum;
            }
        });
    }
    wg.wait();
    auto stop = std::chrono::high_resolution_clock::now();
std::chrono::duration<double> elapsed = stop - start;
printf(" %d %d %e ", num_tasks, chunk_size, elapsed.count());

}

// Solves Ax = b using the conjugate gradient method with Marl for parallelism.
std::vector<double> conjugate_gradient_marl(const SparseMatrix& A, const std::vector<double>& b, int max_iters, double tol) {
    std::vector<double> x(A.num_cols, 0.0);
    std::vector<double> r = b;
    std::vector<double> p = r;
    std::vector<double> ap(A.num_rows);

    double rs_old = dot(r, r);

    for (int i = 0; i < max_iters; ++i) {
        spmv(ap, A, p);
        double alpha = rs_old / dot(p, ap);

        axpy(x, x, p, alpha);
        axpy(r, r, ap, -alpha);

        double rs_new = dot(r, r);
        if (std::sqrt(rs_new) < tol) {
            printf(" %d %e \n", i, std::sqrt(rs_new));
            break;
        }

        axpy(p, r, p, rs_new / rs_old);
        rs_old = rs_new;
printf("\n");
    }
    return x;
}

int main() {
    // Set up a Marl scheduler.
    marl::Scheduler scheduler(marl::Scheduler::Config::allCores());
    scheduler.bind();
    defer(scheduler.unbind());

    const auto numThreads = scheduler.config().workerThread.count;
    std::cout << " # Threads : " << numThreads << std::endl;

{
    // Create a sample problem Ax = b
    // A = [[4, 1], [1, 3]]
    // b = [1, 2]
    // Solution is approximately [0.09, 0.63]
    SparseMatrix A;
    A.num_rows = 2;
    A.num_cols = 2;
    A.values = {4, 1, 1, 3};
    A.col_indices = {0, 1, 0, 1};
    A.row_ptr = {0, 2, 4};

    std::vector<double> b = {1, 2};

    // Solve the system
    std::vector<double> x = conjugate_gradient_marl(A, b, 100, 1e-6);

    // Print the solution
    std::cout << "Solution x: [ ";
    for (auto val : x) {
        std::cout << val << " ";
    }
    std::cout << "]" << std::endl;
}

{
int nx = 314;
int ny = nx;
int n = ny * nx;
SparseMatrix A;
A.num_rows = n;
A.num_cols = n;
A.values.resize(9 * n);
A.col_indices.resize(9 * n);
A.row_ptr.resize(n + 1);
A.row_ptr[0] = 0;
int iNode = 0;
for (int iy = 0; iy < ny; ++iy) {
for (int ix = 0; ix < nx; ++ix) {
int localCount = 0;
if (iy > 0) {
    if (ix > 0) {
        A.values[A.row_ptr[iNode] + localCount] = -1;
        A.col_indices[A.row_ptr[iNode] + localCount] = iNode - nx - 1;
        localCount++;
    }
    A.values[A.row_ptr[iNode] + localCount] = -1;
    A.col_indices[A.row_ptr[iNode] + localCount] = iNode - nx;
    localCount++;
    if (ix + 1 < nx) {
        A.values[A.row_ptr[iNode] + localCount] = -1;
        A.col_indices[A.row_ptr[iNode] + localCount] = iNode - nx + 1;
        localCount++;
    }
}
    if (ix > 0) {
        A.values[A.row_ptr[iNode] + localCount] = -1;
        A.col_indices[A.row_ptr[iNode] + localCount] = iNode - 1;
        localCount++;
    }
    A.values[A.row_ptr[iNode] + localCount] = 8;
    A.col_indices[A.row_ptr[iNode] + localCount] = iNode;
    localCount++;
    if (ix + 1 < nx) {
        A.values[A.row_ptr[iNode] + localCount] = -1;
        A.col_indices[A.row_ptr[iNode] + localCount] = iNode + 1;
        localCount++;
    }
    if (iy + 1 < ny) {
        if (ix > 0) {
            A.values[A.row_ptr[iNode] + localCount] = -1;
            A.col_indices[A.row_ptr[iNode] + localCount] = iNode + nx - 1;
            localCount++;
        }
        A.values[A.row_ptr[iNode] + localCount] = -1;
        A.col_indices[A.row_ptr[iNode] + localCount] = iNode + nx;
        localCount++;
        if (ix + 1 < nx) {
            A.values[A.row_ptr[iNode] + localCount] = -1;
            A.col_indices[A.row_ptr[iNode] + localCount] = iNode + nx + 1;
            localCount++;
        }
    }
    A.row_ptr[iNode + 1] = A.row_ptr[iNode] + localCount;
    iNode += 1;
        }
    }
        std::vector<double> b(n, -1.0);
        // Solve the system
        std::vector<double> x = conjugate_gradient_marl(A, b, 1000, 1e-6);
}

    return 0;
}

