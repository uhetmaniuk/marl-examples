///
/// clang++ -O3 -I/Users/ulrich/TPL/marl/include/ marl_cg.cpp -std=c++17
/// -L/Users/ulrich/TPL/marl/build/ -lmarl
///

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <numeric>
#include <thread>
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

// Pads a double to a cache line so per-task accumulators don't share lines.
struct alignas(64) PaddedDouble {
  double v;
};

// Aggregated per-kernel timing, printed once at end of solve.
struct KernelStats {
  double dot_seconds = 0.0;
  double axpy_seconds = 0.0;
  double spmv_seconds = 0.0;
  int dot_calls = 0;
  int axpy_calls = 0;
  int spmv_calls = 0;
};

// Vector operations parallelized with Marl

// Computes the dot product of two vectors in parallel.
double dot(
    const std::vector<double>& a,
    const std::vector<double>& b,
    KernelStats* stats = nullptr) {
  auto start = std::chrono::high_resolution_clock::now();
  auto* sch = marl::Scheduler::get();
  int num_tasks = sch->config().workerThread.count;
  num_tasks = std::min<int>(num_tasks, static_cast<int>(a.size()));
  if (num_tasks <= 0)
    num_tasks = 1;
  std::vector<PaddedDouble> partial_sums(num_tasks);
  for (auto& p : partial_sums)
    p.v = 0.0;
  marl::WaitGroup wg(num_tasks);

  int chunk_size = static_cast<int>(a.size()) / num_tasks;
  for (int i = 0; i < num_tasks; ++i) {
    marl::schedule([=, &partial_sums, &a, &b] {
      defer(wg.done());
      int start = i * chunk_size;
      int end = (i == num_tasks - 1) ? static_cast<int>(a.size())
                                     : start + chunk_size;
      double sum = 0.0;
      for (int j = start; j < end; ++j) {
        sum += a[j] * b[j];
      }
      partial_sums[i].v = sum;
    });
  }
  wg.wait();
  auto stop = std::chrono::high_resolution_clock::now();
  if (stats) {
    std::chrono::duration<double> elapsed = stop - start;
    stats->dot_seconds += elapsed.count();
    stats->dot_calls += 1;
  }
  double total = 0.0;
  for (const auto& p : partial_sums)
    total += p.v;
  return total;
}

// Computes v_out = v_in1 + scalar * v_in2 in parallel.
// Note: aliasing v_out with v_in1 or v_in2 is safe because each task owns a
// disjoint index range and the update is purely element-wise.
void axpy(
    std::vector<double>& v_out,
    const std::vector<double>& v_in1,
    const std::vector<double>& v_in2,
    double scalar,
    KernelStats* stats = nullptr) {
  auto start = std::chrono::high_resolution_clock::now();
  auto* sch = marl::Scheduler::get();
  int num_tasks = sch->config().workerThread.count;
  num_tasks = std::min<int>(num_tasks, static_cast<int>(v_out.size()));
  if (num_tasks <= 0)
    num_tasks = 1;
  marl::WaitGroup wg(num_tasks);

  int chunk_size = static_cast<int>(v_out.size()) / num_tasks;
  for (int i = 0; i < num_tasks; ++i) {
    marl::schedule([=, &v_out, &v_in1, &v_in2] {
      defer(wg.done());
      int start = i * chunk_size;
      int end = (i == num_tasks - 1) ? static_cast<int>(v_out.size())
                                     : start + chunk_size;
      for (int j = start; j < end; ++j) {
        v_out[j] = v_in1[j] + scalar * v_in2[j];
      }
    });
  }
  wg.wait();
  auto stop = std::chrono::high_resolution_clock::now();
  if (stats) {
    std::chrono::duration<double> elapsed = stop - start;
    stats->axpy_seconds += elapsed.count();
    stats->axpy_calls += 1;
  }
}

// Computes matrix-vector product y = Ax in parallel.
void spmv(
    std::vector<double>& y,
    const SparseMatrix& A,
    const std::vector<double>& x,
    KernelStats* stats = nullptr) {
  auto start = std::chrono::high_resolution_clock::now();
  auto* sch = marl::Scheduler::get();
  int num_tasks = sch->config().workerThread.count;
  num_tasks = std::min<int>(num_tasks, A.num_rows);
  if (num_tasks <= 0)
    num_tasks = 1;
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
  if (stats) {
    std::chrono::duration<double> elapsed = stop - start;
    stats->spmv_seconds += elapsed.count();
    stats->spmv_calls += 1;
  }
}

// Solves Ax = b using the conjugate gradient method with Marl for parallelism.
std::vector<double> conjugate_gradient_marl(
    const SparseMatrix& A,
    const std::vector<double>& b,
    int max_iters,
    double tol) {
  std::vector<double> x(A.num_cols, 0.0);
  std::vector<double> r = b;
  std::vector<double> p = r;
  std::vector<double> ap(A.num_rows);

  KernelStats stats;
  const double tol_sq = tol * tol;

  double rs_old = dot(r, r, &stats);

  int iters_done = 0;
  double final_residual = std::sqrt(rs_old);
  for (int i = 0; i < max_iters; ++i) {
    spmv(ap, A, p, &stats);
    double alpha = rs_old / dot(p, ap, &stats);

    axpy(x, x, p, alpha, &stats);
    axpy(r, r, ap, -alpha, &stats);

    double rs_new = dot(r, r, &stats);
    iters_done = i + 1;
    final_residual = std::sqrt(rs_new);
    if (rs_new < tol_sq) {
      break;
    }

    axpy(p, r, p, rs_new / rs_old, &stats);
    rs_old = rs_new;
  }

  printf(" iters=%d residual=%e\n", iters_done, final_residual);
  printf(
      " dot:  calls=%d total=%e avg=%e\n",
      stats.dot_calls,
      stats.dot_seconds,
      stats.dot_calls ? stats.dot_seconds / stats.dot_calls : 0.0);
  printf(
      " axpy: calls=%d total=%e avg=%e\n",
      stats.axpy_calls,
      stats.axpy_seconds,
      stats.axpy_calls ? stats.axpy_seconds / stats.axpy_calls : 0.0);
  printf(
      " spmv: calls=%d total=%e avg=%e\n",
      stats.spmv_calls,
      stats.spmv_seconds,
      stats.spmv_calls ? stats.spmv_seconds / stats.spmv_calls : 0.0);
  return x;
}

int main(int argc, char** argv) {
  // Pick worker thread count: CLI arg > MARL_NUM_THREADS env > all cores.
  marl::Scheduler::Config cfg = marl::Scheduler::Config::allCores();
  int requested = 0;
  if (argc > 1) {
    requested = std::atoi(argv[1]);
  } else if (const char* env = std::getenv("MARL_NUM_THREADS")) {
    requested = std::atoi(env);
  }
  if (requested > 0) {
    cfg.setWorkerThreadCount(requested);
  }

  marl::Scheduler scheduler(cfg);
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
