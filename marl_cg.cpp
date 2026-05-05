///
/// clang++ -O3 -I/Users/ulrich/TPL/marl/include/ marl_cg.cpp -std=c++17
/// -L/Users/ulrich/TPL/marl/build/ -lmarl
///

#include <algorithm>
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

// Multiplier on the per-solve task count. num_tasks defaults to the marl
// worker thread count; setting g_tasks_per_worker > 1 over-decomposes so
// each OS thread runs multiple stripe-fibers per iteration. Set from main()
// before each timing sweep.
int g_tasks_per_worker = 1;

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
  int num_tasks = sch->config().workerThread.count * g_tasks_per_worker;
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
  int num_tasks = sch->config().workerThread.count * g_tasks_per_worker;
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
  int num_tasks = sch->config().workerThread.count * g_tasks_per_worker;
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
    if (rs_new < tol_sq) {
      final_residual = std::sqrt(rs_new);
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

// Fused per-iteration variant: schedules one stripe-task per worker per
// iteration (instead of one per kernel) and uses marl::Event/WaitGroup to
// synchronize phases. Same math as conjugate_gradient_marl, fewer fork-joins.
//
// Per iteration each stripe-task runs:
//   A: ap[s..e] = A * p             (writes only own slice of ap)
//   B: partial_pAp[i] = sum p[j]*ap[j] over [s,e)  (reads only own slice of ap)
//      <wg_pAp + ev_alpha — main reduces, computes alpha, broadcasts>
//   C: x[s..e] += alpha*p; r[s..e] -= alpha*ap; partial_rs[i] = sum r[j]^2
//      <wg_rs + ev_beta — main reduces, decides convergence, computes beta>
//   D: p[s..e] = r + beta*p   (skipped if converged)
// No A->B barrier is needed: B reads only ap[s..e), which A just wrote on the
// same stripe. Cross-stripe reads of p in the next iteration's spmv are
// covered by wg_iter on the main thread before re-scheduling.
std::vector<double> conjugate_gradient_marl_fused(
    const SparseMatrix& A,
    const std::vector<double>& b,
    int max_iters,
    double tol) {
  const int n = A.num_rows;
  std::vector<double> x(A.num_cols, 0.0);
  std::vector<double> r = b;
  std::vector<double> p = r;

  auto* sch = marl::Scheduler::get();
  int num_tasks = sch->config().workerThread.count * g_tasks_per_worker;
  num_tasks = std::min<int>(num_tasks, n);
  if (num_tasks <= 0)
    num_tasks = 1;
  const int chunk = (n + num_tasks - 1) / num_tasks;

  // Per-task scratch for the spmv result A*p restricted to the task's stripe.
  // ap is only read by the same stripe in Phase C, so it does not need to be
  // a globally-visible vector — keeping it task-local improves cache locality
  // (the stripe stays in the worker's L1/L2 between Phases A+B and C) and
  // removes the size-n global allocation.
  std::vector<std::vector<double>> ap_local(num_tasks);
  for (int i = 0; i < num_tasks; ++i) {
    const int s = i * chunk;
    const int e = std::min(s + chunk, n);
    ap_local[i].resize(e - s);
  }

  const double tol_sq = tol * tol;

  std::vector<PaddedDouble> partial_pAp(num_tasks);
  std::vector<PaddedDouble> partial_rs(num_tasks);

  // Initial rs_old = dot(r, r) — single fork-join, before the fused loop.
  double rs_old = 0.0;
  {
    marl::WaitGroup wg(num_tasks);
    for (int i = 0; i < num_tasks; ++i) {
      marl::schedule([&, i] {
        defer(wg.done());
        const int s = i * chunk;
        const int e = (i == num_tasks - 1) ? n : s + chunk;
        double sum = 0.0;
        for (int j = s; j < e; ++j)
          sum += r[j] * r[j];
        partial_rs[i].v = sum;
      });
    }
    wg.wait();
    for (const auto& pp : partial_rs)
      rs_old += pp.v;
  }

  int iters_done = 0;

  // Hoisted out of the loop: each marl::WaitGroup / marl::Event ctor does a
  // heap allocation (shared_ptr to a Data/Shared block holding mutex + CV).
  // Reused via add()/clear() each iteration; wg_iter.wait() at end of iter
  // guarantees no leftover task observes the reset.
  marl::WaitGroup wg_pAp;
  marl::WaitGroup wg_rs;
  marl::WaitGroup wg_iter;
  marl::Event ev_alpha(marl::Event::Mode::Manual);
  marl::Event ev_beta(marl::Event::Mode::Manual);

  for (int iter = 0; iter < max_iters; ++iter) {
    wg_pAp.add(num_tasks);
    wg_rs.add(num_tasks);
    wg_iter.add(num_tasks);
    ev_alpha.clear();
    ev_beta.clear();
    double alpha = 0.0;
    double beta = 0.0;
    double rs_new = 0.0;

    for (int i = 0; i < num_tasks; ++i) {
      marl::schedule([&, i] {
        defer(wg_iter.done());
        const int s = i * chunk;
        const int e = (i == num_tasks - 1) ? n : s + chunk;

        double* ap_s = ap_local[i].data();

        // Phase A+B fused: spmv stripe and partial dot(p, ap) in one pass.
        // sum (= ap[row]) is in-register at the moment we'd write it, and
        // p[row] is the current row index, so we accumulate p[row]*sum here
        // and skip the second read of ap[s..e).
        double dpAp = 0.0;
        for (int row = s; row < e; ++row) {
          double sum = 0.0;
          for (int j = A.row_ptr[row]; j < A.row_ptr[row + 1]; ++j) {
            sum += A.values[j] * p[A.col_indices[j]];
          }
          ap_s[row - s] = sum;
          dpAp += p[row] * sum;
        }
        partial_pAp[i].v = dpAp;
        // Last to finish Phase B reduces, computes alpha, and broadcasts —
        // skips a CV roundtrip via the main thread.
        if (wg_pAp.done()) {
          double pAp = 0.0;
          for (const auto& pp : partial_pAp)
            pAp += pp.v;
          alpha = rs_old / pAp;
          ev_alpha.signal();
        }

        ev_alpha.wait();

        // Phase C: x += alpha*p; r -= alpha*ap; partial dot(r, r).
        // Stripe-local restrict pointers tell the compiler that xs/rs/ps/aps
        // do not alias, which is necessary for clean SIMD codegen of this
        // loop. Scoped to the loop body so the restrict guarantee is local.
        {
          double* __restrict__ xs = x.data() + s;
          double* __restrict__ rs = r.data() + s;
          const double* __restrict__ ps = p.data() + s;
          const double* __restrict__ aps = ap_s;
          const int len = e - s;
          double drs = 0.0;
          for (int k = 0; k < len; ++k) {
            xs[k] += alpha * ps[k];
            rs[k] -= alpha * aps[k];
            drs += rs[k] * rs[k];
          }
          partial_rs[i].v = drs;
        }
        // Last to finish Phase C reduces, publishes rs_new + beta, broadcasts.
        if (wg_rs.done()) {
          double sum = 0.0;
          for (const auto& pp : partial_rs)
            sum += pp.v;
          rs_new = sum;
          beta = sum / rs_old;
          ev_beta.signal();
        }

        ev_beta.wait();

        // Phase D: p = r + beta*p. On the converged iteration beta ~= 0
        // and this write is harmless (the solve is already over).
        {
          double* __restrict__ ps = p.data() + s;
          const double* __restrict__ rs = r.data() + s;
          const int len = e - s;
          for (int k = 0; k < len; ++k)
            ps[k] = rs[k] + beta * ps[k];
        }
      });
    }

    // Reductions, alpha, beta, and the event signals all happen on the
    // last-to-arrive worker now (see if (wg_*.done()) blocks above), so the
    // main thread no longer participates per phase — it just waits for the
    // whole iteration to finish and then checks convergence.
    wg_iter.wait();
    iters_done = iter + 1;
    rs_old = rs_new;
    if (rs_new < tol_sq)
      break;
  }

  // Compute final_residual from rs_old after the loop so it reflects the
  // last iteration's residual whether we converged or hit max_iters.
  const double final_residual = std::sqrt(rs_old);
  printf(" [fused] iters=%d residual=%e\n", iters_done, final_residual);
  return x;
}

// Persistent-workers variant. Same math as the fused version, but each worker
// is scheduled exactly once and runs the *entire* CG loop internally. Main
// only sets up per-iteration sync primitives, signals iter start, and waits.
//
// Why: in the fused version, marl::schedule is called num_tasks * max_iters
// times per solve (e.g. 16 * 420 = 6,720 closure constructions and queue
// pushes). Persistent workers schedule num_tasks tasks once and pay zero
// per-iter scheduling overhead.
//
// Synchronization layout per iteration:
//   ev_iter[K&1] : main signals → workers wake at top of iter K
//   wg_pAp/ev_alpha : last-to-arrive in Phase B reduces, broadcasts alpha
//   wg_rs/ev_beta   : last-to-arrive in Phase C reduces, broadcasts beta+rs_new
//   wg_iter         : workers signal completion of Phase D → main proceeds
// ev_iter is a *pair* of events that ping-pong by iteration parity. This
// avoids a race where a fast worker would race past ev_iter.wait of iter K+1
// before main has cleared the previous signal: at iter K, main signals and
// later clears ev_iter[K&1], while workers in iter K+1 wait on ev_iter[(K+1)&1]
// which has been cleared since the end of iter K-1 (or is in initial cleared
// state). ev_alpha/ev_beta only need a single event each because main clears
// them at iter setup before signaling ev_iter, so workers can't race past
// them across iterations.
//
// Convergence/exit: main sets `stop = true` and signals both ev_iter events;
// workers wake from whichever they're waiting on, observe stop, and return.
//
// The structure is preconditioner-ready: r is preserved (not localized) so a
// future PCG variant can apply z = M^-1 r with stripe-local kernels.
std::vector<double> conjugate_gradient_marl_persistent(
    const SparseMatrix& A,
    const std::vector<double>& b,
    int max_iters,
    double tol) {
  const int n = A.num_rows;
  std::vector<double> x(A.num_cols, 0.0);
  std::vector<double> r = b;
  std::vector<double> p = r;

  auto* sch = marl::Scheduler::get();
  int num_tasks = sch->config().workerThread.count * g_tasks_per_worker;
  num_tasks = std::min<int>(num_tasks, n);
  if (num_tasks <= 0)
    num_tasks = 1;
  const int chunk = (n + num_tasks - 1) / num_tasks;
  const double tol_sq = tol * tol;

  std::vector<std::vector<double>> ap_local(num_tasks);
  for (int i = 0; i < num_tasks; ++i) {
    const int s = i * chunk;
    const int e = (i == num_tasks - 1) ? n : s + chunk;
    ap_local[i].resize(e - s);
  }

  // Prelude: rs_old = dot(r, r). One-shot fork-join, before persistent
  // workers are scheduled.
  double rs_old = 0.0;
  {
    std::vector<PaddedDouble> partials(num_tasks);
    marl::WaitGroup wg(num_tasks);
    for (int i = 0; i < num_tasks; ++i) {
      marl::schedule([&, i] {
        defer(wg.done());
        const int s = i * chunk;
        const int e = (i == num_tasks - 1) ? n : s + chunk;
        double sum = 0.0;
        for (int j = s; j < e; ++j)
          sum += r[j] * r[j];
        partials[i].v = sum;
      });
    }
    wg.wait();
    for (const auto& pp : partials)
      rs_old += pp.v;
  }

  std::vector<PaddedDouble> partial_pAp(num_tasks);
  std::vector<PaddedDouble> partial_rs(num_tasks);

  marl::WaitGroup wg_pAp;
  marl::WaitGroup wg_rs;
  marl::WaitGroup wg_iter;
  marl::WaitGroup wg_workers_done(num_tasks);
  marl::Event ev_alpha(marl::Event::Mode::Manual);
  marl::Event ev_beta(marl::Event::Mode::Manual);
  // Alternating per-iter "go" events. Workers wait on ev_iter[local_iter & 1];
  // main signals ev_iter[iter & 1] and clears it after wg_iter.wait — at
  // which point all workers have advanced to the OTHER event in the pair.
  marl::Event ev_iter0(marl::Event::Mode::Manual);
  marl::Event ev_iter1(marl::Event::Mode::Manual);
  auto ev_iter_at = [&](int k) -> marl::Event& {
    return (k & 1) ? ev_iter1 : ev_iter0;
  };

  // Per-iter shared scalars. Writes from a single last-to-arrive worker;
  // reads by main and other workers go through wg/event happens-before.
  double alpha = 0.0;
  double beta = 0.0;
  double rs_new = 0.0;
  bool stop = false;
  int iters_done = 0;

  // Schedule one persistent worker per stripe.
  for (int i = 0; i < num_tasks; ++i) {
    marl::schedule([&, i] {
      defer(wg_workers_done.done());
      const int s = i * chunk;
      const int e = (i == num_tasks - 1) ? n : s + chunk;
      double* ap_s = ap_local[i].data();
      int local_iter = 0;
      while (true) {
        ev_iter_at(local_iter).wait();
        if (stop)
          return;

        // Phase A+B fused.
        double dpAp = 0.0;
        for (int row = s; row < e; ++row) {
          double sum = 0.0;
          for (int j = A.row_ptr[row]; j < A.row_ptr[row + 1]; ++j) {
            sum += A.values[j] * p[A.col_indices[j]];
          }
          ap_s[row - s] = sum;
          dpAp += p[row] * sum;
        }
        partial_pAp[i].v = dpAp;
        if (wg_pAp.done()) {
          double pAp = 0.0;
          for (const auto& pp : partial_pAp)
            pAp += pp.v;
          alpha = rs_old / pAp;
          ev_alpha.signal();
        }
        ev_alpha.wait();

        // Phase C: x += alpha*p; r -= alpha*ap; partial dot(r, r).
        {
          double* __restrict__ xs = x.data() + s;
          double* __restrict__ rs = r.data() + s;
          const double* __restrict__ ps = p.data() + s;
          const double* __restrict__ aps = ap_s;
          const int len = e - s;
          double drs = 0.0;
          for (int k = 0; k < len; ++k) {
            xs[k] += alpha * ps[k];
            rs[k] -= alpha * aps[k];
            drs += rs[k] * rs[k];
          }
          partial_rs[i].v = drs;
        }
        if (wg_rs.done()) {
          double sum = 0.0;
          for (const auto& pp : partial_rs)
            sum += pp.v;
          rs_new = sum;
          beta = sum / rs_old;
          ev_beta.signal();
        }
        ev_beta.wait();

        // Phase D: p = r + beta*p.
        {
          double* __restrict__ ps = p.data() + s;
          const double* __restrict__ rs = r.data() + s;
          const int len = e - s;
          for (int k = 0; k < len; ++k)
            ps[k] = rs[k] + beta * ps[k];
        }

        wg_iter.done();
        ++local_iter;
      }
    });
  }

  // Main loop: per iter, set up barriers and wake workers.
  for (int iter = 0; iter < max_iters; ++iter) {
    wg_pAp.add(num_tasks);
    wg_rs.add(num_tasks);
    wg_iter.add(num_tasks);
    // Cleared BEFORE ev_iter signal so workers can't race past Phase B
    // with stale alpha/beta from the previous iter.
    ev_alpha.clear();
    ev_beta.clear();
    ev_iter_at(iter).signal();
    wg_iter.wait();
    // Safe to clear: all workers have advanced to wait on the OTHER event.
    ev_iter_at(iter).clear();
    iters_done = iter + 1;
    rs_old = rs_new;
    if (rs_new < tol_sq)
      break;
  }

  // Wake workers and ask them to exit. Signal both events because we don't
  // know which one workers were about to wait on.
  stop = true;
  ev_iter0.signal();
  ev_iter1.signal();
  wg_workers_done.wait();

  const double final_residual = std::sqrt(rs_old);
  printf(" [persistent] iters=%d residual=%e\n", iters_done, final_residual);
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

    // Solve with all three variants for parity check.
    std::vector<double> x = conjugate_gradient_marl(A, b, 100, 1e-6);
    std::vector<double> x_fused =
        conjugate_gradient_marl_fused(A, b, 100, 1e-6);
    std::vector<double> x_persistent =
        conjugate_gradient_marl_persistent(A, b, 100, 1e-6);

    std::cout << "Solution x            : [ ";
    for (auto val : x)
      std::cout << val << " ";
    std::cout << "]" << std::endl;
    std::cout << "Solution x_fused      : [ ";
    for (auto val : x_fused)
      std::cout << val << " ";
    std::cout << "]" << std::endl;
    std::cout << "Solution x_persistent : [ ";
    for (auto val : x_persistent)
      std::cout << val << " ";
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

    // Run one warmup (skip first-call scheduler/cache effects), then `reps`
    // timed repetitions. Report mean/min/max so single-run noise on a
    // laptop (thermal, scheduler, core swaps) doesn't dominate.
    constexpr int reps = 16;
    auto time_solve = [&](const char* label, auto solver) {
      solver(); // warmup
      std::vector<double> times;
      times.reserve(reps);
      std::vector<double> x;
      for (int r = 0; r < reps; ++r) {
        auto t0 = std::chrono::high_resolution_clock::now();
        x = solver();
        auto t1 = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration<double>(t1 - t0).count());
      }
      const double sum = std::accumulate(times.begin(), times.end(), 0.0);
      const double mean = sum / reps;
      const double tmin = *std::min_element(times.begin(), times.end());
      const double tmax = *std::max_element(times.begin(), times.end());
      printf(
          " %-8s tpw=%d reps=%d  mean=%e s  min=%e s  max=%e s\n",
          label,
          g_tasks_per_worker,
          reps,
          mean,
          tmin,
          tmax);
      return x;
    };
    // Sweep tasks_per_worker to study the effect of over-decomposition.
    // tpw=1: one stripe per worker (current default).
    // tpw>1: each worker holds multiple stripe-fibers; stripes are smaller.
    for (int tpw : {1, 2}) {
      g_tasks_per_worker = tpw;
      time_solve(
          "classic", [&] { return conjugate_gradient_marl(A, b, 1000, 1e-6); });
      time_solve("fused", [&] {
        return conjugate_gradient_marl_fused(A, b, 1000, 1e-6);
      });
      time_solve("persist", [&] {
        return conjugate_gradient_marl_persistent(A, b, 1000, 1e-6);
      });
    }
  }

  return 0;
}
