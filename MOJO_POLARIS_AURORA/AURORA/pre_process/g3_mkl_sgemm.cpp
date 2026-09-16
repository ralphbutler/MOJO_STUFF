// g3_mkl_sgemm.cpp — vendor-library matmul yardstick for G3: oneMKL SGEMM on one PVC tile.
//
// Aurora analog of the cuBLAS number measured on Polaris (via torch on the A100). Calls
// oneMKL's SYCL BLAS directly (the frameworks module's torch was broken on next-eval,
// 2026-09-15). Same method and formula as the Mojo kernels: USM device buffers, warmup,
// ITERS launches, one wait; GFLOP/s = 2*N^3 / seconds-per-pass. Correctness: the same
// generators as g3_matmul_lz.mojo, checked on sampled cells against a CPU reference.
//
// Build (aurora-uan-0007):  icpx -fsycl -qmkl -O2 g3_mkl_sgemm.cpp -o g3_mkl_sgemm
// Run   (compute node):     ZE_FLAT_DEVICE_HIERARCHY=FLAT ZE_AFFINITY_MASK=0 ./g3_mkl_sgemm 2048 50

#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

static float a_val(std::int64_t i, std::int64_t j) { return float((i * 7 + j * 3) % 13) * 0.1f - 0.6f; }
static float b_val(std::int64_t i, std::int64_t j) { return float((i * 5 + j * 11) % 17) * 0.1f - 0.8f; }

int main(int argc, char **argv) {
    const std::int64_t n = argc > 1 ? std::atoll(argv[1]) : 2048;
    const int iters = argc > 2 ? std::atoi(argv[2]) : 50;

    sycl::queue q{sycl::gpu_selector_v};
    std::cout << "=== G3 yardstick: oneMKL SGEMM (SYCL USM) ===\n"
              << "  device : " << q.get_device().get_info<sycl::info::device::name>() << "\n"
              << "  N      : " << n << "   iters: " << iters << "\n";

    // Column-major storage: element (i, j) at [i + j*n].
    std::vector<float> ha(n * n), hb(n * n), hc(n * n, 0.0f);
    for (std::int64_t j = 0; j < n; ++j)
        for (std::int64_t i = 0; i < n; ++i) {
            ha[i + j * n] = a_val(i, j);
            hb[i + j * n] = b_val(i, j);
        }
    float *a = sycl::malloc_device<float>(n * n, q);
    float *b = sycl::malloc_device<float>(n * n, q);
    float *c = sycl::malloc_device<float>(n * n, q);
    q.memcpy(a, ha.data(), n * n * sizeof(float)).wait();
    q.memcpy(b, hb.data(), n * n * sizeof(float)).wait();

    using oneapi::mkl::transpose;
    auto gemm = [&]() {
        return oneapi::mkl::blas::column_major::gemm(q, transpose::nontrans, transpose::nontrans,
                                                     n, n, n, 1.0f, a, n, b, n, 0.0f, c, n);
    };

    auto w0 = std::chrono::steady_clock::now();
    for (int k = 0; k < 5; ++k) gemm();
    q.wait();
    auto w1 = std::chrono::steady_clock::now();

    auto t0 = std::chrono::steady_clock::now();
    for (int k = 0; k < iters; ++k) gemm();
    q.wait();
    auto t1 = std::chrono::steady_clock::now();

    double avg_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    double gflops = 2.0 * double(n) * double(n) * double(n) / (avg_ms / 1e3) / 1e9;

    q.memcpy(hc.data(), c, n * n * sizeof(float)).wait();
    int mismatches = 0, checked = 0;
    std::uint64_t x = 12345;
    for (int s = 0; s < 260; ++s) {
        x = (x * 1103515245ULL + 12345ULL) % 2147483648ULL;
        std::int64_t cell = std::int64_t(x % std::uint64_t(n * n));
        std::int64_t i = cell % n, j = cell / n;           // column-major
        float expected = 0.0f;
        for (std::int64_t k = 0; k < n; ++k) expected += a_val(i, k) * b_val(k, j);
        double ae = std::fabs(double(hc[cell]) - double(expected));
        double re = ae / (std::fabs(double(expected)) + 1e-12);
        if (re > 1e-3 && ae > 1e-3) ++mismatches;
        ++checked;
    }

    std::cout << "  check  : SAMPLED - " << mismatches << " mismatches / " << checked << " checked\n"
              << "  RESULT : " << (mismatches == 0 ? "PASS" : "FAIL") << "\n"
              << "  warmup : " << std::chrono::duration<double, std::milli>(w1 - w0).count() << " ms (5 calls)\n"
              << "  avg time: " << avg_ms << " ms over " << iters << " iters\n"
              << "  perf    : " << gflops << " GFLOP/s\n";

    sycl::free(a, q);
    sycl::free(b, q);
    sycl::free(c, q);
    return mismatches == 0 ? 0 : 1;
}
