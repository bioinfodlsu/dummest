// Unit tests for the score layer: batch kernels vs scalar oracles, the
// row-scaled compute helpers, ExpMatrix SoA access, and VecTraits.
//
// Build (uses the Kokkos from this repo; links nothing but Kokkos headers):
//   KOK=<repo>/kokkos-5.1.1; BLDCFG=<build>/kokkos-5.1.1
//   g++ -std=c++20 -O2 -march=native -I<repo> -I$KOK/simd/src \
//       -I$KOK/core/src -I$BLDCFG -I$KOK/tpls/desul/include \
//       -I$BLDCFG/core/src -I$KOK/tpls/mdspan/include \
//       tools/test_batch_pairs.cc -o /tmp/test_batch_pairs && /tmp/test_batch_pairs
// Double build: add -DTEST_DOUBLE (exercises the 64-bit bit-trick paths).
#include <Kokkos_SIMD.hpp>
// clang-format off
#include "can_i_haz_simd.hh"
// clang-format on

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <limits>
#include <random>

#ifdef TEST_DOUBLE
typedef double Float;
typedef SimdDbl SimdFloat;
#else
typedef float Float;
typedef SimdFlt SimdFloat;
#endif
using simd_t = Kokkos::Experimental::simd<Float>;
constexpr auto simdWidth = simd_t::size();

#include "dummest-score.hh"
#include "dummest-batch.hh"
#include "dummest-scalerow.hh"
#include "dummest-padded.hh"

static int failures = 0;
#define CHECK(cond, ...) do { \
    if (!(cond)) { ++failures; std::printf("FAIL %s:%d: ", __FILE__, __LINE__); \
        std::printf(__VA_ARGS__); std::printf("\n"); } \
} while (0)

static bool sameScore(const ExpScore &a, const ExpScore &b) {
    uint32_t am, bm;
    std::memcpy(&am, &a.m, 4);
    std::memcpy(&bm, &b.m, 4);
    return am == bm && a.e == b.e;
}

// Random normalized score with a wide exponent range; ~1 in 8 is zero.
template <typename Rng> static ExpScore rndScore(Rng &r) {
    std::uniform_real_distribution<float> m(0.5f, 1.0f);
    std::uniform_int_distribution<int> e(-2000, 2000);
    if (r() % 8 == 0) return ExpScore{0.0f, 0};
    return ExpScore{m(r), e(r)};
}

static Float laneOf(simd_t v, int k) {
    alignas(64) Float buf[simdWidth];
    Kokkos::Experimental::simd_unchecked_store(v, buf, Kokkos::Experimental::simd_flag_default);
    return buf[k];
}

int main() {
    std::mt19937_64 rng(12345);
    constexpr int W = simdWidth;
    ExpScore A[simdWidth], B[simdWidth], O[simdWidth], want[simdWidth];
    double F[simdWidth];

    // Struct batch kernels vs their scalar oracles.
    for (int iter = 0; iter < 20000; ++iter) {
        for (int k = 0; k < W; ++k) {
            A[k] = rndScore(rng);
            B[k] = rndScore(rng);
            F[k] = (k % 7 == 0) ? 0.0 : std::ldexp(1.0, (int)(rng() % 80) - 40);
        }
        batchMulScores(A, B, O, W, 0);
        for (int k = 0; k < W; ++k) {
            want[k] = mulScores(A[k], B[k]);
            CHECK(sameScore(O[k], want[k]), "mul k=%d", k);
        }
        batchMulScores(A, B, O, W, STATIC_SHIFT);
        for (int k = 0; k < W; ++k) {
            want[k] = mulScoresStaticShifted(A[k], B[k]);
            CHECK(sameScore(O[k], want[k]), "mulShift k=%d", k);
        }
        batchAddScores(A, B, O, W);
        for (int k = 0; k < W; ++k) {
            want[k] = addScores(A[k], B[k]);
            CHECK(sameScore(O[k], want[k]), "add k=%d", k);
        }
        batchScaleScores(A, F, O, W);
        for (int k = 0; k < W; ++k) {
            want[k] = scaleScore(A[k], F[k]);
            CHECK(sameScore(O[k], want[k]), "scale k=%d", k);
        }
    }

    // Plane product vs the struct product.
    {
        float am[simdWidth], bm[simdWidth], om[simdWidth];
        int ae[simdWidth], be[simdWidth], oe[simdWidth];
        for (int iter = 0; iter < 5000; ++iter) {
            for (int k = 0; k < W; ++k) {
                ExpScore a = rndScore(rng), b = rndScore(rng);
                am[k] = a.m; ae[k] = a.e; bm[k] = b.m; be[k] = b.e;
            }
            batchMulScorePlanes(am, ae, bm, be, STATIC_SHIFT, om, oe, W);
            for (int k = 0; k < W; ++k) {
                ExpScore o{om[k], oe[k]};
                ExpScore w = mulScoresStaticShifted(ExpScore{am[k], ae[k]}, ExpScore{bm[k], be[k]});
                CHECK(sameScore(o, w), "planes k=%d", k);
            }
        }
    }

    // Promotion: batchPromoteRescaled vs promoteRescaledScore, and the AVX2
    // path (n == simdWidth) against the same oracle.
    {
        std::uniform_real_distribution<double> lg(-30.0, 30.0);
        std::uniform_int_distribution<int> cumDist(-1000005, 1000005);
        for (int iter = 0; iter < 5000; ++iter) {
            Float in[simdWidth];
            int cum[simdWidth];
            for (int k = 0; k < W; ++k) {
                int pick = (int)(rng() % 10);
                double v = std::pow(10.0, lg(rng));
                Float c = (Float)v;
                if (pick == 0) c = (Float)0.0;
                else if (pick == 1) c = -(Float)v;
                else if (pick == 2) c = (Float)INFINITY;
                else if (pick == 3) c = (Float)NAN;
                else if (pick == 4) c = (Float)1e-40; // subnormal in float
                in[k] = c;
                cum[k] = cumDist(rng);
            }
            float pm[simdWidth], om2[simdWidth];
            int pe[simdWidth], oe2[simdWidth];
            batchPromoteRescaled(in, cum, pm, pe, W);
            batchPromoteRescaled(in, cum, om2, oe2, W); // idempotent
            for (int k = 0; k < W; ++k) {
                ExpScore w = promoteRescaledScore(in[k], cum[k]);
                if (std::fpclassify(in[k]) == FP_SUBNORMAL) w = ExpScore{0.0f, 0};
                uint32_t a, b;
                std::memcpy(&a, &pm[k], 4); std::memcpy(&b, &w.m, 4);
                CHECK(a == b && pe[k] == w.e, "promote k=%d", k);
                std::memcpy(&a, &om2[k], 4);
                CHECK(a == b && oe2[k] == pe[k], "promote-idem k=%d", k);
            }
            // Partial batch uses the scalar fallback and must match too.
            {
                float p3[simdWidth]; int e3[simdWidth];
                batchPromoteRescaled(in, cum, p3, e3, W > 3 ? 3 : W);
                for (int k = 0; k < (W > 3 ? 3 : W); ++k) {
                    ExpScore w = promoteRescaledScore(in[k], cum[k]);
                    if (std::fpclassify(in[k]) == FP_SUBNORMAL) w = ExpScore{0.0f, 0};
                    uint32_t a, b;
                    std::memcpy(&a, &p3[k], 4); std::memcpy(&b, &w.m, 4);
                    CHECK(a == b && e3[k] == w.e, "promote3 k=%d", k);
                }
            }
        }
    }

    // splitNormalizedDouble vs frexp+narrow, across magnitudes.
    {
        std::uniform_real_distribution<double> lg(-320.0, 320.0);
        for (int i = 0; i < 20000; ++i) {
            double x = (i % 5 == 0) ? 0.0 : std::pow(10.0, lg(rng));
            float m1; int e1;
            splitNormalizedDouble(x, m1, e1);
            int ex = 0; double m0 = std::frexp(x, &ex);
            float mWant = (float)m0;
            int exWant = ex;
            if (std::fpclassify(x) == FP_SUBNORMAL) { mWant = 0.0f; exWant = 0; }
            uint32_t a, b;
            std::memcpy(&a, &m1, 4); std::memcpy(&b, &mWant, 4);
            CHECK(a == b && e1 == exWant, "split x=%g", x);
        }
        {
            float m; int e;
            splitNormalizedDouble(INFINITY, m, e);
            CHECK(std::isinf(m) && e == 0, "split inf");
            splitNormalizedDouble(std::numeric_limits<double>::denorm_min(), m, e);
            CHECK(m == 0.0f && e == 0, "split subnormal flush");
        }
    }

    // Row scaling round-trip: e <= E, normalized mantissas.
    {
        std::uniform_int_distribution<int> eDist(-3000, 0);
        for (int iter = 0; iter < 50000; ++iter) {
            ExpScore a = rndScore(rng);
            int E = a.e + (int)(rng() % 100);
            float s = scoreToRowScale(a.m, a.e, E);
            ExpScore back = scoreFromRowScale(s, E);
            CHECK(sameScore(back, a), "rowRoundTrip m=%g e=%d E=%d", a.m, a.e, E);
        }
        // A zero mantissa always maps to row-scale zero.
        CHECK(scoreToRowScale(0.0f, -5, 0) == 0.0f, "row zero");
    }

    // ExpMatrix: SoA accessors agree, padding and tail stay zero.
    {
        ExpMatrix<3> M;
        const int rows = 4, cols = 6;
        M.assign(rows, cols);
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                for (int k = 0; k < W; ++k) {
                    ExpScore v{(float)(0.5 + 0.01 * (i * 7 + j * 3 + k)), i * 100 + j * 10 + k};
                    M.m_ptr(i, j)[k] = v.m;
                    M.e_ptr(i, j)[k] = v.e;
                }
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                for (int k = 0; k < W; ++k) {
                    CHECK(M.get_lane(i, j, k).m == M.m_ptr(i, j)[k], "soa m %d %d %d", i, j, k);
                    CHECK(M.get_lane(i, j, k).e == M.e_ptr(i, j)[k], "soa e %d %d %d", i, j, k);
                }
        // Front padding (negative columns) reads as zero.
        for (int i = 0; i < rows; ++i)
            for (int j = -3; j < 0; ++j)
                for (int k = 0; k < W; ++k)
                    CHECK(M.get_lane(i, j, k).m == 0.0f && M.get_lane(i, j, k).e == 0,
                          "pad %d %d %d", i, j, k);
        // No tail columns exist here; a fresh assign of larger cols is zero.
        ExpMatrix<0> Z;
        Z.assign(2, 5);
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 5; ++j)
                for (int k = 0; k < W; ++k)
                    CHECK(Z.get_lane(i, j, k).m == 0.0f && Z.get_lane(i, j, k).e == 0,
                          "fresh %d %d %d", i, j, k);
    }

    // VecTraits: gathers, length mask, and score packing.
    {
        alignas(64) Float table[8] = {1.0f, 2.0f, 4.0f, 8.0f, 16.0f, 32.0f, 64.0f, 128.0f};
        char idx[simdWidth];
        for (int k = 0; k < W; ++k) idx[k] = (char)(k % 8);
        simd_t g = VecTraits<simd_t>::lookupEmit(table, idx);
        for (int k = 0; k < W; ++k)
            CHECK(laneOf(g, k) == table[idx[k]], "lookup k=%d", k);

        alignas(64) Float lens[simdWidth];
        for (int k = 0; k < W; ++k) lens[k] = (Float)(k + 1);
        for (int j = 0; j < W + 1; ++j) {
            simd_t m = VecTraits<simd_t>::activeMask(j, lens);
            for (int k = 0; k < W; ++k)
                CHECK(laneOf(m, k) == (j < (int)lens[k] ? (Float)1 : (Float)0),
                      "mask j=%d k=%d", j, k);
        }

        ExpScore arr[simdWidth];
        for (int k = 0; k < W; ++k) arr[k] = rndScore(rng);
        int active = W > 2 ? W - 2 : 1;
        simd_t p = VecTraits<simd_t>::packScores(arr, active);
        for (int k = 0; k < W; ++k) {
            Float w = (k < active) ? scoreToFloatClamped(arr[k]) : (Float)0.0;
            CHECK(laneOf(p, k) == w, "pack k=%d", k);
        }
    }

    if (failures == 0) std::printf("ALL SCORE TESTS PASSED (width %d, %s)\n",
        (int)W, sizeof(Float) == 8 ? "double" : "float");
    else std::printf("%d FAILURES\n", failures);
    return failures != 0;
}
