// SPDX-License-Identifier: BSD-3-Clause

// Padded DP containers for dummest.cc: PaddedVec, Matrix, and the
// ExpScore-backed vectors/matrices/side accumulators.
//
// Include AFTER Float, simd_t, and simdWidth are defined. Pulls in
// dummest-ovf.hh for ExpScore and the score helpers.
#pragma once

#include "dummest-score.hh"
#include "dummest-vectraits.hh"

#include <algorithm>
#include <array>
#include <cstddef>
#include <vector>

// Poison-fill trait: value used for uninitialized scratch under DUMMEST_POISON.
// Seeds (explicit zero fills) keep using T{}; only "must-overwrite" paths use
// poisonValue(). Without DUMMEST_POISON this is T{} (no-op).
template <typename T> struct PoisonFill { static T value() { return T{}; } };

template <> struct PoisonFill<float> {
    static float value() { return dummestPoisonFloat<float>(); }
};
template <> struct PoisonFill<int> {
    static int value() {
#ifdef DUMMEST_POISON
        return dummestPoisonInt();
#else
        return 0;
#endif
    }
};
template <> struct PoisonFill<ExpScore> {
    static ExpScore value() {
#ifdef DUMMEST_POISON
        return poisonScore();
#else
        return ExpScore{0.0f, 0};
#endif
    }
};
// simd_t poison is defined here (simd_t is complete at this point: padded.hh
// is included AFTER Float/simd_t/simdWidth are defined).
template <> struct PoisonFill<simd_t> {
    static simd_t value() { return VecTraits<simd_t>::poison(); }
};
template <size_t N> struct PoisonFill<std::array<ExpScore, N>> {
    static std::array<ExpScore, N> value() {
        std::array<ExpScore, N> a{};
        for (auto &s : a) s = PoisonFill<ExpScore>::value();
        return a;
    }
};
template <size_t N> struct PoisonFill<std::array<int, N>> {
    static std::array<int, N> value() {
        std::array<int, N> a{};
        a.fill(PoisonFill<int>::value());
        return a;
    }
};
template <size_t N> struct PoisonFill<std::array<float, N>> {
    static std::array<float, N> value() {
        std::array<float, N> a{};
        a.fill(PoisonFill<float>::value());
        return a;
    }
};

template <typename T, int PrePad = 0, int PostPad = 0>
class PaddedVec {
    std::vector<T> data_;
    int logical_size_ = 0;
public:
    PaddedVec() = default;

    T &operator[](int i) { return data_[PrePad + i]; }

    const T &operator[](int i) const { return data_[PrePad + i]; }

    T *data() { return data_.data() + PrePad; }

    const T *data() const { return data_.data() + PrePad; }

    int size() const { return logical_size_; }
    bool empty() const { return logical_size_ == 0; }

    auto begin() { return data(); }
    auto end() { return data() + logical_size_; }
    auto begin() const { return data(); }
    auto end() const { return data() + logical_size_; }

    void resize(int n, T val = T{}) {
        logical_size_ = n;
        data_.assign(n + PrePad + PostPad, val);
        std::fill_n(data_.data(), PrePad, T{});
        std::fill_n(data_.data() + PrePad + n, PostPad, T{});
    }

    void assign(int n, T val = T{}) { resize(n, val); }

    void set_logical_size(int n) {
        logical_size_ = n;
#ifdef DUMMEST_POISON
        // Uninitialized path (Matrix::resizeNoFill): poison the whole logical
        // range so a missing DP write aborts on first read instead of hiding
        // as stale-previous-batch data. vector::resize would preserve old
        // elements, so explicitly fill after growing.
        data_.resize(n + PrePad + PostPad);
        T pv = PoisonFill<T>::value();
        std::fill_n(data_.data() + PrePad, n, pv);
        std::fill_n(data_.data(), PrePad, pv);
        std::fill_n(data_.data() + PrePad + n, PostPad, pv);
#else
        data_.resize(n + PrePad + PostPad, T{});
        std::fill_n(data_.data(), PrePad, T{});
        std::fill_n(data_.data() + PrePad + n, PostPad, T{});
#endif
    }
};

// Multiply one PaddedVec range by a prebuilt per-lane factor (vectorized).
// lo may be negative for front-padded buffers (right_side starts at -3);
// data() points at logical 0, so padding is reachable through it.
template <typename T, int PrePad, int PostPad>
inline void rescaleRow(PaddedVec<T, PrePad, PostPad> &vec, int lo, int hi, simd_t factor) {
    rescaleRow(vec.data(), lo, hi, factor);
}

// Batched 2D matrix: rows of PaddedVec<T>. Lane k of element [i][j]
// holds the value for sequence k at (row i, column j), so a whole batch of
// sequences moves through the DP with one vector op per cell. T is the lane
// vector (simd_t in use).
template <typename T, int PrePad = 0, int PostPad = 0>
class Matrix {
    std::vector<PaddedVec<T, PrePad, PostPad>> rows_;
    int logical_cols_ = 0;
public:
    Matrix() = default;

    void assign(size_t nrows, int ncols, T init = T{}) {
        logical_cols_ = ncols;
        rows_.resize(nrows);
        for (auto& r : rows_) r.resize(ncols, init);
    }

    // Resize without filling the logical range (pads still zeroed). Caller
    // must overwrite every logical cell before reading; use fillRow / manual
    // stores for seed/border cells that are read but never written.
    // Under DUMMEST_POISON the logical range is NaN-filled (see
    // PaddedVec::set_logical_size) so a missing write aborts on first read.
    void resizeNoFill(size_t nrows, int ncols) {
        logical_cols_ = ncols;
        rows_.resize(nrows);
        for (auto& r : rows_) r.set_logical_size(ncols);
    }

    // Zero one full logical row (e.g. W1's unwritten row plen+1).
    void fillRow(size_t i, T val = T{}) {
        std::fill_n(rows_[i].data(), logical_cols_, val);
    }

    T get(size_t i, int j) const { return rows_[i][j]; }
    void set(size_t i, int j, T v) { rows_[i][j] = v; }
    T *row_ptr(size_t i) { return rows_[i].data(); }
    const T *row_ptr(size_t i) const { return rows_[i].data(); }

    int cols() const { return logical_cols_; }
    size_t rows() const { return rows_.size(); }
};

// Per-lane flank totals. PrePad allows negative indexing.
template <int PrePad = 0>
class ExpVector {
    std::vector<std::array<ExpScore, simdWidth>> data_;
public:
    ExpVector() = default;

    void assign(int n) {
        // Stays zeroed: zero is the correct start for these totals.
        data_.assign((size_t)n + PrePad, std::array<ExpScore, simdWidth>{});
    }

    std::array<ExpScore, simdWidth> &operator[](int j) { return data_[(size_t)PrePad + j]; }
    const std::array<ExpScore, simdWidth> &operator[](int j) const { return data_[(size_t)PrePad + j]; }
};

// Match and best tables, one score per lane.
// Reads before position 0 see zero. Tiny values read as zero.
template <int PrePad = 0>
class ExpMatrix {
    std::vector<std::array<float, simdWidth>> m_;
    std::vector<std::array<int32_t, simdWidth>> e_;
    int logical_cols_ = 0;
public:
    ExpMatrix() = default;

    void assign(size_t nrows, int ncols) {
        logical_cols_ = ncols;
        // Stays zeroed: edge cells are valid zero starts.
        m_.assign(nrows * (ncols + PrePad), std::array<float, simdWidth>{});
        e_.assign(nrows * (ncols + PrePad), std::array<int32_t, simdWidth>{});
    }

    ExpScore get_lane(size_t i, int j, int k) const {
        const size_t o = i * ((size_t)logical_cols_ + PrePad) + (j + PrePad);
        ExpScore s{m_[o][k], e_[o][k]};
        DUMMEST_POISON_ASSERT(!isPoisonScore(s));
        return s;
    }
    // Fast access to one cell's lanes.
    float *m_ptr(size_t i, int j) { return m_[i * ((size_t)logical_cols_ + PrePad) + (j + PrePad)].data(); }
    const float *m_ptr(size_t i, int j) const { return m_[i * ((size_t)logical_cols_ + PrePad) + (j + PrePad)].data(); }
    int32_t *e_ptr(size_t i, int j) { return e_[i * ((size_t)logical_cols_ + PrePad) + (j + PrePad)].data(); }
    const int32_t *e_ptr(size_t i, int j) const { return e_[i * ((size_t)logical_cols_ + PrePad) + (j + PrePad)].data(); }
    int cols() const { return logical_cols_; }
    size_t rows() const {
        size_t stride = (size_t)logical_cols_ + PrePad;
        return stride == 0 ? 0 : m_.size() / stride;
    }
};

// Running cut totals. Halved whole-row at a time when too large.
template <typename Vec, int PrePad = 0, int PostPad = 0>
struct PaddedSide {
    PaddedVec<Vec, PrePad, PostPad> m;

    void assign(int n, Vec val = VecTraits<Vec>::zero()) {
        m.assign(n, val);
    }
    Vec &operator[](int i) { return m[i]; }
    const Vec &operator[](int i) const { return m[i]; }
    int size() const { return m.size(); }
};
