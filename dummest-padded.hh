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
        data_.resize(n + PrePad + PostPad, T{});
        std::fill_n(data_.data(), PrePad, T{});
        std::fill_n(data_.data() + PrePad + n, PostPad, T{});
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

// Scalar per-lane ExpScore vector for EV side buffers (left_side_EV,
// right_side_EV). PrePad supports negative indexing (right_side_EV writes
// j-3); operator[] exposes the per-lane array used by the fixup loops.
template <int PrePad = 0>
class ExpVector {
    std::vector<std::array<ExpScore, simdWidth>> data_;
public:
    ExpVector() = default;

    void assign(int n) {
        data_.assign((size_t)n + PrePad, std::array<ExpScore, simdWidth>{});
    }

    std::array<ExpScore, simdWidth> &operator[](int j) { return data_[(size_t)PrePad + j]; }
    const std::array<ExpScore, simdWidth> &operator[](int j) const { return data_[(size_t)PrePad + j]; }
};

// Scalar per-lane ExpScore matrix for combined scores (X, prefixOptimal,
// suffixOptimal). SoA layout: separate mantissa/exponent planes, one
// float[simdWidth] + one int32[simdWidth] per cell, so lane-vectorized
// kernels can stream each plane unit-stride. get_lane is by value; PrePad
// supports negative-column reads (j-3). Subnormal mantissas are not
// preserved: any lane that would need one maps to the zero sentinel
// (matches FTZ/DAZ hardware semantics).
template <int PrePad = 0>
class ExpMatrix {
    std::vector<std::array<float, simdWidth>> m_;
    std::vector<std::array<int32_t, simdWidth>> e_;
    int logical_cols_ = 0;
public:
    ExpMatrix() = default;

    void assign(size_t nrows, int ncols) {
        logical_cols_ = ncols;
        m_.assign(nrows * (ncols + PrePad), std::array<float, simdWidth>{});
        e_.assign(nrows * (ncols + PrePad), std::array<int32_t, simdWidth>{});
    }

    ExpScore get_lane(size_t i, int j, int k) const {
        const size_t o = i * ((size_t)logical_cols_ + PrePad) + (j + PrePad);
        return ExpScore{m_[o][k], e_[o][k]};
    }
    // SoA-native lane access for vectorized kernels: 8 contiguous mantissas
    // / exponents of cell (i,j), lane k = ptr[k].
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

// Side accumulator at row scale: plain mantissas, no per-lane
// exponent. Overflow is handled by whole-row 2^-64 rescaling
// (inherited row scale), not by side adjustment.
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
