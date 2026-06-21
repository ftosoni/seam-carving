// Seam Carving for Content-Aware Image Resizing
//
// Copyright (c) 2026 Francesco Tosoni
//
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause License. This program is distributed in the hope
// that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
// warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//
// You should have received a copy of the BSD 3-Clause License along with this
// program (see the LICENSE file); if not, see
// <https://opensource.org/license/bsd-3-clause>.
//
// SPDX-License-Identifier: BSD-3-Clause

// -----------------------------------------------------------------------------
// Visualisation helpers (implementation). See visualize.h for the rationale and
// the public contract. This file owns the colour-map data and the pixel-level
// rendering; it has no knowledge of how energy maps or seams are computed.
// -----------------------------------------------------------------------------

#include "visualize.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace viz {

namespace {

// Each colour map is stored as a small table of anchor colours, evenly spaced in
// [0, 1]. A scalar t is mapped by locating the surrounding anchors and linearly
// interpolating between them. The anchors are sampled from matplotlib's viridis
// and magma maps (public domain, CC0); a dozen anchors reproduce these smooth
// maps closely enough for figures, and keep the embedded data readable.
using Anchor = std::array<uint8_t, 3>;

// Viridis: 11 anchors at t = 0.0, 0.1, ..., 1.0.
constexpr std::array<Anchor, 11> kViridis = {{
    { 68,   1,  84}, { 72,  36, 117}, { 65,  68, 135}, { 53,  95, 141},
    { 42, 120, 142}, { 33, 145, 140}, { 34, 168, 132}, { 68, 191, 112},
    {122, 209,  81}, {189, 223,  38}, {253, 231,  37},
}};

// Magma: 9 anchors at t = 0.0, 0.125, ..., 1.0.
constexpr std::array<Anchor, 9> kMagma = {{
    {  0,   0,   4}, { 28,  16,  68}, { 79,  18, 123}, {129,  37, 129},
    {181,  54, 122}, {229,  80, 100}, {251, 135,  97}, {254, 196, 136},
    {252, 253, 191},
}};

// Map t in [0, 1] through a table of evenly spaced anchors by linear interpolation.
template <std::size_t N>
Anchor sample_map(const std::array<Anchor, N>& table, double t) {
    t = std::clamp(t, 0.0, 1.0);
    const double scaled = t * (N - 1);
    const std::size_t lo = static_cast<std::size_t>(scaled);
    const std::size_t hi = std::min(lo + 1, N - 1);
    const double frac = scaled - static_cast<double>(lo);
    Anchor out{};
    for (int c = 0; c < 3; ++c) {
        const double v = table[lo][c] + frac * (table[hi][c] - table[lo][c]);
        out[c] = static_cast<uint8_t>(std::lround(v));
    }
    return out;
}

// BT.601 luminance of an RGB triple, matching the energy computation.
inline uint8_t luma601(uint8_t r, uint8_t g, uint8_t b) {
    const double y = 0.299 * r + 0.587 * g + 0.114 * b;
    return static_cast<uint8_t>(std::lround(std::clamp(y, 0.0, 255.0)));
}

} // namespace

Colormap colormap_from_string(const std::string& name, bool* ok) {
    if (ok) *ok = true;
    if (name == "viridis") return Colormap::Viridis;
    if (name == "magma") return Colormap::Magma;
    if (name == "grey" || name == "gray") return Colormap::Grey;
    if (ok) *ok = false;
    return Colormap::Viridis;
}

Image render_scalar_field(const std::vector<double>& field, int width, int height,
                          Colormap cmap) {
    Image out;
    out.width = width;
    out.height = height;
    out.channels = 3;
    out.data.resize(static_cast<size_t>(width) * height * 3);

    // Linear min-max normalisation over the whole field.
    double lo = field.empty() ? 0.0 : field[0];
    double hi = lo;
    for (double v : field) {
        lo = std::min(lo, v);
        hi = std::max(hi, v);
    }
    const double range = (hi > lo) ? (hi - lo) : 1.0;

    for (int i = 0; i < width * height; ++i) {
        const double t = (field[i] - lo) / range;
        Anchor rgb{};
        switch (cmap) {
            case Colormap::Viridis: rgb = sample_map(kViridis, t); break;
            case Colormap::Magma:   rgb = sample_map(kMagma, t);   break;
            case Colormap::Grey: {
                const uint8_t g = static_cast<uint8_t>(std::lround(std::clamp(t, 0.0, 1.0) * 255.0));
                rgb = {g, g, g};
                break;
            }
        }
        out.data[i * 3 + 0] = rgb[0];
        out.data[i * 3 + 1] = rgb[1];
        out.data[i * 3 + 2] = rgb[2];
    }
    return out;
}

Image render_seam_overlay(const Image& background,
                          const std::vector<std::vector<int>>& seams,
                          const uint8_t seam_rgb[3],
                          bool greyscale_background) {
    const int w = background.width;
    const int h = background.height;
    const int bc = background.channels;

    Image out;
    out.width = w;
    out.height = h;
    out.channels = 3;
    out.data.resize(static_cast<size_t>(w) * h * 3);

    // Copy the background into a 3-channel image, optionally desaturating it so
    // the coloured seams stand out. Any alpha channel of the source is dropped.
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const int src = (y * w + x) * bc;
            const uint8_t r = background.data[src + 0];
            const uint8_t g = background.data[src + 1];
            const uint8_t b = background.data[src + 2];
            const int dst = (y * w + x) * 3;
            if (greyscale_background) {
                const uint8_t l = luma601(r, g, b);
                out.data[dst + 0] = l;
                out.data[dst + 1] = l;
                out.data[dst + 2] = l;
            } else {
                out.data[dst + 0] = r;
                out.data[dst + 1] = g;
                out.data[dst + 2] = b;
            }
        }
    }

    // Paint the seams. Each seam holds one column per row; bounds are checked
    // defensively so an out-of-range column is simply skipped.
    for (const auto& seam : seams) {
        const int rows = std::min<int>(static_cast<int>(seam.size()), h);
        for (int y = 0; y < rows; ++y) {
            const int x = seam[y];
            if (x < 0 || x >= w) continue;
            const int dst = (y * w + x) * 3;
            out.data[dst + 0] = seam_rgb[0];
            out.data[dst + 1] = seam_rgb[1];
            out.data[dst + 2] = seam_rgb[2];
        }
    }
    return out;
}

} // namespace viz
