// An Analysis and Implementation of Seam Carving for Content-Aware Image Resizing
//
// Copyright (c) 2026 Francesco Tosoni
//
// This file is part of a program implementing an algorithm possibly linked to the
// patent US 7,747,107 B2 ("Method for retargeting images", S. Avidan and A. Shamir,
// Mitsubishi Electric Research Laboratories). This file is made available for the
// exclusive aim of serving as scientific tool to verify the soundness and
// completeness of the algorithm description. Compilation, execution and
// redistribution of this file may violate patents rights in certain countries.
// The situation being different for every country and changing over time, it is
// your responsibility to determine which patent rights restrictions apply to you
// before you compile, use, modify, or redistribute this file. A patent lawyer is
// qualified to make this determination. If and only if they don't conflict with
// any patent terms, you can benefit from the following license terms attached to
// this file.
//
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause License. This program is distributed in the hope
// that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
// warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//
// You should have received a copy of the BSD 3-Clause License along with this
// program (see the LICENCE file); if not, see
// <https://opensource.org/license/bsd-3-clause>.
//
// SPDX-License-Identifier: BSD-3-Clause

// -----------------------------------------------------------------------------
// Visualisation helpers (implementation). See visualise.h for the rationale and
// the public contract. This file owns the colour-map data and the pixel-level
// rendering; it has no knowledge of how energy maps or seams are computed.
// -----------------------------------------------------------------------------

#include "visualise.h"

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

// Colour of t in [0, 1] (clamped) under the chosen colour map.
Anchor colour_at(Colourmap cmap, double t) {
    switch (cmap) {
        case Colourmap::Magma: return sample_map(kMagma, t);
        case Colourmap::Grey: {
            const uint8_t g = static_cast<uint8_t>(std::lround(std::clamp(t, 0.0, 1.0) * 255.0));
            return {g, g, g};
        }
        case Colourmap::Viridis:
        default: return sample_map(kViridis, t);
    }
}

// BT.601 luminance of an RGB triple, matching the energy computation.
inline uint8_t luma601(uint8_t r, uint8_t g, uint8_t b) {
    const double y = 0.299 * r + 0.587 * g + 0.114 * b;
    return static_cast<uint8_t>(std::lround(std::clamp(y, 0.0, 255.0)));
}

// Grey range of the --seam-on-grey background (see overlay_background).
constexpr int kGreyLow = 80;
constexpr int kGreyHigh = 176;

// Copy `background` into a 3-channel image, optionally desaturating it (and
// compressing its contrast around mid-grey) so the coloured seams stand out. Greyscale (1) and grey+alpha (2) sources use
// channel 0 as the luminance; any alpha channel of the source is dropped.
Image overlay_background(const Image& background, bool greyscale_background) {
    const int w = background.width;
    const int h = background.height;
    const int bc = background.channels;

    Image out;
    out.width = w;
    out.height = h;
    out.channels = 3;
    out.data.resize(static_cast<size_t>(w) * h * 3);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const int src = (y * w + x) * bc;
            uint8_t r, g, b;
            if (bc >= 3) {
                r = background.data[src + 0];
                g = background.data[src + 1];
                b = background.data[src + 2];
            } else {
                r = g = b = background.data[src];
            }
            const int dst = (y * w + x) * 3;
            if (greyscale_background) {
                // Luminance compressed linearly into [kGreyLow, kGreyHigh], a band
                // around mid-grey: the colour maps span from dark (first seams) to
                // light (last seams), so a full-range background would hide the
                // dark seams in its shadows and the light ones in its highlights.
                const uint8_t l = static_cast<uint8_t>(
                    kGreyLow + std::lround(luma601(r, g, b) * (kGreyHigh - kGreyLow) / 255.0));
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
    return out;
}

// Paint the seams over `out`. Seams are numbered 0, 1, ... in removal order
// (vertical first, then horizontal) and seam k is drawn in colour_of(k).
// Each vertical seam holds one column x per row y; each horizontal seam holds
// one row y per column x. Out-of-range coordinates are skipped.
template <class ColourOf>
void paint_seams(Image& out,
                 const std::vector<std::vector<int>>& vertical_seams,
                 const std::vector<std::vector<int>>& horizontal_seams,
                 ColourOf colour_of) {
    const int w = out.width;
    const int h = out.height;
    auto put = [&](int x, int y, const Anchor& rgb) {
        const int dst = (y * w + x) * 3;
        out.data[dst + 0] = rgb[0];
        out.data[dst + 1] = rgb[1];
        out.data[dst + 2] = rgb[2];
    };

    std::size_t k = 0;
    for (const auto& seam : vertical_seams) {
        const Anchor rgb = colour_of(k++);
        const int rows = std::min<int>(static_cast<int>(seam.size()), h);
        for (int y = 0; y < rows; ++y) {
            const int x = seam[y];
            if (x >= 0 && x < w) put(x, y, rgb);
        }
    }
    for (const auto& seam : horizontal_seams) {
        const Anchor rgb = colour_of(k++);
        const int cols = std::min<int>(static_cast<int>(seam.size()), w);
        for (int x = 0; x < cols; ++x) {
            const int y = seam[x];
            if (y >= 0 && y < h) put(x, y, rgb);
        }
    }
}

} // namespace

Colourmap colourmap_from_string(const std::string& name, bool* ok) {
    if (ok) *ok = true;
    if (name == "viridis") return Colourmap::Viridis;
    if (name == "magma") return Colourmap::Magma;
    if (name == "grey" || name == "gray") return Colourmap::Grey;
    if (ok) *ok = false;
    return Colourmap::Viridis;
}

Image render_scalar_field(const std::vector<double>& field, int width, int height,
                          Colourmap cmap, double clip_percentile) {
    Image out;
    out.width = width;
    out.height = height;
    out.channels = 3;
    out.data.resize(static_cast<size_t>(width) * height * 3);

    // Linear normalisation from [min, q] to [0, 1], q being the clip_percentile-th
    // percentile of the field (nearest rank); with 100 this is the plain maximum.
    double lo = 0.0;
    double hi = 0.0;
    if (!field.empty()) {
        lo = *std::min_element(field.begin(), field.end());
        const double p = std::clamp(clip_percentile, 0.0, 100.0);
        if (p >= 100.0) {
            hi = *std::max_element(field.begin(), field.end());
        } else {
            std::vector<double> sorted(field);
            const std::size_t rank = static_cast<std::size_t>(
                std::lround(p / 100.0 * static_cast<double>(sorted.size() - 1)));
            std::nth_element(sorted.begin(), sorted.begin() + rank, sorted.end());
            hi = sorted[rank];
        }
    }
    const double range = (hi > lo) ? (hi - lo) : 1.0;

    for (int i = 0; i < width * height; ++i) {
        const Anchor rgb = colour_at(cmap, (field[i] - lo) / range); // clamps t > 1
        out.data[i * 3 + 0] = rgb[0];
        out.data[i * 3 + 1] = rgb[1];
        out.data[i * 3 + 2] = rgb[2];
    }
    return out;
}

Image render_seam_overlay(const Image& background,
                          const std::vector<std::vector<int>>& vertical_seams,
                          const std::vector<std::vector<int>>& horizontal_seams,
                          const uint8_t seam_rgb[3],
                          bool greyscale_background) {
    Image out = overlay_background(background, greyscale_background);
    const Anchor rgb = {seam_rgb[0], seam_rgb[1], seam_rgb[2]};
    paint_seams(out, vertical_seams, horizontal_seams,
                [&](std::size_t) { return rgb; });
    return out;
}

Image render_seam_overlay_by_order(const Image& background,
                                   const std::vector<std::vector<int>>& vertical_seams,
                                   const std::vector<std::vector<int>>& horizontal_seams,
                                   Colourmap cmap,
                                   bool greyscale_background) {
    Image out = overlay_background(background, greyscale_background);
    const std::size_t total = vertical_seams.size() + horizontal_seams.size();
    const double last = (total > 1) ? static_cast<double>(total - 1) : 1.0;
    paint_seams(out, vertical_seams, horizontal_seams,
                [&](std::size_t k) { return colour_at(cmap, static_cast<double>(k) / last); });
    return out;
}

} // namespace viz
