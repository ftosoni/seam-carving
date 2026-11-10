// An Analysis and Implementation of Seam Carving for Content-Aware Image Resizing
//
// Copyright (c) 2026 Francesco Tosoni
//
// This file implements an algorithm possibly linked to the patent US 7,747,107 B2
// ("Method for retargeting images", S. Avidan and A. Shamir, Mitsubishi Electric
// Research Laboratories). This file is made available for the exclusive aim of
// serving as scientific tool to verify the soundness and completeness of the
// algorithm description. Compilation, execution and redistribution of this file
// may violate patents rights in certain countries. The situation being different
// for every country and changing over time, it is your responsibility to
// determine which patent rights restrictions apply to you before you compile,
// use, modify, or redistribute this file. A patent lawyer is qualified to make
// this determination. If and only if they don't conflict with any patent terms,
// you can benefit from the following license terms attached to this file.
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
// Seam carving operator (implementation).
//
// Content-aware image resizing by repeated removal/insertion of seams: 8-connected
// monotonic pixel paths that contain exactly one pixel per row (or per column).
// Two energy criteria are supported:
//
//   * Backward energy: the gradient-magnitude energy e1 of
//       Avidan & Shamir, "Seam Carving for Content-Aware Image Resizing",
//       ACM SIGGRAPH 2007 (energy functions: section 3.2). The paper's e1 is the
//       L1 gradient; both L1 and L2 were reported as tested. We use the L2
//       gradient by default and offer L1 as an option (computed on BT.601
//       luminance unless disabled).
//       Two further energy functions of the same section are available: e_HoG
//       (gradient divided by the local histogram-of-gradients maximum) and
//       e_Entropy (gradient plus local entropy).
//   * Forward energy: the criterion of
//       Rubinstein, Shamir & Avidan, "Improved Seam Carving for Video
//       Retargeting", ACM SIGGRAPH 2008 (section 5.1, Eq. (2)), which charges
//       each seam step with the energy its removal *introduces* into the result.
//
// The optimal seam is found by dynamic programming (Cormen et al., "Introduction
// to Algorithms"). Enlargement duplicates the first k seams in their removal
// order (2007, section 4.3); a weight mask realises object protection/removal
// (2007, section 4.6) via the optional per-pixel term P(i,j) of the 2008
// recurrence. Section numbers in the comments below refer to these two papers
// and to the companion IPOL article.
// -----------------------------------------------------------------------------

#include "seam_carving.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <limits>
#include <iostream>

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace {
    // Helper to get RGB values of a pixel with periodic (wrap-around) boundaries,
    // so out-of-range neighbours map back into the image (manuscript, section Energy:
    // "out-of-range coordinates wrap around periodically").
    inline void get_pixel_rgb(const std::vector<uint8_t>& data, int x, int y, int width, int height, int channels, double rgb[3]) {
        // Wrap around boundaries
        int px = (x + width) % width;
        int py = (y + height) % height;
        int idx = (py * width + px) * channels;
        if (channels >= 3) {
            rgb[0] = data[idx];
            rgb[1] = data[idx + 1];
            rgb[2] = data[idx + 2];
        } else {
            // Greyscale (1 channel) or grey+alpha (2): treat channel 0 as the
            // luminance and replicate it, so every energy formula keeps working.
            rgb[0] = rgb[1] = rgb[2] = data[idx];
        }
    }

    // Absolute intensity difference between two pixels, used by the forward-energy
    // costs (Rubinstein et al. 2008, section 5.1): luminance difference in luma mode,
    // or the L2 (or, with use_l1, L1) norm of the RGB difference otherwise.
    // Boundaries wrap (see above).
    inline double get_pixel_diff(const std::vector<uint8_t>& data, int x1, int y1, int x2, int y2, int width, int height, int channels, bool use_luminance, bool use_l1) {
        double rgb1[3], rgb2[3];
        get_pixel_rgb(data, x1, y1, width, height, channels, rgb1);
        get_pixel_rgb(data, x2, y2, width, height, channels, rgb2);
        if (use_luminance) {
            // Convert to luminance (BT.601 luma formula)
            double l1 = 0.299 * rgb1[0] + 0.587 * rgb1[1] + 0.114 * rgb1[2];
            double l2 = 0.299 * rgb2[0] + 0.587 * rgb2[1] + 0.114 * rgb2[2];
            return std::abs(l1 - l2);
        } else {
            double dr = rgb1[0] - rgb2[0];
            double dg = rgb1[1] - rgb2[1];
            double db = rgb1[2] - rgb2[2];
            if (use_l1) {
                return std::abs(dr) + std::abs(dg) + std::abs(db);
            }
            return std::sqrt(dr * dr + dg * dg + db * db);
        }
    }

    // Periodic index: maps any integer a to [0, n), consistently with the
    // wrap-around boundaries used everywhere else (also for |a| > n, which occurs
    // when a local window is wider than the image).
    inline int wrap(int a, int n) {
        int r = a % n;
        return (r < 0) ? r + n : r;
    }

    // BT.601 luminance plane of the image (size width*height, row-major).
    std::vector<double> luminance_plane(const std::vector<uint8_t>& data, int width, int height, int channels) {
        std::vector<double> lum(static_cast<size_t>(width) * height);
        double rgb[3];
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                get_pixel_rgb(data, x, y, width, height, channels, rgb);
                lum[y * width + x] = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2];
            }
        }
        return lum;
    }

    // Sweep a (2r+1)x(2r+1) window along row y, left to right, with periodic
    // boundaries. Instead of rebuilding the window statistic at every pixel (cost
    // (2r+1)^2), the window is updated incrementally: moving from x-1 to x removes
    // the leftmost column and adds a new rightmost one (cost 2(2r+1)).
    //   update(px, py, sign): add (sign = +1) or remove (sign = -1) pixel (px, py);
    //   emit(x):              called once the window centred on (x, y) is complete.
    template <class Update, class Emit>
    void slide_window_along_row(int y, int r, int width, int height, Update update, Emit emit) {
        auto column = [&](int cx, int sign) {
            const int px = wrap(cx, width);
            for (int dy = -r; dy <= r; ++dy) {
                update(px, wrap(y + dy, height), sign);
            }
        };
        for (int dx = -r; dx <= r; ++dx) column(dx, +1);
        emit(0);
        for (int x = 1; x < width; ++x) {
            column(x - r - 1, -1);
            column(x + r, +1);
            emit(x);
        }
    }

    // Parameters of the energy functions of Avidan & Shamir 2007, section 3.2.
    constexpr int kHogRadius = 5;       // 11x11 window
    constexpr int kHogBins = 8;         // 8-bin orientation histogram
    constexpr double kHogEpsilon = 1.0; // guards the division in perfectly flat windows
    constexpr int kEntropyRadius = 4;   // 9x9 window
    // The paper adds the entropy (in bits, at most log2(81) ~ 6.3 here) to e1
    // without stating the intensity scale. We take intensities in [0, 1], the
    // usual convention: our gradient is computed on [0, 255], so the entropy is
    // multiplied by 255, i.e. e_Entropy = 255 * (e / 255 + H).
    constexpr double kEntropyWeight = 255.0;
}

SeamCarving::SeamCarving(const Image& image)
    : width_(image.width), height_(image.height), channels_(image.channels), data_(image.data) {
    if (width_ <= 0 || height_ <= 0) {
        throw std::runtime_error("Image dimensions must be positive.");
    }
    if (channels_ < 1) {
        throw std::runtime_error("Image must have at least one channel.");
    }
}

Image SeamCarving::get_image() const {
    return Image{width_, height_, channels_, data_};
}

void SeamCarving::set_mask_weights(const std::vector<double>& mask_weights) {
    if (!mask_weights.empty() && mask_weights.size() != static_cast<size_t>(width_ * height_)) {
        throw std::runtime_error("Mask weights dimensions do not match the image dimensions.");
    }
    mask_weights_ = mask_weights;
}

void SeamCarving::set_forward_energy(bool enable) {
    use_forward_energy_ = enable;
}

void SeamCarving::set_luminance_energy(bool enable) {
    use_luminance_ = enable;
}

void SeamCarving::set_energy_function(EnergyFunction function) {
    energy_function_ = function;
}

void SeamCarving::set_gradient_norm(GradientNorm norm) {
    gradient_norm_ = norm;
}

void SeamCarving::set_height_first(bool enable) {
    height_first_ = enable;
}

void SeamCarving::set_verbose(bool enable) {
    verbose_ = enable;
}

void SeamCarving::set_parallel_dp(bool enable) {
    use_parallel_dp_ = enable;
}

void SeamCarving::resize(int target_width, int target_height, int num_threads) {
    if (target_width <= 0 || target_height <= 0) {
        throw std::runtime_error("Target dimensions must be positive.");
    }

    // Schedule: all width changes first, then all height changes (or the reverse
    // with set_height_first). This is the simple alternative to the optimal
    // interleaving order of horizontal/vertical removals (transport map, Avidan &
    // Shamir 2007, section 4.2), which we do not implement; it is faster and uses
    // less memory at the price of not guaranteeing the globally optimal order when
    // both dimensions change (IPOL manuscript, "Image Reduction" section).
    auto resize_width = [&]() {
        if (target_width < width_) {
            carve_width(target_width, num_threads);
        } else if (target_width > width_) {
            insert_width(target_width, num_threads);
        }
    };
    auto resize_height = [&]() {
        if (target_height < height_) {
            carve_height(target_height, num_threads);
        } else if (target_height > height_) {
            insert_height(target_height, num_threads);
        }
    };

    if (height_first_) {
        resize_height();
        resize_width();
    } else {
        resize_width();
        resize_height();
    }
}

// Implements Algorithm 1 (width reduction by seam removal), lines 1-7, of the IPOL
// article: line 2 is compute_energy_map, lines 3-5 are find_vertical_seam and
// line 6 is remove_vertical_seam.
void SeamCarving::carve_width(int target_width, int num_threads) {
    int total_seams = width_ - target_width;
    int carved = 0;
    if (verbose_ && total_seams > 0) {
        std::cout << "Carving " << total_seams << " width seam(s)..." << std::endl;
    }
    while (width_ > target_width) {
        std::vector<double> energy = compute_energy_map(num_threads);
        std::vector<int> seam = find_vertical_seam(energy, num_threads);
        remove_vertical_seam(seam, num_threads);
        ++carved;
        if (verbose_ && (carved % 10 == 0 || width_ == target_width)) {
            std::cout << "\rProgress: " << carved << "/" << total_seams 
                      << " (" << (carved * 100 / total_seams) << "%)" << std::flush;
        }
    }
    if (verbose_ && total_seams > 0) {
        std::cout << std::endl;
    }
}

// Height reduction reuses the vertical-seam machinery on the transposed image, so a
// single routine serves both directions (IPOL manuscript, section "Image Reduction"):
// Algorithm 1 applied to the transpose.
void SeamCarving::carve_height(int target_height, int num_threads) {
    if (height_ <= target_height) return;

    int total_seams = height_ - target_height;
    if (verbose_ && total_seams > 0) {
        std::cout << "Carving " << total_seams << " height seam(s)..." << std::endl;
    }

    // Transpose to make horizontal seams vertical
    transpose();

    // Carve width of transposed image (original height)
    carve_width(target_height, num_threads);

    // Transpose back to original orientation
    transpose();
}

// Backward energy: the dual-gradient ("gradient magnitude") energy e1 of
// Avidan & Shamir 2007 (section 3.2). Each pixel's energy is the magnitude of the
// image gradient, estimated by central differences I(.+1) - I(.-1). We use the L2
// gradient (the paper's e1 is the L1 gradient; both were reported as tested). Only
// relative energy matters for the minimisation, so the missing 1/2 factor of the
// central difference is irrelevant (IPOL manuscript, Remark in section "Energy").
// Implements line 2 of Algorithm 1 and Equations (3)-(5) of the IPOL article
// (Equations (6) and (7) for the alternative energy functions, see below).
std::vector<double> SeamCarving::compute_energy_map(int num_threads) {
    std::vector<double> energy(width_ * height_);

    // Parallelise the energy computation across rows using OpenMP.
    // Each row is independent of the others, so there is no synchronisation here
    // and the result does not depend on the thread count.
    #ifdef USE_OPENMP
    #pragma omp parallel for num_threads(num_threads) schedule(static)
    #endif
    for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
            double r_left[3], r_right[3], r_up[3], r_down[3];

            // Get RGB values of neighbours. Wrapping boundaries are used here:
            // e.g. x - 1 wraps to width_ - 1, and x + 1 wraps to 0.
            get_pixel_rgb(data_, x - 1, y, width_, height_, channels_, r_left); // to the left
            get_pixel_rgb(data_, x + 1, y, width_, height_, channels_, r_right); // to the right
            get_pixel_rgb(data_, x, y - 1, width_, height_, channels_, r_up); // above
            get_pixel_rgb(data_, x, y + 1, width_, height_, channels_, r_down); // below

            // Compute gradients. If use_luminance_ is enabled, we use perceptual weights (BT.601)
            // to compute luminance gradients (approximating greyscale gradients).
            // Otherwise, we calculate Euclidean distance (L2 norm) in RGB space.
            double rx = r_right[0] - r_left[0];
            double gx = r_right[1] - r_left[1];
            double bx = r_right[2] - r_left[2];

            double ry = r_down[0] - r_up[0];
            double gy = r_down[1] - r_up[1];
            double by = r_down[2] - r_up[2];

            // L2 (default): Equation (3) of the IPOL article; L1: the e1 of the
            // original paper, Equation (2). In RGB mode the L1 norm sums the absolute
            // differences of all channels and both directions.
            double val = 0.0;
            const bool l1 = (gradient_norm_ == GradientNorm::L1);
            if (use_luminance_) {
                double gx_luma = 0.299 * rx + 0.587 * gx + 0.114 * bx;
                double gy_luma = 0.299 * ry + 0.587 * gy + 0.114 * by;
                val = l1 ? std::abs(gx_luma) + std::abs(gy_luma)
                         : std::sqrt(gx_luma * gx_luma + gy_luma * gy_luma);
            } else if (l1) {
                val = std::abs(rx) + std::abs(gx) + std::abs(bx)
                    + std::abs(ry) + std::abs(gy) + std::abs(by);
            } else {
                double delta_x_sq = rx * rx + gx * gx + bx * bx;
                double delta_y_sq = ry * ry + gy * gy + by * by;
                val = std::sqrt(delta_x_sq + delta_y_sq);
            }

            // Save the computed energy magnitude
            energy[y * width_ + x] = val;
        }
    }

    // Optional local modulation of the gradient magnitude (section 3.2).
    if (energy_function_ == EnergyFunction::HoG) {
        apply_hog(energy, num_threads);
    } else if (energy_function_ == EnergyFunction::Entropy) {
        add_entropy(energy, num_threads);
    }

    return energy;
}

// e_HoG = e / max(HoG) (Avidan & Shamir 2007, section 3.2). HoG(i,j) is an 8-bin
// histogram of oriented gradients over the 11x11 window centred on the pixel. As
// in Dalal & Triggs (2005), each pixel votes in the bin of its unsigned gradient
// orientation in [0, pi) with a weight equal to its gradient magnitude; the
// gradient is the central difference of the luminance, as for e. Dividing by the
// strongest orientation of the neighbourhood makes pixels near strong edges cheap
// (attracting seams to them), while the numerator keeps the seam from crossing the
// edge, so that seams tend to run parallel to it. Implements Equation (6) of the
// IPOL article.
void SeamCarving::apply_hog(std::vector<double>& energy, int num_threads) const {
    const std::vector<double> lum = luminance_plane(data_, width_, height_, channels_);

    // Per-pixel gradient magnitude and orientation bin.
    std::vector<double> magnitude(lum.size());
    std::vector<int> bin(lum.size());
    const double kPi = std::acos(-1.0);
    #ifdef USE_OPENMP
    #pragma omp parallel for num_threads(num_threads) schedule(static)
    #endif
    for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
            const double gx = lum[y * width_ + wrap(x + 1, width_)] - lum[y * width_ + wrap(x - 1, width_)];
            const double gy = lum[wrap(y + 1, height_) * width_ + x] - lum[wrap(y - 1, height_) * width_ + x];
            // Unsigned orientation in [0, pi): opposite gradients (theta and theta + pi)
            // describe the same edge direction and share a bin.
            double theta = std::atan2(gy, gx);       // (-pi, pi]
            if (theta < 0.0) theta += kPi;
            if (theta >= kPi) theta -= kPi;
            int b = static_cast<int>(theta / kPi * kHogBins);
            magnitude[y * width_ + x] = (gradient_norm_ == GradientNorm::L1)
                                            ? std::abs(gx) + std::abs(gy)
                                            : std::sqrt(gx * gx + gy * gy);
            bin[y * width_ + x] = std::min(b, kHogBins - 1);  // guards against rounding
        }
    }

    // Sliding-window histogram along each row; rows are independent.
    #ifdef USE_OPENMP
    #pragma omp parallel for num_threads(num_threads) schedule(static)
    #endif
    for (int y = 0; y < height_; ++y) {
        double hist[kHogBins] = {0.0};
        slide_window_along_row(y, kHogRadius, width_, height_,
            [&](int px, int py, int sign) {
                const int i = py * width_ + px;
                hist[bin[i]] += sign * magnitude[i];
            },
            [&](int x) {
                const double hog_max = *std::max_element(hist, hist + kHogBins);
                energy[y * width_ + x] /= std::max(hog_max, kHogEpsilon);
            });
    }
}

// e_Entropy = e + H (Avidan & Shamir 2007, section 3.2), H being the Shannon
// entropy (in bits) of the 8-bit luminance values in the 9x9 window centred on
// the pixel: H = log2(N) - (1/N) sum_v c_v log2 c_v, with c_v the number of window
// pixels of value v and N = 81. Textured regions have high entropy and are
// protected; uniform ones have zero entropy. The sum is updated incrementally as
// the window slides, using a table of c log2 c. See kEntropyWeight for the scale.
// Implements Equation (7) of the IPOL article.
void SeamCarving::add_entropy(std::vector<double>& energy, int num_threads) const {
    const std::vector<double> lum = luminance_plane(data_, width_, height_, channels_);
    std::vector<uint8_t> level(lum.size());
    for (size_t i = 0; i < lum.size(); ++i) {
        level[i] = static_cast<uint8_t>(std::lround(std::clamp(lum[i], 0.0, 255.0)));
    }

    const int n = (2 * kEntropyRadius + 1) * (2 * kEntropyRadius + 1);
    std::vector<double> c_log_c(n + 1, 0.0);  // c * log2(c), with 0 log 0 = 0
    for (int c = 1; c <= n; ++c) c_log_c[c] = c * std::log2(static_cast<double>(c));
    const double log2_n = std::log2(static_cast<double>(n));

    #ifdef USE_OPENMP
    #pragma omp parallel for num_threads(num_threads) schedule(static)
    #endif
    for (int y = 0; y < height_; ++y) {
        int count[256] = {0};
        double sum_c_log_c = 0.0;
        slide_window_along_row(y, kEntropyRadius, width_, height_,
            [&](int px, int py, int sign) {
                int& c = count[level[py * width_ + px]];
                sum_c_log_c -= c_log_c[c];
                c += sign;
                sum_c_log_c += c_log_c[c];
            },
            [&](int x) {
                const double entropy = std::max(0.0, log2_n - sum_c_log_c / n);
                energy[y * width_ + x] += kEntropyWeight * entropy;
            });
    }
}

// Optimal seam by dynamic programming (Cormen et al.). We build the cumulative cost
// map M top-down, then backtrack the minimum of the last row to recover the seam.
// The optional mask term is the per-pixel weight P(i,j) of Rubinstein et al. 2008,
// Eq. (2): it is added to M and realises object protection (large +) / removal
// (large -) as described in Avidan & Shamir 2007, section 4.6 (IPOL manuscript, "Masking").
// Implements lines 3-5 of Algorithm 1 of the IPOL article: the cumulative cost map of
// Equation (9) (backward) or Equation (11) (forward, with the step costs of
// Equation (10)), its minimum in the last row, and the backtracking.
std::vector<int> SeamCarving::find_vertical_seam(const std::vector<double>& energy, int num_threads) {
    // M is the cumulative cost matrix of size width_ * height_
    std::vector<double> M(width_ * height_);

    // First row of M. Backward energy seeds it with the pixel energy e(0,x); forward
    // energy seeds it with P(0,x) only (its step costs are derived per-row below).
    // The mask weight P is added in both cases.
    for (int x = 0; x < width_; ++x) {
        double mask_w = mask_weights_.empty() ? 0.0 : mask_weights_[x];
        M[x] = use_forward_energy_ ? mask_w : (energy[x] + mask_w);
    }

    // Dynamic programming: each row y depends only on row y - 1, so the recurrence is
    // inherently serial in y. The inner loop over x is independent and may be run in
    // parallel (use_parallel_dp_), but each row then needs a thread barrier; that
    // barrier costs more than a row's arithmetic, so the serial scan is the default
    // and is faster even on large images (IPOL manuscript, "Parallelisation" section).
    for (int y = 1; y < height_; ++y) {
        #ifdef USE_OPENMP
        #pragma omp parallel for num_threads(num_threads) schedule(static) if(use_parallel_dp_)
        #endif
        for (int x = 0; x < width_; ++x) {
            double mask_w = mask_weights_.empty() ? 0.0 : mask_weights_[y * width_ + x];

            if (use_forward_energy_) {
                // Forward energy (Rubinstein et al. 2008, section 5.1, Eq. (2)).
                // Instead of the current pixel energy, each step is charged the energy
                // *introduced* when the seam is removed and formerly non-adjacent
                // neighbours become adjacent. The three step costs (their C_U/C_L/C_R)
                // measure the new pixel edges created for a vertical/left/right step:
                // c_u: new horizontal edge between the left and right neighbours:
                //      c_u = |I(x+1, y) - I(x-1, y)|                         (their C_U)
                // c_l: c_u + new vertical edge when stepping from the left diagonal:
                //      c_l = c_u + |I(x, y-1) - I(x-1, y)|                   (their C_L)
                // c_r: c_u + new vertical edge when stepping from the right diagonal:
                //      c_r = c_u + |I(x, y-1) - I(x+1, y)|                   (their C_R)
                double c_u = get_pixel_diff(data_, x + 1, y, x - 1, y, width_, height_, channels_, use_luminance_, gradient_norm_ == GradientNorm::L1);
                double c_l = c_u + get_pixel_diff(data_, x, y - 1, x - 1, y, width_, height_, channels_, use_luminance_, gradient_norm_ == GradientNorm::L1);
                double c_r = c_u + get_pixel_diff(data_, x, y - 1, x + 1, y, width_, height_, channels_, use_luminance_, gradient_norm_ == GradientNorm::L1);

                // Recurrence: M(y, x) = min( M(y-1, x-1) + c_l, M(y-1, x) + c_u, M(y-1, x+1) + c_r )
                double val_l = (x > 0) ? (M[(y - 1) * width_ + (x - 1)] + c_l) : 1e30;
                double val_u = M[(y - 1) * width_ + x] + c_u;
                double val_r = (x < width_ - 1) ? (M[(y - 1) * width_ + (x + 1)] + c_r) : 1e30;

                M[y * width_ + x] = std::min({val_l, val_u, val_r}) + mask_w;
            } else {
                // Classic backward energy (Avidan & Shamir 2007).
                // Recurrence: M(y, x) = e(y, x) + min( M(y-1, x-1), M(y-1, x), M(y-1, x+1) )
                double min_prev = M[(y - 1) * width_ + x];
                if (x > 0) {
                    min_prev = std::min(min_prev, M[(y - 1) * width_ + (x - 1)]);
                }
                if (x < width_ - 1) {
                    min_prev = std::min(min_prev, M[(y - 1) * width_ + (x + 1)]);
                }
                M[y * width_ + x] = energy[y * width_ + x] + min_prev + mask_w;
            }
        }
    }

    // Backtrack from the bottom row to the top to recover the minimum-cost seam: start
    // at the smallest entry of the last row, then at each step move to the predecessor
    // (left/up/right) that realised the minimum used to fill M.
    std::vector<int> seam(height_);

    // Find the starting column index (min_x) of the minimum energy in the last row (height_ - 1)
    double min_energy = M[(height_ - 1) * width_];
    int min_x = 0;
    for (int x = 1; x < width_; ++x) {
        double val = M[(height_ - 1) * width_ + x];
        if (val < min_energy) {
            min_energy = val;
            min_x = x;
        }
    }
    seam[height_ - 1] = min_x;

    // Backtrack upwards row-by-row from y = height_ - 2 to y = 0
    for (int y = height_ - 2; y >= 0; --y) {
        int prev_x = seam[y + 1];
        int best_x = prev_x;
        double best_val = M[y * width_ + prev_x];

        if (use_forward_energy_) {
            // Recompute the forward step costs at (y+1, prev_x) so the descent uses
            // exactly the same costs that built M (Rubinstein et al. 2008, Eq. (2)).
            double c_u = get_pixel_diff(data_, prev_x + 1, y + 1, prev_x - 1, y + 1, width_, height_, channels_, use_luminance_, gradient_norm_ == GradientNorm::L1);
            double c_l = c_u + get_pixel_diff(data_, prev_x, y, prev_x - 1, y + 1, width_, height_, channels_, use_luminance_, gradient_norm_ == GradientNorm::L1);
            double c_r = c_u + get_pixel_diff(data_, prev_x, y, prev_x + 1, y + 1, width_, height_, channels_, use_luminance_, gradient_norm_ == GradientNorm::L1);

            // The straight (vertical) predecessor also pays the c_u edge cost in the
            // forward recurrence, so add it to the running best before comparing it
            // against the diagonal candidates. Without this, the up step is undercounted
            // by c_u and an occasionally suboptimal predecessor would be chosen.
            best_val += c_u;

            if (prev_x > 0) {
                double val = M[y * width_ + (prev_x - 1)] + c_l;
                if (val < best_val) {
                    best_val = val;
                    best_x = prev_x - 1;
                }
            }
            if (prev_x < width_ - 1) {
                double val = M[y * width_ + (prev_x + 1)] + c_r;
                if (val < best_val) {
                    best_val = val;
                    best_x = prev_x + 1;
                }
            }
        } else {
            // Classic backward backtracking (check left, centre, right neighbours in row y)
            if (prev_x > 0) {
                double val = M[y * width_ + (prev_x - 1)];
                if (val < best_val) {
                    best_val = val;
                    best_x = prev_x - 1;
                }
            }
            if (prev_x < width_ - 1) {
                double val = M[y * width_ + (prev_x + 1)];
                if (val < best_val) {
                    best_val = val;
                    best_x = prev_x + 1;
                }
            }
        }
        seam[y] = best_x;
    }

    return seam;
}

// Rebuild the image one column narrower by dropping the seam pixel in each row (and the
// matching mask entry, so the mask stays aligned with the image across iterations).
// Rows are independent, so this is parallelised and is O(n*m) per seam.
// Implements line 6 of Algorithm 1 of the IPOL article.
void SeamCarving::remove_vertical_seam(const std::vector<int>& seam, int num_threads) {
    int new_width = width_ - 1;
    std::vector<uint8_t> new_data(new_width * height_ * channels_);
    
    std::vector<double> new_mask_weights;
    if (!mask_weights_.empty()) {
        new_mask_weights.resize(new_width * height_);
    }

    #ifdef USE_OPENMP
    #pragma omp parallel for num_threads(num_threads) schedule(static)
    #endif
    for (int y = 0; y < height_; ++y) {
        int seam_x = seam[y];
        int src_row_start = y * width_ * channels_;
        int dst_row_start = y * new_width * channels_;

        // Copy everything before seam_x
        if (seam_x > 0) {
            std::copy(data_.begin() + src_row_start,
                      data_.begin() + src_row_start + seam_x * channels_,
                      new_data.begin() + dst_row_start);
        }

        // Copy everything after seam_x
        if (seam_x < width_ - 1) {
            std::copy(data_.begin() + src_row_start + (seam_x + 1) * channels_,
                      data_.begin() + src_row_start + width_ * channels_,
                      new_data.begin() + dst_row_start + seam_x * channels_);
        }

        // Handle mask weights if present
        if (!mask_weights_.empty()) {
            int src_mask_start = y * width_;
            int dst_mask_start = y * new_width;
            if (seam_x > 0) {
                std::copy(mask_weights_.begin() + src_mask_start,
                          mask_weights_.begin() + src_mask_start + seam_x,
                          new_mask_weights.begin() + dst_mask_start);
            }
            if (seam_x < width_ - 1) {
                std::copy(mask_weights_.begin() + src_mask_start + seam_x + 1,
                          mask_weights_.begin() + src_mask_start + width_,
                          new_mask_weights.begin() + dst_mask_start + seam_x);
            }
        }
    }

    width_ = new_width;
    data_ = std::move(new_data);
    if (!mask_weights_.empty()) {
        mask_weights_ = std::move(new_mask_weights);
    }
}

// Transpose the image so that horizontal seams become vertical and the existing
// vertical-seam routines can handle them. Beyond reusing one code path, this keeps
// the inner loops cache-friendly: with row-major storage a vertical seam touches one
// contiguous row at a time, whereas carving horizontal seams in place would stride
// across rows (column-wise) and thrash the cache. Paying for two transposes is
// cheaper than the repeated strided accesses over many seam iterations.
void SeamCarving::transpose() {
    std::vector<uint8_t> transposed_data(width_ * height_ * channels_);
    std::vector<double> transposed_mask_weights;
    if (!mask_weights_.empty()) {
        transposed_mask_weights.resize(width_ * height_);
    }

    #ifdef USE_OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
    #endif
    for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
            int src_idx = (y * width_ + x) * channels_;
            int dst_idx = (x * height_ + y) * channels_;
            for (int c = 0; c < channels_; ++c) {
                transposed_data[dst_idx + c] = data_[src_idx + c];
            }
            if (!mask_weights_.empty()) {
                transposed_mask_weights[x * height_ + y] = mask_weights_[y * width_ + x];
            }
        }
    }

    std::swap(width_, height_);
    data_ = std::move(transposed_data);
    if (!mask_weights_.empty()) {
        mask_weights_ = std::move(transposed_mask_weights);
    }
}

// Implements Algorithm 2 (one enlargement pass), lines 1-8, of the IPOL article,
// repeated in passes of at most 50% of the current width (section "Image
// Enlargement"): step 1 below is line 1, step 2 lines 2-5, steps 3-4 lines 6-8.
void SeamCarving::insert_width(int target_width, int num_threads) {
    if (width_ >= target_width) return;

    if (verbose_) {
        std::cout << "Enlarging width from " << width_ << " to " << target_width << "..." << std::endl;
    }

    // Seam insertion (Avidan & Shamir 2007, section 4.3, Fig. 8): find the first k
    // seams that *would* be removed, in order, then duplicate all of them in the
    // original image. Naively re-inserting the single optimal seam k times would pick
    // the same seam repeatedly and stretch one path; computing the k seams on a
    // shrinking working copy avoids this. Inserting a number of seams comparable to
    // the width is equivalent to uniform scaling, so a large expansion is split into
    // passes that each insert at most 50% of the current width.
    while (width_ < target_width) {
        int K = target_width - width_;
        int k = std::min(K, std::max(1, width_ / 2)); // insert at most 50% of the current width

        // 1. Create a copy of the carver and an index mapping grid.
        // The index_map tracks the original x-coordinate of every pixel as we
        // remove seams from the temporary copy.
        SeamCarving temp_carver(*this);
        std::vector<int> index_map(width_ * height_);
        for (int y = 0; y < height_; ++y) {
            for (int x = 0; x < width_; ++x) {
                index_map[y * width_ + x] = x;
            }
        }

        // 2. Find k seams iteratively on the temporary copy
        std::vector<std::vector<int>> seams_to_insert(k, std::vector<int>(height_));
        for (int j = 0; j < k; ++j) {
            std::vector<double> energy = temp_carver.compute_energy_map(num_threads);
            std::vector<int> seam = temp_carver.find_vertical_seam(energy, num_threads);

            // Translate current seam coordinates to the original image coordinates using index_map
            for (int y = 0; y < height_; ++y) {
                seams_to_insert[j][y] = index_map[y * temp_carver.width_ + seam[y]];
            }

            // Remove the seam from our index tracking map so that subsequent seam finds
            // align with the remaining pixels in the temp_carver.
            std::vector<int> new_index_map((temp_carver.width_ - 1) * height_);
            for (int y = 0; y < height_; ++y) {
                int seam_x = seam[y];
                int src_row_start = y * temp_carver.width_;
                int dst_row_start = y * (temp_carver.width_ - 1);
                if (seam_x > 0) {
                    std::copy(index_map.begin() + src_row_start,
                              index_map.begin() + src_row_start + seam_x,
                              new_index_map.begin() + dst_row_start);
                }
                if (seam_x < temp_carver.width_ - 1) {
                    std::copy(index_map.begin() + src_row_start + seam_x + 1,
                              index_map.begin() + src_row_start + temp_carver.width_,
                              new_index_map.begin() + dst_row_start + seam_x);
                }
            }
            index_map = std::move(new_index_map);

            // Remove seam from temp_carver to find the next optimal seam
            temp_carver.remove_vertical_seam(seam, num_threads);
        }

        // 3. Collect and sort the column indices to duplicate for each row.
        // Sorting ensures we can insert them sequentially from left to right.
        std::vector<std::vector<int>> row_cols_to_duplicate(height_, std::vector<int>(k));
        for (int y = 0; y < height_; ++y) {
            for (int j = 0; j < k; ++j) {
                row_cols_to_duplicate[y][j] = seams_to_insert[j][y];
            }
            std::sort(row_cols_to_duplicate[y].begin(), row_cols_to_duplicate[y].end());
        }

        // 4. Construct the new enlarged image data and updated mask weights
        int new_width = width_ + k;
        std::vector<uint8_t> new_data(new_width * height_ * channels_);
        std::vector<double> new_mask_weights;
        if (!mask_weights_.empty()) {
            new_mask_weights.resize(new_width * height_);
        }

        #ifdef USE_OPENMP
        #pragma omp parallel for num_threads(num_threads) schedule(static)
        #endif
        for (int y = 0; y < height_; ++y) {
            // cols contains the sorted column indices (relative to the original width)
            // that we need to duplicate for the current row y.
            const auto& cols = row_cols_to_duplicate[y];
            
            int src_col = 0;        // Index scanning columns of the original image (0 to width_ - 1)
            int dst_col = 0;        // Index writing columns of the new enlarged image (0 to new_width - 1)
            int cols_inserted = 0;  // Count of seams we have duplicated so far in this row

            while (src_col < width_) {
                // Step 4.1: Always copy the original pixel from the source to the destination.
                int src_pixel_idx = (y * width_ + src_col) * channels_;
                int dst_pixel_idx = (y * new_width + dst_col) * channels_;
                for (int c = 0; c < channels_; ++c) {
                    new_data[dst_pixel_idx + c] = data_[src_pixel_idx + c];
                }
                if (!mask_weights_.empty()) {
                    new_mask_weights[y * new_width + dst_col] = mask_weights_[y * width_ + src_col];
                }
                ++dst_col; // Move the destination pointer forward

                // Step 4.2: Check if the current source column (src_col) matches the next seam column 
                // we want to duplicate (cols[cols_inserted]).
                if (cols_inserted < k && cols[cols_inserted] == src_col) {
                    // Determine which neighbour to average with. We prefer the right neighbour (src_col + 1),
                    // but if we are at the right edge, we average with the left neighbour (src_col - 1).
                    // This is the one-sided rule of Avidan & Shamir 2007, section 5 (multi-size
                    // images); their section 4.3 averages with both neighbours instead and also
                    // modifies the seam pixel (IPOL manuscript, section "Image Enlargement").
                    int next_col = (src_col + 1 < width_) ? (src_col + 1) : (src_col - 1);
                    if (next_col < 0) next_col = src_col; // Fallback for 1-pixel wide images

                    int nbr_pixel_idx = (y * width_ + next_col) * channels_;
                    int dup_pixel_idx = (y * new_width + dst_col) * channels_;

                    // Set the duplicated pixel to the average of the current pixel and its neighbour.
                    // This smooths out the transition and avoids stark vertical seam replicas.
                    for (int c = 0; c < channels_; ++c) {
                        new_data[dup_pixel_idx + c] = static_cast<uint8_t>((static_cast<int>(data_[src_pixel_idx + c]) + static_cast<int>(data_[nbr_pixel_idx + c])) / 2);
                    }
                    if (!mask_weights_.empty()) {
                        new_mask_weights[y * new_width + dst_col] = (mask_weights_[y * width_ + src_col] + mask_weights_[y * width_ + next_col]) / 2.0;
                    }
                    
                    ++dst_col;         // Move the destination pointer forward again to accommodate the new pixel
                    ++cols_inserted;   // Advance to the next column to duplicate
                }
                ++src_col; // Move to the next source column
            }
        }

        width_ = new_width;
        data_ = std::move(new_data);
        if (!mask_weights_.empty()) {
            mask_weights_ = std::move(new_mask_weights);
        }
    }
}

// --- Visualisation support ---------------------------------------------------
// Thin wrapper exposing the backward energy map for figures (see visualise.h).
// The map honours use_luminance_ but ignores use_forward_energy_ by design: the
// energy figure always shows the backward dual-gradient magnitude.
std::vector<double> SeamCarving::backward_energy_map(int num_threads) {
    return compute_energy_map(num_threads);
}

// Replay width reduction on an internal copy to collect the first `count` seams
// that would be removed, translating each back to the input's coordinate system
// via an index map (the same back-mapping ordered seam insertion uses). The
// original object is untouched, so this can be called before carving the image.
std::vector<std::vector<int>> SeamCarving::seams_to_remove(int count, int num_threads) {
    std::vector<std::vector<int>> seams;
    if (count <= 0) {
        return seams;
    }
    count = std::min(count, width_ - 1); // cannot remove the last remaining column
    seams.reserve(count);

    // Work on a copy: the original image must remain intact for the actual carve.
    SeamCarving work(*this);

    // index_map[y * w + x] holds the column that pixel (x, y) of the shrinking
    // copy occupied in the original image, so recorded seams use input coordinates.
    std::vector<int> index_map(static_cast<size_t>(work.width_) * work.height_);
    for (int y = 0; y < work.height_; ++y) {
        for (int x = 0; x < work.width_; ++x) {
            index_map[y * work.width_ + x] = x;
        }
    }

    for (int s = 0; s < count; ++s) {
        // The energy map is unused in forward mode (its costs are derived in the
        // DP), but computing it unconditionally keeps this in step with the rest
        // of the code and costs only the viz path.
        std::vector<double> energy = work.compute_energy_map(num_threads);
        std::vector<int> seam = work.find_vertical_seam(energy, num_threads);

        // Record the seam in original coordinates.
        std::vector<int> original(work.height_);
        for (int y = 0; y < work.height_; ++y) {
            original[y] = index_map[y * work.width_ + seam[y]];
        }
        seams.push_back(std::move(original));

        // Drop the seam columns from the index map, mirroring remove_vertical_seam.
        std::vector<int> new_index_map(static_cast<size_t>(work.width_ - 1) * work.height_);
        for (int y = 0; y < work.height_; ++y) {
            int seam_x = seam[y];
            int src_row_start = y * work.width_;
            int dst_row_start = y * (work.width_ - 1);
            if (seam_x > 0) {
                std::copy(index_map.begin() + src_row_start,
                          index_map.begin() + src_row_start + seam_x,
                          new_index_map.begin() + dst_row_start);
            }
            if (seam_x < work.width_ - 1) {
                std::copy(index_map.begin() + src_row_start + seam_x + 1,
                          index_map.begin() + src_row_start + work.width_,
                          new_index_map.begin() + dst_row_start + seam_x);
            }
        }
        index_map = std::move(new_index_map);

        work.remove_vertical_seam(seam, num_threads);
    }

    return seams;
}

void SeamCarving::insert_height(int target_height, int num_threads) {
    if (height_ >= target_height) return;

    if (verbose_) {
        std::cout << "Enlarging height from " << height_ << " to " << target_height << "..." << std::endl;
    }

    // Transpose to make horizontal seams vertical
    transpose();

    // Insert width of transposed image (original height)
    insert_width(target_height, num_threads);

    // Transpose back to original orientation
    transpose();
}

std::vector<std::vector<int>> SeamCarving::seams_to_remove_horizontal(int count, int num_threads) {
    SeamCarving work(*this);
    work.transpose();
    return work.seams_to_remove(count, num_threads);
}
