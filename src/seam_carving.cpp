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
//       gradient by default (computed on BT.601 luminance unless disabled).
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
            // Grayscale (1 channel) or gray+alpha (2): treat channel 0 as the
            // luminance and replicate it, so every energy formula keeps working.
            rgb[0] = rgb[1] = rgb[2] = data[idx];
        }
    }

    // Absolute intensity difference between two pixels, used by the forward-energy
    // costs (Rubinstein et al. 2008, section 5.1): luminance difference in luma mode,
    // or the L2 norm of the RGB difference otherwise. Boundaries wrap (see above).
    inline double get_pixel_diff(const std::vector<uint8_t>& data, int x1, int y1, int x2, int y2, int width, int height, int channels, bool use_luminance) {
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
            return std::sqrt(dr * dr + dg * dg + db * db);
        }
    }
}

SeamCarving::SeamCarving(const Image& image)
    : width_(image.width), height_(image.height), channels_(image.channels), data_(image.data) {
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

    // Fixed schedule: all width changes first, then all height changes. This is the
    // simple alternative to the optimal interleaving order of horizontal/vertical
    // removals (transport map, Avidan & Shamir 2007, section 4.2), which we do not
    // implement; it is faster and uses less memory at the price of not guaranteeing
    // the globally optimal order when both dimensions change (manuscript, section
    // Image Reduction).

    // Process width resizing
    if (target_width < width_) {
        carve_width(target_width, num_threads);
    } else if (target_width > width_) {
        insert_width(target_width, num_threads);
    }

    // Process height resizing
    if (target_height < height_) {
        carve_height(target_height, num_threads);
    } else if (target_height > height_) {
        insert_height(target_height, num_threads);
    }
}

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
        carved++;
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
// single routine serves both directions (manuscript, section Image Reduction).
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
// central difference is irrelevant (manuscript, Remark in section Energy).
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
            get_pixel_rgb(data_, x - 1, y, width_, height_, channels_, r_left);
            get_pixel_rgb(data_, x + 1, y, width_, height_, channels_, r_right);
            get_pixel_rgb(data_, x, y - 1, width_, height_, channels_, r_up);
            get_pixel_rgb(data_, x, y + 1, width_, height_, channels_, r_down);

            // Compute gradients. If use_luminance_ is enabled, we use perceptual weights (BT.601)
            // to compute luminance gradients (approximating greyscale gradients).
            // Otherwise, we calculate Euclidean distance (L2 norm) in RGB space.
            double rx = r_right[0] - r_left[0];
            double gx = r_right[1] - r_left[1];
            double bx = r_right[2] - r_left[2];

            double ry = r_down[0] - r_up[0];
            double gy = r_down[1] - r_up[1];
            double by = r_down[2] - r_up[2];

            double val = 0.0;
            if (use_luminance_) {
                double gx_luma = 0.299 * rx + 0.587 * gx + 0.114 * bx;
                double gy_luma = 0.299 * ry + 0.587 * gy + 0.114 * by;
                val = std::sqrt(gx_luma * gx_luma + gy_luma * gy_luma);
            } else {
                double delta_x_sq = rx * rx + gx * gx + bx * bx;
                double delta_y_sq = ry * ry + gy * gy + by * by;
                val = std::sqrt(delta_x_sq + delta_y_sq);
            }

            // Save the computed energy magnitude
            energy[y * width_ + x] = val;
        }
    }

    return energy;
}

// Optimal seam by dynamic programming (Cormen et al.). We build the cumulative cost
// map M top-down, then backtrack the minimum of the last row to recover the seam.
// The optional mask term is the per-pixel weight P(i,j) of Rubinstein et al. 2008,
// Eq. (2): it is added to M and realises object protection (large +) / removal
// (large -) as described in Avidan & Shamir 2007, section 4.6 (manuscript, Masking).
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
    // and is faster even on large images (manuscript, section Parallelisation).
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
                double c_u = get_pixel_diff(data_, x + 1, y, x - 1, y, width_, height_, channels_, use_luminance_);
                double c_l = c_u + get_pixel_diff(data_, x, y - 1, x - 1, y, width_, height_, channels_, use_luminance_);
                double c_r = c_u + get_pixel_diff(data_, x, y - 1, x + 1, y, width_, height_, channels_, use_luminance_);

                // Recurrence: M(y, x) = min( M(y-1, x-1) + c_l, M(y-1, x) + c_u, M(y-1, x+1) + c_r )
                double val_l = (x > 0) ? (M[(y - 1) * width_ + (x - 1)] + c_l) : 1e30;
                double val_u = M[(y - 1) * width_ + x] + c_u;
                double val_r = (x < width_ - 1) ? (M[(y - 1) * width_ + (x + 1)] + c_r) : 1e30;

                M[y * width_ + x] = std::min({val_l, val_u, val_r}) + mask_w;
            } else {
                // Classic backward energy (Avidan & Shamir 2007, dynamic program).
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
            double c_u = get_pixel_diff(data_, prev_x + 1, y + 1, prev_x - 1, y + 1, width_, height_, channels_, use_luminance_);
            double c_l = c_u + get_pixel_diff(data_, prev_x, y, prev_x - 1, y + 1, width_, height_, channels_, use_luminance_);
            double c_r = c_u + get_pixel_diff(data_, prev_x, y, prev_x + 1, y + 1, width_, height_, channels_, use_luminance_);

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
    // passes that each insert at most `width_` seams (std::min(K, width_)).
    while (width_ < target_width) {
        int K = target_width - width_;
        int k = std::min(K, width_);

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
            const auto& cols = row_cols_to_duplicate[y];
            int src_col = 0;
            int dst_col = 0;
            int cols_inserted = 0;

            while (src_col < width_) {
                // Copy the original pixel data
                int src_pixel_idx = (y * width_ + src_col) * channels_;
                int dst_pixel_idx = (y * new_width + dst_col) * channels_;
                for (int c = 0; c < channels_; ++c) {
                    new_data[dst_pixel_idx + c] = data_[src_pixel_idx + c];
                }
                if (!mask_weights_.empty()) {
                    new_mask_weights[y * new_width + dst_col] = mask_weights_[y * width_ + src_col];
                }
                dst_col++;

                // If this column index is chosen for duplication, insert a new pixel.
                // The new pixel's value is the average of current pixel and its right/left neighbour.
                if (cols_inserted < k && cols[cols_inserted] == src_col) {
                    int next_col = (src_col + 1 < width_) ? (src_col + 1) : (src_col - 1);
                    if (next_col < 0) next_col = src_col;

                    int nbr_pixel_idx = (y * width_ + next_col) * channels_;
                    int dup_pixel_idx = (y * new_width + dst_col) * channels_;

                    for (int c = 0; c < channels_; ++c) {
                        new_data[dup_pixel_idx + c] = static_cast<uint8_t>((static_cast<int>(data_[src_pixel_idx + c]) + static_cast<int>(data_[nbr_pixel_idx + c])) / 2);
                    }
                    if (!mask_weights_.empty()) {
                        new_mask_weights[y * new_width + dst_col] = (mask_weights_[y * width_ + src_col] + mask_weights_[y * width_ + next_col]) / 2.0;
                    }
                    dst_col++;
                    cols_inserted++;
                }
                src_col++;
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
