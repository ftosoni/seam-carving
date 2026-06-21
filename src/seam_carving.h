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
// program (see the LICENSE file); if not, see
// <https://opensource.org/license/bsd-3-clause>.
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SEAM_CARVING_H
#define SEAM_CARVING_H

#include <vector>
#include <cstdint>

struct Image {
    int width = 0;
    int height = 0;
    int channels = 0;
    std::vector<uint8_t> data;
};

class SeamCarving {
public:
    explicit SeamCarving(const Image& image);

    // Get the current image state
    Image get_image() const;

    // Resize the image to target dimensions
    void resize(int target_width, int target_height, int num_threads);

    // Set the per-pixel weight mask (size width*height). It is the term P(i,j) of
    // Rubinstein et al. 2008, Eq. (2): large positive protects a region, large
    // negative carves it away first (object protection/removal, Avidan & Shamir
    // 2007, section 4.6).
    void set_mask_weights(const std::vector<double>& mask_weights);

    // Select the forward-energy criterion (Rubinstein et al. 2008, section 5.1)
    // instead of the default backward energy (Avidan & Shamir 2007, section 3.2).
    void set_forward_energy(bool enable);

    // Compute gradients on the BT.601 luminance (default) or directly on RGB.
    void set_luminance_energy(bool enable);

    // Enable or disable verbose progress logging to stdout
    void set_verbose(bool enable);

    // Experimental: parallelise the inner loop of the DP recurrence (OpenMP). The
    // per-row barrier makes this slower than the default serial scan; kept so the
    // trade-off can be measured (see find_vertical_seam).
    void set_parallel_dp(bool enable);

    // --- Visualisation support -----------------------------------------------
    // These two methods expose data the operator already computes, so that the
    // separate visualisation module (visualize.h) can render figures without
    // duplicating any carving logic. They are the only additions made to the
    // class for visualisation purposes.

    // Backward dual-gradient energy map of the current image (size width*height,
    // row-major). This is always the backward criterion (Avidan & Shamir 2007,
    // section 3.2), honouring the luminance setting: forward energy is defined on
    // the edges that removal creates, not as a per-pixel map, so it has no direct
    // visual analogue here.
    std::vector<double> backward_energy_map(int num_threads);

    // The first `count` vertical seams that width reduction would remove, in
    // removal order, each expressed in the coordinate system of the *current*
    // image (one column per row). The work is done on an internal copy, so the
    // object itself is left unchanged. The active energy criterion (forward or
    // backward), the luminance setting and any mask are all respected. Used to
    // overlay the seams on the input image for figures.
    std::vector<std::vector<int>> seams_to_remove(int count, int num_threads);

private:
    // Carve width (reduces width using vertical seams)
    void carve_width(int target_width, int num_threads);
    
    // Carve height (reduces height using horizontal seams via transpose)
    void carve_height(int target_height, int num_threads);

    // Insert width (enlarges width using vertical seams)
    void insert_width(int target_width, int num_threads);

    // Insert height (enlarges height using horizontal seams via transpose)
    void insert_height(int target_height, int num_threads);

    // Backward energy map: dual-gradient magnitude e1 (Avidan & Shamir 2007, 3.2).
    // Unused (and not built) in forward-energy mode, which derives its costs in the DP.
    std::vector<double> compute_energy_map(int num_threads);

    // Optimal vertical seam by dynamic programming. Uses the precomputed backward
    // energy map, or the forward-energy step costs (Rubinstein et al. 2008, Eq. (2))
    // when forward energy is enabled.
    std::vector<int> find_vertical_seam(const std::vector<double>& energy, int num_threads);

    // Removes the vertical seam from the image
    void remove_vertical_seam(const std::vector<int>& seam, int num_threads);

    // Helper to transpose the image data (width <-> height)
    void transpose();

    int width_;
    int height_;
    int channels_;
    std::vector<uint8_t> data_;
    
    bool use_forward_energy_ = false;
    bool use_luminance_ = true;
    bool verbose_ = false;
    bool use_parallel_dp_ = false;
    std::vector<double> mask_weights_; // Vector of size width_ * height_
};

#endif // SEAM_CARVING_H
