// Seam Carving for Content-Aware Image Resizing
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

#include "../src/seam_carving.h"
#include <iostream>
#include <cassert>
#include <cmath>

// Helper to create a simple synthetic 10x10 RGB image
Image create_synthetic_image() {
    Image img;
    img.width = 10;
    img.height = 10;
    img.channels = 3;
    img.data.resize(10 * 10 * 3, 255); // Initialise all white

    // Add a simple diagonal red line (so there's an energy gradient)
    for (int i = 0; i < 10; ++i) {
        int idx = (i * 10 + i) * 3;
        img.data[idx] = 255;   // R
        img.data[idx+1] = 0;   // G
        img.data[idx+2] = 0;   // B
    }
    return img;
}

// Test case 1: Shrinking dimensions
void test_shrinking() {
    Image img = create_synthetic_image();
    SeamCarving carver(img);

    // Carve width by 2 and height by 3 using backward energy
    carver.set_forward_energy(false);
    carver.resize(8, 7, 2);

    Image result = carver.get_image();
    assert(result.width == 8);
    assert(result.height == 7);
    assert(result.channels == 3);
    assert(result.data.size() == 8 * 7 * 3);
    std::cout << "test_shrinking passed!" << std::endl;
}

// Test case 2: Enlarging dimensions
void test_enlarging() {
    Image img = create_synthetic_image();
    SeamCarving carver(img);

    // Enlarge width by 2 and height by 3 using backward energy
    carver.set_forward_energy(false);
    carver.resize(12, 13, 2);

    Image result = carver.get_image();
    assert(result.width == 12);
    assert(result.height == 13);
    assert(result.channels == 3);
    assert(result.data.size() == 12 * 13 * 3);
    std::cout << "test_enlarging passed!" << std::endl;
}

// Test case 3: Forward energy resizing
void test_forward_energy() {
    Image img = create_synthetic_image();
    SeamCarving carver(img);

    // Forward energy shrinking & enlarging
    carver.set_forward_energy(true);
    carver.resize(9, 11, 2);

    Image result = carver.get_image();
    assert(result.width == 9);
    assert(result.height == 11);
    std::cout << "test_forward_energy passed!" << std::endl;
}

// Test case 4: Mask weights
void test_mask_weights() {
    Image img = create_synthetic_image();
    SeamCarving carver(img);

    // Create mask weights: protect column 5 (assign huge weight), force remove column 2 (assign small weight)
    std::vector<double> mask_weights(img.width * img.height, 0.0);
    for (int y = 0; y < img.height; ++y) {
        mask_weights[y * img.width + 5] = 1e6;  // Protect column 5
        mask_weights[y * img.width + 2] = -1e6; // Remove column 2
    }
    carver.set_mask_weights(mask_weights);

    // Carve width by 1
    carver.resize(9, 10, 1);

    // Get output image and check that column 2 was removed first, while column 5 is intact
    Image result = carver.get_image();
    assert(result.width == 9);
    assert(result.height == 10);
    std::cout << "test_mask_weights passed!" << std::endl;
}

// Test case 5: Large upscaling (> 2x width/height) to trigger multi-pass
void test_large_upscaling() {
    Image img = create_synthetic_image();
    SeamCarving carver(img);

    // Enlarge from 10x10 to 25x25 (which is > 2x)
    carver.resize(25, 25, 4);

    Image result = carver.get_image();
    assert(result.width == 25);
    assert(result.height == 25);
    std::cout << "test_large_upscaling passed!" << std::endl;
}

// Test case 6: Luminance disabled (raw L2 RGB norm)
void test_no_luminance() {
    Image img = create_synthetic_image();
    SeamCarving carver(img);
    carver.set_luminance_energy(false);
    carver.resize(8, 8, 2);
    Image result = carver.get_image();
    assert(result.width == 8);
    assert(result.height == 8);
    std::cout << "test_no_luminance passed!" << std::endl;
}

// Helper to create a larger synthetic 30x30 RGB image
Image create_larger_synthetic_image() {
    Image img;
    img.width = 30;
    img.height = 30;
    img.channels = 3;
    img.data.resize(30 * 30 * 3);
    for (int y = 0; y < 30; ++y) {
        for (int x = 0; x < 30; ++x) {
            int idx = (y * 30 + x) * 3;
            img.data[idx] = static_cast<uint8_t>((x * 255) / 30);       // R
            img.data[idx+1] = static_cast<uint8_t>((y * 255) / 30);     // G
            img.data[idx+2] = static_cast<uint8_t>(((x + y) * 255) / 60); // B
        }
    }
    return img;
}

// Test case 7: Verify output is identical across different thread counts
void test_parallelism_consistency() {
    Image img = create_larger_synthetic_image();

    struct Config {
        int target_w;
        int target_h;
        bool forward;
        bool luma;
        bool use_mask;
    };

    std::vector<Config> configs = {
        {24, 22, false, true, false},
        {24, 22, true, true, false},
        {35, 36, false, true, false},
        {35, 36, true, true, false},
        {26, 26, false, true, true},
        {25, 25, false, false, false}
    };

    // Prepare mask weights if needed
    std::vector<double> mask_weights(img.width * img.height, 0.0);
    for (int y = 0; y < img.height; ++y) {
        mask_weights[y * img.width + 15] = 1000.0; // Protect column 15
    }

    for (size_t i = 0; i < configs.size(); ++i) {
        const auto& cfg = configs[i];

        // 1 thread (baseline) - run with parallel_dp disabled (since 1 thread doesn't benefit from parallel DP anyway)
        SeamCarving carver1(img);
        carver1.set_forward_energy(cfg.forward);
        carver1.set_luminance_energy(cfg.luma);
        carver1.set_parallel_dp(false);
        if (cfg.use_mask) {
            carver1.set_mask_weights(mask_weights);
        }
        carver1.resize(cfg.target_w, cfg.target_h, 1);
        Image baseline = carver1.get_image();

        // Check for 2, 4, 8 threads
        for (int threads : {2, 4, 8}) {
            for (bool parallel_dp : {false, true}) {
                SeamCarving carver_p(img);
                carver_p.set_forward_energy(cfg.forward);
                carver_p.set_luminance_energy(cfg.luma);
                carver_p.set_parallel_dp(parallel_dp);
                if (cfg.use_mask) {
                    carver_p.set_mask_weights(mask_weights);
                }
                carver_p.resize(cfg.target_w, cfg.target_h, threads);
                Image result = carver_p.get_image();

                // Assert exact equality of the resulting images
                assert(result.width == baseline.width);
                assert(result.height == baseline.height);
                assert(result.channels == baseline.channels);
                assert(result.data == baseline.data);
            }
        }
    }
    std::cout << "test_parallelism_consistency passed!" << std::endl;
}

int main() {
    std::cout << "Running seam carving tests..." << std::endl;
    test_shrinking();
    test_enlarging();
    test_forward_energy();
    test_mask_weights();
    test_large_upscaling();
    test_no_luminance();
    test_parallelism_consistency();
    std::cout << "All tests passed successfully!" << std::endl;
    return 0;
}
