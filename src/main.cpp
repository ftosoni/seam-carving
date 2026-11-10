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

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "seam_carving.h"
#include "visualise.h"
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <algorithm>
#include <cctype>
#include <cmath>

#ifdef USE_OPENMP
#include <omp.h>
#endif

#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>

std::wstring utf8_to_utf16(const std::string& utf8) {
    int size_needed = MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), -1, NULL, 0);
    if (size_needed <= 0) return L"";
    std::wstring wstr(size_needed - 1, 0);
    MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), -1, &wstr[0], size_needed);
    return wstr;
}
#endif

FILE* fopen_utf8(const std::string& path, const char* mode) {
#ifdef _WIN32
    std::wstring wpath = utf8_to_utf16(path);
    std::wstring wmode = utf8_to_utf16(mode);
    return _wfopen(wpath.c_str(), wmode.c_str());
#else
    return fopen(path.c_str(), mode);
#endif
}

void stbi_write_to_file_callback(void *context, void *data, int size) {
    fwrite(data, 1, size, static_cast<FILE*>(context));
}

// Write an RGB Image to a PNG file, handling UTF-8 paths on Windows. Used both
// for the carved result and for the visualisation dumps. Returns true on success.
bool write_png_utf8(const std::string& path, const Image& image) {
    FILE* f = fopen_utf8(path, "wb");
    if (!f) {
        std::cerr << "Error: Failed to open output file for writing: " << path << "\n";
        return false;
    }
    int ok = stbi_write_png_to_func(stbi_write_to_file_callback, f,
                                    image.width, image.height, image.channels,
                                    image.data.data(), image.width * image.channels);
    fclose(f);
    if (!ok) {
        std::cerr << "Error: Failed to write image: " << path << "\n";
        return false;
    }
    return true;
}

// Parse an "r,g,b" triple (each component 0-255) into out[3]. Returns false on
// malformed input, leaving out unchanged.
bool parse_rgb_triple(const std::string& text, uint8_t out[3]) {
    int values[3];
    size_t start = 0;
    for (int i = 0; i < 3; ++i) {
        size_t comma = text.find(',', start);
        if (i < 2 && comma == std::string::npos) return false;
        std::string token = text.substr(start, (i < 2) ? (comma - start) : std::string::npos);
        try {
            values[i] = std::stoi(token);
        } catch (...) {
            return false;
        }
        if (values[i] < 0 || values[i] > 255) return false;
        start = comma + 1;
    }
    out[0] = static_cast<uint8_t>(values[0]);
    out[1] = static_cast<uint8_t>(values[1]);
    out[2] = static_cast<uint8_t>(values[2]);
    return true;
}

// Parse a strictly positive real number. Returns false on malformed input or on
// a value <= 0, leaving *out unchanged.
bool parse_positive_double(const std::string& text, double* out) {
    try {
        size_t used = 0;
        double v = std::stod(text, &used);
        if (used != text.size() || !(v > 0.0) || !std::isfinite(v)) return false;
        *out = v;
        return true;
    } catch (...) {
        return false;
    }
}

// Convert a relative target size (ratio of the input size) to pixels, rounding to
// the nearest integer and never going below one pixel.
int ratio_to_pixels(double ratio, int input_size) {
    long pixels = std::lround(ratio * static_cast<double>(input_size));
    return static_cast<int>(std::max(1L, pixels));
}

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <input_image> [output_image] [options]\n"
              << "Options:\n"
              << "  -w, --width <val>   Target width in pixels (defaults to input width)\n"
              << "  -h, --height <val>  Target height in pixels (defaults to input height)\n"
              << "  --width-ratio <r>   Target width as a fraction of the input width (r > 0;\n"
              << "                      r < 1 reduces, r > 1 enlarges). Alternative to -w\n"
              << "  --height-ratio <r>  Target height as a fraction of the input height. Alternative to -h\n"
              << "  -t, --threads <val> Number of threads for OpenMP (defaults to hardware concurrency)\n"
              << "  --forward           Use forward energy variant (Rubinstein 2008)\n"
              << "  --energy <name>     Backward energy function (Avidan & Shamir 2007, section 3.2):\n"
              << "                      gradient (default), hog (gradient / local HoG maximum, 11x11),\n"
              << "                      entropy (gradient + local entropy, 9x9). Not used with --forward\n"
              << "  --norm <name>       Norm of the gradient and of the colour differences: l2 (default)\n"
              << "                      or l1 (the e1 of Avidan & Shamir 2007, section 3.2)\n"
              << "  --no-luma           Disable perceptual luminance weights (use the RGB norm instead)\n"
              << "  --height-first      When both dimensions change, process the height before the width\n"
              << "  -m, --mask <path>   Path to mask image for pixel protection/removal (Green/White to protect, Red/Black to remove)\n"
              << "  --parallel-dp       Enable parallel dynamic programming step (OpenMP)\n"
              << "  --help              Show this help message\n"
              << "Visualisation (optional, written from the input before carving):\n"
              << "  --dump-energy <path>  Write the backward energy map as a colour-mapped PNG\n"
              << "  --dump-seams <path>   Write the input with the seams removed to reach -w overlaid\n"
              << "  --viz-colourmap <name> Colour map for --dump-energy: viridis (default), magma, grey\n"
              << "  --viz-clip <p>        Clip the energy map at its p-th percentile before colour\n"
              << "                        mapping (0 < p <= 100, default 99; 100 = plain min-max)\n"
              << "  --seam-colour <r,g,b>  Seam overlay colour (default 220,20,60, crimson)\n"
              << "  --seam-order <name>   Colour each seam by its removal order with the given colour\n"
              << "                        map (viridis, magma, grey) instead of --seam-colour\n"
              << "  --seam-on-grey        Draw seams over a greyscale copy of the input, with its\n"
              << "                        grey levels compressed into [80, 176]\n";
}

std::string get_default_output_path(const std::string& input_path) {
    size_t dot_idx = input_path.find_last_of('.');
    if (dot_idx == std::string::npos) {
        return input_path + "_carved.png";
    }
    return input_path.substr(0, dot_idx) + "_carved" + input_path.substr(dot_idx);
}

int main(int argc, char* argv[]) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
    // Parse arguments with UTF-8 support on Windows
    std::vector<std::string> args;
#ifdef _WIN32
    int argc_w;
    wchar_t** argv_w = CommandLineToArgvW(GetCommandLineW(), &argc_w);
    if (argv_w) {
        for (int i = 0; i < argc_w; ++i) {
            int size_needed = WideCharToMultiByte(CP_UTF8, 0, argv_w[i], -1, NULL, 0, NULL, NULL);
            if (size_needed > 0) {
                std::string arg(size_needed - 1, 0);
                WideCharToMultiByte(CP_UTF8, 0, argv_w[i], -1, &arg[0], size_needed, NULL, NULL);
                args.push_back(arg);
            }
        }
        LocalFree(argv_w);
    }
#else
    for (int i = 0; i < argc; ++i) {
        args.push_back(argv[i]);
    }
#endif

    if (args.size() < 2) {
        print_usage(args.empty() ? "seam_carving" : args[0].c_str());
        return 1;
    }

    std::string input_path = args[1];
    if (input_path == "--help" || input_path == "-h") {
        print_usage(args[0].c_str());
        return 0;
    }

    std::string output_path = "";
    size_t arg_start_idx = 2;

    if (args.size() >= 3 && args[2][0] != '-') {
        output_path = args[2];
        arg_start_idx = 3;
    } else {
        output_path = get_default_output_path(input_path);
    }

    int target_width = -1;
    int target_height = -1;
    double width_ratio = -1.0;  // --width-ratio / --height-ratio (unset if <= 0)
    double height_ratio = -1.0;
    int num_threads = std::thread::hardware_concurrency();
    if (num_threads <= 0) num_threads = 1;

    bool use_forward = false;
    EnergyFunction energy_function = EnergyFunction::Gradient;
    GradientNorm gradient_norm = GradientNorm::L2;
    bool use_luma = true;
    bool height_first = false;
    bool use_parallel_dp = false;
    std::string mask_path = "";

    // Visualisation options (all optional; empty dump paths mean "do not dump").
    std::string dump_energy_path = "";
    std::string dump_seams_path = "";
    std::string viz_colourmap_name = "viridis";
    double viz_clip_percentile = viz::kDefaultClipPercentile;
    std::string seam_order_name = "";  // empty: single-colour seams (--seam-colour)
    uint8_t seam_colour[3] = {220, 20, 60}; // crimson by default
    bool seam_on_grey = false;

    for (size_t i = arg_start_idx; i < args.size(); ++i) {
        std::string arg = args[i];
        if ((arg == "-w" || arg == "--width") && i + 1 < args.size()) {
            target_width = std::stoi(args[++i]);
        } else if ((arg == "-h" || arg == "--height") && i + 1 < args.size()) {
            target_height = std::stoi(args[++i]);
        } else if (arg == "--width-ratio" && i + 1 < args.size()) {
            if (!parse_positive_double(args[++i], &width_ratio)) {
                std::cerr << "Error: --width-ratio expects a positive number, got '" << args[i] << "'.\n";
                return 1;
            }
        } else if (arg == "--height-ratio" && i + 1 < args.size()) {
            if (!parse_positive_double(args[++i], &height_ratio)) {
                std::cerr << "Error: --height-ratio expects a positive number, got '" << args[i] << "'.\n";
                return 1;
            }
        } else if ((arg == "-t" || arg == "--threads") && i + 1 < args.size()) {
            num_threads = std::stoi(args[++i]);
        } else if (arg == "--forward") {
            use_forward = true;
        } else if (arg == "--energy" && i + 1 < args.size()) {
            const std::string name = args[++i];
            if (name == "gradient") {
                energy_function = EnergyFunction::Gradient;
            } else if (name == "hog") {
                energy_function = EnergyFunction::HoG;
            } else if (name == "entropy") {
                energy_function = EnergyFunction::Entropy;
            } else {
                std::cerr << "Error: Unknown energy function '" << name
                          << "', expected gradient, hog or entropy.\n";
                return 1;
            }
        } else if (arg == "--norm" && i + 1 < args.size()) {
            const std::string name = args[++i];
            if (name == "l2" || name == "L2") {
                gradient_norm = GradientNorm::L2;
            } else if (name == "l1" || name == "L1") {
                gradient_norm = GradientNorm::L1;
            } else {
                std::cerr << "Error: Unknown norm '" << name << "', expected l1 or l2.\n";
                return 1;
            }
        } else if (arg == "--no-luma") {
            use_luma = false;
        } else if (arg == "--height-first") {
            height_first = true;
        } else if ((arg == "-m" || arg == "--mask") && i + 1 < args.size()) {
            mask_path = args[++i];
        } else if (arg == "--parallel-dp") {
            use_parallel_dp = true;
        } else if (arg == "--dump-energy" && i + 1 < args.size()) {
            dump_energy_path = args[++i];
        } else if (arg == "--dump-seams" && i + 1 < args.size()) {
            dump_seams_path = args[++i];
        } else if (arg == "--viz-colourmap" && i + 1 < args.size()) {
            viz_colourmap_name = args[++i];
        } else if (arg == "--viz-clip" && i + 1 < args.size()) {
            if (!parse_positive_double(args[++i], &viz_clip_percentile) || viz_clip_percentile > 100.0) {
                std::cerr << "Error: --viz-clip expects a percentile in (0, 100], got '" << args[i] << "'.\n";
                return 1;
            }
        } else if (arg == "--seam-order" && i + 1 < args.size()) {
            seam_order_name = args[++i];
        } else if (arg == "--seam-colour" && i + 1 < args.size()) {
            if (!parse_rgb_triple(args[++i], seam_colour)) {
                std::cerr << "Warning: Could not parse --seam-colour '" << args[i]
                          << "', expected r,g,b with values 0-255. Using default.\n";
            }
        } else if (arg == "--seam-on-grey") {
            seam_on_grey = true;
        } else {
            std::cerr << "Warning: Unknown option or missing value: " << arg << "\n";
        }
    }

    int width, height, channels;
    // Load image using fopen_utf8
    FILE* f = fopen_utf8(input_path, "rb");
    if (!f) {
        std::cerr << "Error: Failed to open image file: " << input_path << "\n";
        return 1;
    }
    uint8_t* img_data = stbi_load_from_file(f, &width, &height, &channels, 0);
    fclose(f);
    if (!img_data) {
        std::cerr << "Error: Failed to load image: " << input_path << "\n";
        return 1;
    }

    std::cout << "Loaded image: " << input_path << " (" << width << "x" << height << ", " << channels << " channels)\n";

    // The energy functions of --energy modulate the backward per-pixel energy;
    // forward energy has no per-pixel energy map to modulate.
    if (use_forward && energy_function != EnergyFunction::Gradient) {
        std::cerr << "Error: --energy hog/entropy applies to backward energy only; "
                  << "it cannot be combined with --forward.\n";
        return 1;
    }

    // Relative targets (--width-ratio / --height-ratio) are resolved against the
    // input size; they are an alternative to the absolute -w / -h, not a modifier.
    if (width_ratio > 0.0 && target_width != -1) {
        std::cerr << "Error: use either -w/--width or --width-ratio, not both.\n";
        return 1;
    }
    if (height_ratio > 0.0 && target_height != -1) {
        std::cerr << "Error: use either -h/--height or --height-ratio, not both.\n";
        return 1;
    }
    if (width_ratio > 0.0) target_width = ratio_to_pixels(width_ratio, width);
    if (height_ratio > 0.0) target_height = ratio_to_pixels(height_ratio, height);

    if (target_width == -1) target_width = width;
    if (target_height == -1) target_height = height;



    // Set number of threads in OpenMP if compiled with OpenMP
    #ifdef USE_OPENMP
    omp_set_num_threads(num_threads);
    std::cout << "Running with " << num_threads << " OpenMP thread(s).\n";
    #else
    std::cout << "Compiled without OpenMP. Running single-threaded.\n";
    #endif

    // Setup input image struct
    Image input_image;
    input_image.width = width;
    input_image.height = height;
    input_image.channels = channels;
    input_image.data.assign(img_data, img_data + (width * height * channels));
    stbi_image_free(img_data);

    // Load mask weights if provided
    std::vector<double> mask_weights;
    if (!mask_path.empty()) {
        int m_width, m_height, m_channels;
        FILE* mf = fopen_utf8(mask_path, "rb");
        if (!mf) {
            std::cerr << "Error: Failed to open mask file: " << mask_path << "\n";
            return 1;
        }
        uint8_t* m_data = stbi_load_from_file(mf, &m_width, &m_height, &m_channels, 0);
        fclose(mf);
        if (!m_data) {
            std::cerr << "Error: Failed to load mask image: " << mask_path << "\n";
            return 1;
        }
        if (m_width != width || m_height != height) {
            std::cerr << "Error: Mask dimensions (" << m_width << "x" << m_height
                      << ") must match input image dimensions (" << width << "x" << height << ").\n";
            stbi_image_free(m_data);
            return 1;
        }

        // Convert the mask image to per-pixel weights P(i,j) (Rubinstein et al. 2008,
        // Eq. (2); object protection/removal of Avidan & Shamir 2007, section 4.6).
        // The weights are (IPOL manuscript, section "Masking", Equations (12)-(13)):
        //   colour mask    (R, G, B):  P = kMaskScale * (G - R)          in [-255, 255] * kMaskScale
        //   greyscale mask (V):        P = kMaskScale * 2 * (V - 128)    in [-256, 254] * kMaskScale
        // The map is continuous: only G == R (colour) or V == 128 (greyscale) is
        // neutral. The scale kMaskScale = 1e6 makes even the smallest non-zero weight
        // (one intensity level) exceed the energy of a whole seam, so any deviation
        // from neutral acts as a strong constraint rather than a graded weight:
        // positive weights repel seams (protect), negative ones attract them (remove
        // first); larger magnitudes only rank marked pixels among themselves.
        // (Per row, the default gradient energy is at most 255*sqrt(6) ~ 625 and the
        // forward step cost at most 2*255*sqrt(3) ~ 883, so a whole seam stays below
        // 1e6 for images of up to about 1100 rows. e_HoG is smaller than the gradient;
        // e_Entropy adds up to 255*log2(81) ~ 1617 per row, so the bound drops to
        // about 450 rows, beyond which marked regions still dominate but no longer
        // strictly outweigh every unmarked seam.)
        constexpr double kMaskScale = 1e6;
        mask_weights.resize(width * height);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int idx = (y * width + x) * m_channels;
                double weight = 0.0;
                if (m_channels >= 3) {
                    // Colour mask: weight proportional to (green - red).
                    double r_val = m_data[idx];
                    double g_val = m_data[idx + 1];
                    weight = (g_val - r_val) * kMaskScale;
                } else {
                    // Greyscale mask: map intensity continuously relative to mid-grey (128).
                    // White (255) protects, black (0) removes, and mid-grey (128) is neutral.
                    // The scale matches the range of the colour mask difference [-255, 255].
                    double val = m_data[idx];
                    weight = (val - 128.0) * 2.0 * kMaskScale;
                }
                mask_weights[y * width + x] = weight;
            }
        }
        stbi_image_free(m_data);
        std::cout << "Loaded mask: " << mask_path << " (" << m_width << "x" << m_height << ")\n";
    }

    // Run Seam Carving
    auto start_time = std::chrono::high_resolution_clock::now();
    
    SeamCarving carver(input_image);
    if (!mask_weights.empty()) {
        carver.set_mask_weights(mask_weights);
    }
    carver.set_forward_energy(use_forward);
    carver.set_energy_function(energy_function);
    carver.set_luminance_energy(use_luma);
    carver.set_gradient_norm(gradient_norm);
    carver.set_height_first(height_first);
    carver.set_parallel_dp(use_parallel_dp);
    carver.set_verbose(true);

    // Visualisation dumps are produced from the input, before any carving, so a
    // single run yields the carved result alongside the energy and seam figures.
    // backward_energy_map() does not mutate the carver, and seams_to_remove()
    // works on an internal copy, so neither call disturbs the resize below.
    if (!dump_energy_path.empty()) {
        bool ok = true;
        viz::Colourmap cmap = viz::colourmap_from_string(viz_colourmap_name, &ok);
        if (!ok) {
            std::cerr << "Warning: Unknown colour map '" << viz_colourmap_name
                      << "', falling back to viridis.\n";
        }
        std::vector<double> energy = carver.backward_energy_map(num_threads);
        Image energy_img = viz::render_scalar_field(energy, width, height, cmap, viz_clip_percentile);
        if (write_png_utf8(dump_energy_path, energy_img)) {
            std::cout << "Wrote energy map to: " << dump_energy_path << "\n";
        }
    }
    if (!dump_seams_path.empty()) {
        int v_count = width - target_width;
        int h_count = height - target_height;
        if (v_count <= 0 && h_count <= 0) {
            std::cerr << "Warning: --dump-seams needs a target width (-w) or height (-h) smaller than the input "
                      << "dimensions; skipping the seam overlay.\n";
        } else {
            std::vector<std::vector<int>> v_seams;
            std::vector<std::vector<int>> h_seams;
            if (v_count > 0) {
                v_seams = carver.seams_to_remove(v_count, num_threads);
            }
            if (h_count > 0) {
                h_seams = carver.seams_to_remove_horizontal(h_count, num_threads);
            }
            Image seams_img;
            if (seam_order_name.empty()) {
                seams_img = viz::render_seam_overlay(input_image, v_seams, h_seams, seam_colour, seam_on_grey);
            } else {
                bool ok = true;
                viz::Colourmap order_cmap = viz::colourmap_from_string(seam_order_name, &ok);
                if (!ok) {
                    std::cerr << "Warning: Unknown colour map '" << seam_order_name
                              << "' for --seam-order, falling back to viridis.\n";
                }
                seams_img = viz::render_seam_overlay_by_order(input_image, v_seams, h_seams, order_cmap, seam_on_grey);
            }
            if (write_png_utf8(dump_seams_path, seams_img)) {
                std::cout << "Wrote seam overlay (" << v_seams.size() << " vertical, "
                          << h_seams.size() << " horizontal seams) to: "
                          << dump_seams_path << "\n";
            }
        }
    }

    try {
        carver.resize(target_width, target_height, num_threads);
    } catch (const std::exception& e) {
        std::cerr << "Error during seam carving: " << e.what() << "\n";
        return 1;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;

    Image output_image = carver.get_image();
    std::cout << "Seam carving completed in " << diff.count() << " seconds.\n";
    std::cout << "Output dimensions: " << output_image.width << "x" << output_image.height << "\n";

    // Write output image
    std::string ext = "";
    size_t dot_idx = output_path.find_last_of('.');
    if (dot_idx != std::string::npos) {
        ext = output_path.substr(dot_idx + 1);
        for (auto& c : ext) c = std::tolower(c);
    }

    FILE* out_f = fopen_utf8(output_path, "wb");
    if (!out_f) {
        std::cerr << "Error: Failed to open output file for writing: " << output_path << "\n";
        return 1;
    }

    int write_result = 0;
    if (ext == "png") {
        write_result = stbi_write_png_to_func(stbi_write_to_file_callback, out_f, output_image.width, output_image.height, 
                                              output_image.channels, output_image.data.data(), output_image.width * output_image.channels);
    } else if (ext == "jpg" || ext == "jpeg") {
        write_result = stbi_write_jpg_to_func(stbi_write_to_file_callback, out_f, output_image.width, output_image.height, 
                                              output_image.channels, output_image.data.data(), 90);
    } else if (ext == "bmp") {
        write_result = stbi_write_bmp_to_func(stbi_write_to_file_callback, out_f, output_image.width, output_image.height, 
                                              output_image.channels, output_image.data.data());
    } else {
        std::cout << "Unknown extension ." << ext << ", defaulting to PNG.\n";
        write_result = stbi_write_png_to_func(stbi_write_to_file_callback, out_f, output_image.width, output_image.height, 
                                              output_image.channels, output_image.data.data(), output_image.width * output_image.channels);
    }
    fclose(out_f);

    if (!write_result) {
        std::cerr << "Error: Failed to write image: " << output_path << "\n";
        return 1;
    }

    std::cout << "Successfully saved carved image to: " << output_path << "\n";
    return 0;
}
