// An Analysis and Implementation of Seam Carving for Content-Aware Image Resizing
//
// Copyright (c) 2026 Francesco Tosoni
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

#ifndef VISUALISE_H
#define VISUALISE_H

// -----------------------------------------------------------------------------
// Visualisation helpers for the seam-carving operator.
//
// These functions are deliberately kept separate from the SeamCarving class:
// they only *consume* its outputs (the backward energy map and the list of
// seams that would be removed) and turn them into RGB images for figures. No
// carving logic lives here, and nothing in this module mutates a SeamCarving
// object. The two products are:
//
//   * an energy map rendered through a perceptually-uniform colour map
//     (render_scalar_field), and
//   * the input image with the seams to be removed drawn on top
//     (render_seam_overlay).
//
// Everything is configurable by the caller (colour map, seam colour, greyscale
// background); see the command-line flags in main.cpp for the user-facing knobs.
// -----------------------------------------------------------------------------

#include "seam_carving.h"

#include <cstdint>
#include <string>
#include <vector>

namespace viz {

// Perceptually-uniform colour maps for rendering a scalar field. Viridis is the
// default. The colour data are sampled from matplotlib's maps, which are in the
// public domain (CC0), so they can be redistributed without licensing concerns.
// Grey is a plain linear intensity ramp.
enum class Colourmap { Viridis, Magma, Grey };

// Parse a colour-map name ("viridis", "magma", "grey"/"gray"). On an unknown
// name, returns Colourmap::Viridis and sets *ok (if non-null) to false.
Colourmap colourmap_from_string(const std::string& name, bool* ok = nullptr);

// Render a scalar field of size width*height (row-major) as an RGB Image. The
// field is linearly normalised from its [min, max] range to [0, 1] and then
// mapped through the chosen colour map. A flat field (max == min) maps to the
// low end of the map. This min-max normalisation is faithful and reproducible,
// at the cost that a few very high-energy pixels can compress the mid-tones.
Image render_scalar_field(const std::vector<double>& field, int width, int height,
                          Colourmap cmap);

// Draw the given seams over a copy of `background` (1 to 4 channels; greyscale
// and grey+alpha use channel 0 as the luminance, any alpha is dropped). The
// vertical_seams and horizontal_seams parameters hold the list of seams to paint.
// Each vertical seam is a vector of length background.height (giving the column
// coordinate x for every row y). Each horizontal seam is a vector of length
// background.width (giving the row coordinate y for every column x).
// Seams are drawn in `seam_rgb` (the default crimson is 220, 20, 60). When
// `greyscale_background` is true the image is first converted to BT.601 luminance
// so the coloured seams stand out more strongly.
Image render_seam_overlay(const Image& background,
                          const std::vector<std::vector<int>>& vertical_seams,
                          const std::vector<std::vector<int>>& horizontal_seams,
                          const uint8_t seam_rgb[3],
                          bool greyscale_background);

} // namespace viz

#endif // VISUALISE_H
