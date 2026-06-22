================================================================================
An Analysis and Implementation of Seam Carving for Content-Aware Image Resizing
================================================================================

IPOL article:
    An Analysis and Implementation of Seam Carving for Content-Aware Image Resizing
    Image Processing On Line (IPOL), http://www.ipol.im/
    (The final article URL and DOI will be assigned upon publication.)

Version: 1.0.0
Date:    2026-06-21

Author:  Francesco Tosoni
Contact: Francesco.Tosoni@santannapisa.it
         L'EMbeDS, Sant'Anna School of Advanced Studies, Pisa, Italy

Source repository: https://github.com/ftosoni/seam-carving


--------------------------------------------------------------------------------
1. Description and organisation of the files
--------------------------------------------------------------------------------

This archive contains a C++17 implementation of the seam carving content-aware
image resizing operator (Avidan & Shamir, 2007), including the forward-energy
criterion (Rubinstein, Shamir & Avidan, 2008). It supports image reduction,
enlargement by ordered seam insertion, multi-pass enlargement, a user-supplied
weight mask for object protection and removal, and optional OpenMP
parallelisation.

Directory layout:

    CMakeLists.txt        Build configuration (CMake).
    LICENCE               BSD 3-Clause license text.
    CITATION.cff          Citation metadata (Citation File Format).
    codemeta.json         Software metadata (CodeMeta schema).
    README.txt            This file.

    src/                  Source code:
        main.cpp          Command-line front end: argument parsing, image and
                          mask input/output (via stb_image), and the
                          visualisation dumps.
        seam_carving.h    Declaration of the SeamCarving operator class.
        seam_carving.cpp  Core operator: energy, dynamic programming, seam
                          removal/insertion, transpose, masking.
        visualise.h       Declaration of the visualisation helpers.
        visualise.cpp     Rendering of the energy map (colour maps) and of the
                          seam overlays used for the figures.

    tests/                Unit tests:
        tests.cpp         Self-contained test driver (no external framework).

    thirdparty/           Third-party, public-domain image I/O headers:
        stb_image.h       Image loading (public domain / MIT, by Sean Barrett).
        stb_image_write.h Image writing (public domain / MIT, by Sean Barrett).

    images/               Test data: input images and masks used in the article
                          (portal, pont, birds, aurora) and their masks.


--------------------------------------------------------------------------------
2. Where each algorithm is implemented
--------------------------------------------------------------------------------

All references below are to the sections, equations and algorithms of the
accompanying IPOL article.

    - Energy map (Section "Energy", the backward dual-gradient energy):
          SeamCarving::compute_energy_map   in src/seam_carving.cpp

    - Optimal seam by dynamic programming (Section "Optimal Seam by Dynamic
      Programming") and the forward-energy criterion (Section "Forward Energy",
      Eq. (2)), together with the mask term P(i,j):
          SeamCarving::find_vertical_seam    in src/seam_carving.cpp

    - Algorithm 1 (Width reduction by seam removal):
          SeamCarving::carve_width           in src/seam_carving.cpp
      using find_vertical_seam and SeamCarving::remove_vertical_seam.
      Height reduction reuses the same routine on the transposed image:
          SeamCarving::carve_height / SeamCarving::transpose

    - Algorithm 2 (Width enlargement by ordered seam insertion, one pass):
          SeamCarving::insert_width          in src/seam_carving.cpp
      Height enlargement reuses it via transposition:
          SeamCarving::insert_height

    - Masking (Section "Masking: Object Protection and Removal"): the mask image
      is converted to per-pixel weights in
          main()                              in src/main.cpp
      and applied as the term P(i,j) in find_vertical_seam.

    - Visualisation (the energy and seam figures of the "Examples" section):
          SeamCarving::backward_energy_map and SeamCarving::seams_to_remove
          in src/seam_carving.cpp, rendered by the functions in
          src/visualise.cpp.


--------------------------------------------------------------------------------
3. Compilation
--------------------------------------------------------------------------------

Prerequisites:
    - A C++17 compliant compiler (GCC, Clang or MSVC).
    - CMake >= 3.12.
    - OpenMP (optional, enables multi-threading; the program builds and runs
      without it).

Build steps (from the root of this archive):

    cmake -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build --config Release

This produces the executable "seam_carving" (or "seam_carving.exe" on Windows)
inside the "build" directory. The unit tests are built as "seam_carving_tests"
and can be run with:

    ctest --test-dir build


--------------------------------------------------------------------------------
4. Examples of use
--------------------------------------------------------------------------------

General invocation:

    ./build/seam_carving <input_image> [output_image] [options]

If the output path is omitted, the result is named <input>_carved.<ext>.

Main options:
    -w, --width  <val>    Target width  (default: input width).
    -h, --height <val>    Target height (default: input height).
    -t, --threads <val>   Number of OpenMP threads (default: hardware cores).
    --forward             Use forward energy (Rubinstein et al., 2008).
    --no-luma             Compute energy on the raw RGB norm instead of the
                          BT.601 luminance.
    -m, --mask <path>     Weight mask for object protection/removal
                          (green protects, red removes; see Section 6).
    --help                Show the full help message.

Visualisation options (written from the input, before carving):
    --dump-energy <path>  Save the backward energy map as a colour-mapped PNG.
    --dump-seams  <path>  Save the input with the seams removed to reach -w
                          overlaid.
    --viz-colourmap <name> Colour map for --dump-energy: viridis (default),
                          magma, grey.
    --seam-colour <r,g,b> Seam overlay colour (default 220,20,60, crimson).
    --seam-on-grey        Draw the seams over a greyscale copy of the input.

Examples:

    (a) Reduce the width to 640 with backward energy:
        ./build/seam_carving images/portal_orig.jpg out.png -w 640

    (b) Reduce width to 560 with forward energy:
        ./build/seam_carving images/pont_orig.jpg out.png -w 560 --forward

    (c) Enlarge the width to 1280 by ordered seam insertion:
        ./build/seam_carving images/portal_orig.jpg out.png -w 1280

    (d) Object removal/protection with a mask:
        ./build/seam_carving images/birds_orig.png out.png -w 700 \
            -m images/birds_mask.png

    (e) Reproduce the energy and seam figures of the first example:
        ./build/seam_carving images/portal_orig.jpg out.png -w 640 \
            --dump-energy energy.png --dump-seams seams.png --seam-on-grey


--------------------------------------------------------------------------------
5. Copyright and license
--------------------------------------------------------------------------------

Copyright (c) 2026 Francesco Tosoni

Patent notice: the seam carving algorithm implemented here is possibly linked to
the patent US 7,747,107 B2 ("Method for retargeting images", S. Avidan and
A. Shamir, Mitsubishi Electric Research Laboratories;
<https://patents.google.com/patent/US7747107B2/en>; anticipated expiry
2028-12-07, listed by patent databases as expired, fee-related). The source
files implementing the algorithm carry the full IPOL patent warning in their
header. This software is provided as a scientific tool to verify the algorithm
description; it is your responsibility to determine which patent restrictions
apply before you compile, use, modify, or redistribute it. A patent lawyer is
qualified to make this determination.

This program is free software: you can redistribute it and/or modify it under
the terms of the BSD 3-Clause License. This program is distributed in the hope
that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the full
license text in the LICENCE file, or <https://opensource.org/license/bsd-3-clause>.

The third-party headers in thirdparty/ (stb_image.h, stb_image_write.h, by
Sean Barrett) are in the public domain (or MIT, at your option) and are
distributed under their own terms, stated in the files themselves.

================================================================================
