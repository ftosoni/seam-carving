================================================================================
An Analysis and Implementation of Seam Carving for Content-Aware Image Resizing
================================================================================

IPOL article:
    An Analysis and Implementation of Seam Carving for Content-Aware Image Resizing
    Image Processing On Line (IPOL), http://www.ipol.im/
    (The final article URL and DOI will be assigned upon publication.)

Version: 1.1.0
Date:    2026-11-10

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

    tests/                Tests, run by ctest:
        tests.cpp         Unit tests (self-contained, no external framework).
        cli_tests.cmake   Command-line tests: every option, energy function and
                          error case of the program, on the images in images/.

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

    - Energy map (Section "Energy", Equations (3)-(5); line 2 of Algorithm 1):
          SeamCarving::compute_energy_map   in src/seam_carving.cpp
      and the alternative energy functions e_HoG (Equation (6)) and
      e_Entropy (Equation (7)):
          SeamCarving::apply_hog, SeamCarving::add_entropy

    - Optimal seam by dynamic programming (Section "Optimal Seam by Dynamic
      Programming", Equation (9)) and the forward-energy criterion (Section
      "Forward Energy", Equations (10)-(11)), together with the mask term
      P(i,j); lines 3-5 of Algorithm 1:
          SeamCarving::find_vertical_seam    in src/seam_carving.cpp

    - Algorithm 1 (Width reduction by seam removal), lines 1-7:
          SeamCarving::carve_width           in src/seam_carving.cpp
      using find_vertical_seam and SeamCarving::remove_vertical_seam (line 6).
      Height reduction reuses the same routine on the transposed image:
          SeamCarving::carve_height / SeamCarving::transpose

    - Algorithm 2 (Width enlargement by ordered seam insertion, one pass),
      lines 1-8, repeated in passes of at most 50% of the width:
          SeamCarving::insert_width          in src/seam_carving.cpp
      Height enlargement reuses it via transposition:
          SeamCarving::insert_height

    - Masking (Section "Masking: Object Protection and Removal", Equations
      (12)-(13)): the mask image is converted to per-pixel weights in
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
inside the "build" directory. The unit tests are built as "seam_carving_tests";
together with the command-line tests of tests/cli_tests.cmake they can be run
with:

    ctest --test-dir build -C Release

The same tests run on Linux, Windows and macOS at every push to the GitHub
repository (.github/workflows/ci.yml).


--------------------------------------------------------------------------------
4. Examples of use
--------------------------------------------------------------------------------

General invocation:

    ./build/seam_carving <input_image> [output_image] [options]

If the output path is omitted, the result is named <input>_carved.<ext>.

Main options:
    -w, --width  <val>    Target width in pixels  (default: input width).
    -h, --height <val>    Target height in pixels (default: input height).
    --width-ratio  <r>    Target width as a fraction r > 0 of the input width
                          (r < 1 reduces, r > 1 enlarges; rounded to the
                          nearest pixel). Alternative to -w.
    --height-ratio <r>    Same for the height. Alternative to -h.
    -t, --threads <val>   Number of OpenMP threads (default: hardware cores).
    --forward             Use forward energy (Rubinstein et al., 2008).
    --energy <name>       Backward energy function (Avidan & Shamir 2007,
                          Section 3.2): gradient (default, gradient magnitude),
                          hog (gradient divided by the maximum of an 8-bin
                          histogram of oriented gradients over an 11x11 window)
                          or entropy (gradient plus the local entropy over a
                          9x9 window). Backward energy only: it cannot be
                          combined with --forward.
    --norm <name>         Norm of the gradient (and of the RGB differences with
                          --no-luma): l2 (default) or l1, the e1 of Avidan &
                          Shamir (2007, Section 3.2).
    --no-luma             Compute energy on the raw RGB norm instead of the
                          BT.601 luminance.
    --height-first        When both dimensions change, process the height
                          (horizontal seams) before the width; by default the
                          width is processed first.
    -m, --mask <path>     Weight mask for object protection/removal
                          (see "Weight masks" below and Section 6).
    --parallel-dp         Experimental: parallelise the inner loop of the
                          dynamic-programming recurrence with OpenMP. This is
                          slower than the default serial scan (see Section 7)
                          and is kept only to measure the trade-off; the result
                          is identical.
    --help                Show the full help message.

Visualisation options (written from the input, before carving):
    --dump-energy <path>  Save the backward energy map as a colour-mapped PNG.
    --dump-seams  <path>  Save the input with the seams removed to reach the
                          target width/height overlaid.
    --viz-colourmap <name> Colour map for --dump-energy: viridis (default),
                          magma, grey.
    --viz-clip <p>        Clip the energy map at its p-th percentile before
                          colour mapping (0 < p <= 100, default 99), so that a
                          few very strong edges do not darken all intermediate
                          values; 100 gives the plain min-max normalisation.
    --seam-colour <r,g,b> Seam overlay colour (default 220,20,60, crimson).
    --seam-order <name>   Colour each seam by its removal order through the
                          given colour map (viridis, magma, grey): the first
                          seam removed takes the low end of the map, the last
                          one the high end. Overrides --seam-colour.
    --seam-on-grey        Draw the seams over a greyscale copy of the input,
                          with its grey levels compressed into [80, 176] so
                          that both dark and light seam colours stand out.

Weight masks (-m):
    The mask must have the same size as the input image. Each mask pixel is
    turned into a weight P that is added to the cumulative seam cost:

        colour mask (3 or 4 channels):  P = 10^6 * (G - R)
        greyscale mask (1 or 2 ch.):    P = 10^6 * 2 * (V - 128)

    where R, G are the red and green channels and V is the grey level (0-255).
    P > 0 protects the pixel (seams avoid it), P < 0 marks it for removal
    (seams are attracted to it), P = 0 leaves it unchanged. Only G = R (e.g.
    black, white, any grey) or V = 128 is neutral. Because of the 10^6 factor,
    even a one-level deviation from neutral outweighs the energy of a whole
    seam, so any non-neutral pixel acts as a strong constraint rather than a
    graded weight. To build a mask:
        - colour:    paint pure green (0,255,0) to protect, pure red
                     (255,0,0) to remove, and leave the rest black or grey;
        - greyscale: paint white (255) to protect, black (0) to remove, and
                     fill the rest with mid-grey exactly 128.
    Beware of anti-aliased or JPEG-compressed masks: slightly off-neutral
    pixels near the painted regions act as constraints too. Save masks as PNG.

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
            --dump-energy energy.png --dump-seams seams.png \
            --seam-order viridis --seam-on-grey

    (f) Relative target size: reduce the width to 75% and enlarge the height
        by 20%:
        ./build/seam_carving images/portal_orig.jpg out.png \
            --width-ratio 0.75 --height-ratio 1.2

    The exact commands producing every figure of the article are listed in
    images/README.md.


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
