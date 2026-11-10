# Command-line tests of the seam_carving program, run by ctest (and by the CI).
#
# Each test runs the program on an image of images/ and checks its console
# output with a regular expression: the reported output size for the resizing
# options, the confirmation message for the visualisation dumps, or the error
# message for invalid input. When PASS_REGULAR_EXPRESSION is set, ctest ignores
# the exit code, so error cases are tested through their message.
#
# The input is images/birds_orig.png (960x640) with small size changes, so the
# whole set runs in a few seconds.

set(SC_IMG  ${CMAKE_CURRENT_SOURCE_DIR}/images/birds_orig.png)
set(SC_MASK ${CMAKE_CURRENT_SOURCE_DIR}/images/birds_mask.png)
set(SC_OUT  ${CMAKE_CURRENT_BINARY_DIR}/cli_out)
file(MAKE_DIRECTORY ${SC_OUT})

# sc_cli_test(<name> <regex> <args...>): run the program, expect <regex> in its output.
function(sc_cli_test name regex)
    add_test(NAME cli_${name} COMMAND $<TARGET_FILE:seam_carving> ${ARGN})
    set_tests_properties(cli_${name} PROPERTIES PASS_REGULAR_EXPRESSION "${regex}")
endfunction()

# --- Target size: absolute and relative (--width-ratio / --height-ratio) ------
sc_cli_test(width_pixels        "Output dimensions: 955x640"
            ${SC_IMG} ${SC_OUT}/w955.png -w 955 -t 2)
sc_cli_test(width_ratio_reduce  "Output dimensions: 950x640"          # 0.99 * 960 = 950.4
            ${SC_IMG} ${SC_OUT}/wr099.png --width-ratio 0.99 -t 2)
sc_cli_test(height_ratio_enlarge "Output dimensions: 960x653"         # 1.02 * 640 = 652.8
            ${SC_IMG} ${SC_OUT}/hr102.png --height-ratio 1.02 -t 2)
sc_cli_test(ratio_one_unchanged "Output dimensions: 960x640"
            ${SC_IMG} ${SC_OUT}/r1.png --width-ratio 1 --height-ratio 1)
sc_cli_test(ratio_and_pixels_conflict "use either -w/--width or --width-ratio"
            ${SC_IMG} ${SC_OUT}/x.png -w 900 --width-ratio 0.9)
sc_cli_test(ratio_negative      "--height-ratio expects a positive number"
            ${SC_IMG} ${SC_OUT}/x.png --height-ratio -0.5)
sc_cli_test(ratio_not_a_number  "--width-ratio expects a positive number"
            ${SC_IMG} ${SC_OUT}/x.png --width-ratio half)

# --- Energy functions (--energy) and energy criteria ---------------------------
sc_cli_test(energy_gradient     "Output dimensions: 955x640"
            ${SC_IMG} ${SC_OUT}/eg.png -w 955 --energy gradient -t 2)
sc_cli_test(energy_hog          "Output dimensions: 955x640"
            ${SC_IMG} ${SC_OUT}/eh.png -w 955 --energy hog -t 2)
sc_cli_test(energy_entropy      "Output dimensions: 955x640"
            ${SC_IMG} ${SC_OUT}/ee.png -w 955 --energy entropy -t 2)
sc_cli_test(energy_hog_no_luma  "Output dimensions: 955x640"
            ${SC_IMG} ${SC_OUT}/ehn.png -w 955 --energy hog --no-luma -t 2)
sc_cli_test(energy_forward      "Output dimensions: 955x640"
            ${SC_IMG} ${SC_OUT}/ef.png -w 955 --forward -t 2)
sc_cli_test(energy_unknown      "Unknown energy function 'foo'"
            ${SC_IMG} ${SC_OUT}/x.png -w 955 --energy foo)
sc_cli_test(energy_with_forward "applies to backward energy only"
            ${SC_IMG} ${SC_OUT}/x.png -w 955 --energy entropy --forward)
sc_cli_test(norm_l1             "Output dimensions: 955x640"
            ${SC_IMG} ${SC_OUT}/nl1.png -w 955 --norm l1 -t 2)
sc_cli_test(norm_l1_forward_rgb "Output dimensions: 955x640"
            ${SC_IMG} ${SC_OUT}/nl1f.png -w 955 --norm l1 --forward --no-luma -t 2)
sc_cli_test(norm_unknown        "Unknown norm 'l3'"
            ${SC_IMG} ${SC_OUT}/x.png -w 955 --norm l3)
sc_cli_test(height_first        "Carving 2 height seam.*Carving 3 width seam.*Output dimensions: 957x638"
            ${SC_IMG} ${SC_OUT}/hf.png -w 957 -h 638 --height-first -t 2)
sc_cli_test(parallel_dp        "Output dimensions: 955x640"
            ${SC_IMG} ${SC_OUT}/pdp.png -w 955 --parallel-dp -t 2)

# --- Mask ----------------------------------------------------------------------
sc_cli_test(mask_removal        "Loaded mask:.*Output dimensions: 955x640"
            ${SC_IMG} ${SC_OUT}/m.png -w 955 -m ${SC_MASK} -t 2)

# --- Visualisation: clipping (--viz-clip) and seam order (--seam-order) --------
sc_cli_test(viz_clip_default    "Wrote energy map"
            ${SC_IMG} ${SC_OUT}/v.png --dump-energy ${SC_OUT}/energy99.png)
sc_cli_test(viz_clip_minmax     "Wrote energy map"
            ${SC_IMG} ${SC_OUT}/v.png --dump-energy ${SC_OUT}/energy100.png --viz-clip 100)
sc_cli_test(viz_clip_magma_hog  "Wrote energy map"
            ${SC_IMG} ${SC_OUT}/v.png --dump-energy ${SC_OUT}/energy_hog.png
            --energy hog --viz-colourmap magma --viz-clip 95)
sc_cli_test(viz_clip_zero       "--viz-clip expects a percentile in \\(0, 100\\]"
            ${SC_IMG} ${SC_OUT}/x.png --dump-energy ${SC_OUT}/e.png --viz-clip 0)
sc_cli_test(viz_clip_above_100  "--viz-clip expects a percentile in \\(0, 100\\]"
            ${SC_IMG} ${SC_OUT}/x.png --dump-energy ${SC_OUT}/e.png --viz-clip 101)
sc_cli_test(seam_order_vertical "Wrote seam overlay \\(5 vertical, 0 horizontal seams\\)"
            ${SC_IMG} ${SC_OUT}/v.png -w 955 --dump-seams ${SC_OUT}/seams_v.png
            --seam-order viridis --seam-on-grey -t 2)
sc_cli_test(seam_order_both     "Wrote seam overlay \\(3 vertical, 2 horizontal seams\\)"
            ${SC_IMG} ${SC_OUT}/v.png -w 957 -h 638 --dump-seams ${SC_OUT}/seams_vh.png
            --seam-order magma -t 2)
sc_cli_test(seam_order_unknown  "falling back to viridis"
            ${SC_IMG} ${SC_OUT}/v.png -w 958 --dump-seams ${SC_OUT}/seams_u.png
            --seam-order nosuchmap)
sc_cli_test(seam_colour         "Wrote seam overlay \\(2 vertical"
            ${SC_IMG} ${SC_OUT}/v.png -w 958 --dump-seams ${SC_OUT}/seams_c.png
            --seam-colour 0,0,255)

# --- Determinism: the result must not depend on the number of threads ----------
# Run the new energy functions with 1 and 4 threads and compare the output
# files byte by byte.
foreach(energy hog entropy)
    foreach(threads 1 4)
        add_test(NAME cli_threads_${energy}_${threads}
                 COMMAND $<TARGET_FILE:seam_carving> ${SC_IMG}
                         ${SC_OUT}/threads_${energy}_${threads}.png
                         -w 950 --energy ${energy} -t ${threads})
        set_tests_properties(cli_threads_${energy}_${threads} PROPERTIES
                             FIXTURES_SETUP threads_${energy})
    endforeach()
    add_test(NAME cli_threads_${energy}_identical
             COMMAND ${CMAKE_COMMAND} -E compare_files
                     ${SC_OUT}/threads_${energy}_1.png ${SC_OUT}/threads_${energy}_4.png)
    set_tests_properties(cli_threads_${energy}_identical PROPERTIES
                         FIXTURES_REQUIRED threads_${energy})
endforeach()
