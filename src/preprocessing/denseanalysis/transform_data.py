import sys
from pathlib import Path
import numpy as np




def transform_to_pseudo_cardiac_coordinates(config, imaging_parameters, endo_contours_ics, epi_contours_ics, locations_ics, displacements_ics):
    '''
    Transform the contours and displacements to a pseudo-cardiac coordinate system.

    For DENSEanalysis inputs, this coordinate system is defined with respect to the image plane.
    x and y = in-plane
    z = out-of-plane
    origin = top left pixel

    For fitting and analysis, we want these in the pseudo-cardiac coordinate system.
    x = long-axis - out-of-plane
    y and z = in-plane (Note: this could be improved by rotating with respect to the RV centroid)
    origin = LV centroid

    LV centroid determined from contours.
    '''

    # Get frame/s to be analysed
    if config["parameters"]["frames_to_fit"] == "single":
        # Get frame of interest from config
        frames = [config["parameters"]["frame_of_interest"]]
        fixed_frame = 0
    elif config["parameters"]["frames_to_fit"] == "all":
        # Get all frames from config
        frames = list(range(0, imaging_parameters["Frames"]))
        fixed_frame = config["parameters"]["frame_of_interest"]



    # Rotation

    # Define rotation matrix
    rotation_matrix = np.array([
    [0, 0, -1],  # new x = -old z
    [0, 1, 0],   # new y = old x
    [1, 0, 0]    # new z = old y
    ])

    nSlices = len(endo_contours_ics)
    nFrames = len(frames)

    # Initialise as dicts
    endo_contours_ccs = {}
    epi_contours_ccs = {}
    locations_ccs = {}
    displacements_ccs = {}

    # Loop through list of slices
    for slice_idx in range(nSlices):

        # Ensure dicts have initialized dicts for this slice
        slice_key = f"Slice_{slice_idx}"
        if slice_key not in locations_ccs:
            endo_contours_ccs[slice_key] = {}
            epi_contours_ccs[slice_key] = {}
            locations_ccs[slice_key] = {}
            displacements_ccs[slice_key] = {}

        # Access fixed frame of interest
        for frame_idx in frames:
            endo_contours_ccs[slice_key][str(frame_idx)] = endo_contours_ics[slice_key][str(frame_idx)] @ rotation_matrix.T
            epi_contours_ccs[slice_key][str(frame_idx)] = epi_contours_ics[slice_key][str(frame_idx)] @ rotation_matrix.T
            locations_ccs[slice_key][str(frame_idx)] = locations_ics[slice_key][str(frame_idx)] @ rotation_matrix.T
            displacements_ccs[slice_key][str(frame_idx)] = displacements_ics[slice_key][str(frame_idx)] @ rotation_matrix.T


    # Translation

    # Calculate LV centroid (w.r.t. endo contours)
    lv_centroids = [np.mean(endo_contours_ccs[f"Slice_{slice}"][str(fixed_frame)], axis=0) for slice in range(nSlices)]

    # Calculate mean for all slices
    lv_centroid = np.mean(lv_centroids, axis=0)

    # Translate contours and locations so LV centroid at origin (ONLY translate in-plane - y and z)
    for slice_idx in range(nSlices):

        slice_key = f"Slice_{slice_idx}"

        for frame_idx in frames:
            endo_contours_ccs[slice_key][str(frame_idx)][:, 1:3] -= lv_centroid[1:3]
            epi_contours_ccs[slice_key][str(frame_idx)][:, 1:3] -= lv_centroid[1:3]
            locations_ccs[slice_key][str(frame_idx)][:, 1:3] -= lv_centroid[1:3]
            # Displacements need no translation

    return endo_contours_ccs, epi_contours_ccs, locations_ccs, displacements_ccs