import sys
import numpy as np


def compute_slice_correction_contour_centroid(config, imaging_parameters, endo_contours_ics, epi_contours_ics, locations_ics, displacements_ics, mylogger):

    # Assign frame for centroid calculation based on config
    if config["parameters"]["frames_to_fit"] == "all":
        frame = config["parameters"]["frame_of_interest"] # Default to specified frame - can be edited later if another method is desired
        mylogger.info(f'Adjusting based on contours from frame {frame} (edit config "frame_of_interest" to change)...')
        frames = range(len(endo_contours_ics[0]))
    elif config["parameters"]["frames_to_fit"] == "single":
        frame = config["parameters"]["frame_of_interest"]
        frames = range(frame, frame+1)
        mylogger.info(f'Adjusting based on contours from frame {frame}...')
    else:
        mylogger.error(f'Invalid frame selection: {config["parameters"]["frames_to_fit"]}')
        sys.exit(1)

    # Loop through slices
    nSlices = len(endo_contours_ics)
    slice_corrections_list = []
    for slice_idx in range(nSlices):

        # Get contours at specified frame
        epi_f = epi_contours_ics[slice_idx][frame][:, 0:2] # get inplane [x,y] only for centroid calculation

        if slice_idx == 0:
            # Compute centroid of epicardial contour at chosen frame, and assign as global centroid if first slice
            slice_centroid = epi_f.mean(axis=0)  # shape (2,)
            global_centroid = slice_centroid  # Store global centroid
            translation_vector = np.zeros_like(slice_centroid)
        else:
            # Compute centroid of epicardial contour at chosen frame
            slice_centroid = epi_f.mean(axis=0)  # shape (2,)

            # Compute translation vector
            translation_vector = global_centroid - slice_centroid  # shape (2,)

            # Apply same translation to all frames
            for f in frames:

                # Apply translation to contours
                endo_contours_ics[slice_idx][f][:, 0:2] += translation_vector
                epi_contours_ics[slice_idx][f][:, 0:2] += translation_vector

                # Apply translation to locations
                locations_ics[slice_idx][f][:, 0:2] += translation_vector
                # Displacements remain unchanged as they are relative

        # create per-slice dict and append to list
        temp_slice_corrections = {
            "slice_index": int(slice_idx),
            "slice_centroid": slice_centroid.tolist(),
            "translation_vector": translation_vector.tolist()
        }
        slice_corrections_list.append(temp_slice_corrections)

    slice_corrections = {
        "method": "contour_centroid",
        "frame": frame,
        "global_centroid": global_centroid.tolist(),
        "slice_corrections": slice_corrections_list
    }

    return endo_contours_ics, epi_contours_ics, locations_ics, displacements_ics, slice_corrections



def apply_slice_correction(config, imaging_parameters, endo_contours_ics, epi_contours_ics, locations_ics, displacements_ics, mylogger):
    """
    Applies slice correction to the contours and displacements based on the config.

    Parameters:
    - config: Configuration dictionary.
    - imaging_parameters: Imaging parameters dictionary.
    - endo_contours_ics: Endocardial contours in image coordinate system (x-y = inplane, z = through-plane).
    - epi_contours_ics: Epicardial contours in image coordinate system.
    - locations_ics: Locations in image coordinate system.
    - displacements_ics: Displacements in image coordinate system.
    - mylogger: Logger object for logging information.

    Returns:
    - Corrected endo_contours_ics, epi_contours_ics, locations_ics, displacements_ics.
    """

    mylogger.info(f'Applying slice correction: {config["preprocessing_slicecorrection"]["slice_correction_method"]}...')

    if config["preprocessing_slicecorrection"]["slice_correction_method"] == "contour_centroid":
        endo_contours_ics, epi_contours_ics, locations_ics, displacements_ics, slice_corrections = compute_slice_correction_contour_centroid(config, imaging_parameters, endo_contours_ics, epi_contours_ics, locations_ics, displacements_ics, mylogger)
        mylogger.info('Epicardial contour centroid slice correction applied. (Warning: this is a crude method and may not be suitable.)')
    elif config["preprocessing_slicecorrection"]["slice_correction_method"] == "bivme":
        # Placeholder
        mylogger.info('bivme slice correction a WIP. Change config to apply a valid method.')
        sys.exit(1)
    else:
        mylogger.warning(f'Invalid slice correction method: {config["preprocessing_slicecorrection"]["slice_correction_method"]}. No slice correction applied.')
        sys.exit(1)
    return endo_contours_ics, epi_contours_ics, locations_ics, displacements_ics, slice_corrections