import sys
from pathlib import Path
import tomli
import argparse
import shutil
from loguru import logger
from scipy.io import loadmat
import numpy as np


def collect_imaging_parameters(idx, imaging_parameters, mat_data, mylogger):
    """
    Collect the imaging parameters associated with the input DENSE data, check consistency across slices, and return for further processing.
    """
    
    seq_info = mat_data["SequenceInfo"]

    # If idx == 0, collect imaging parameters from the first slice
    if idx == 0:
        imaging_parameters = {
            "SliceThickness": np.array(seq_info["SliceThickness"][0][0]).flatten(),
            "Rows": np.array(seq_info["Rows"][0][0]).flatten(),
            "Columns": np.array(seq_info["Columns"][0][0]).flatten(),
            "PixelSpacing": np.array(seq_info["PixelSpacing"][0][0]).flatten(),
            "Frames": len(seq_info["DENSEindex"][0][0])
        }
        mylogger.info(f'Collected imaging parameters.')
    else:
        for key in imaging_parameters.keys():
            try:
                # Get the reference value from imaging_parameters
                ref_value = imaging_parameters[key]
                # Get the value for the current slice
                if key == "Frames":
                    comp_value = len(seq_info["DENSEindex"][0][0])
                else:
                    comp_value = np.array(seq_info[key][0][0]).flatten()
                # Compare comp_value to the reference value
                if not np.array_equal(comp_value, ref_value):
                    mylogger.warning(f'{key} for current slice differs from input imaging_parameters: {comp_value} vs {ref_value}')
                    imaging_parameters[key] = None
                else:
                    imaging_parameters[key] = comp_value
            except Exception as e:
                mylogger.error(f'Error accessing {key}: {e}')
                imaging_parameters[key] = None
    return imaging_parameters


def get_slice_location(mat_data, mylogger):
    """
    Get slice location for a given slice from the MATLAB data.

    Slice location should be calculated from ImagePositionPatient and ImageOrientationPatient header info stored in "SequenceInfo" of the MATLAB data.
    """
    try:
        seq_info = mat_data["SequenceInfo"]
        image_position = seq_info["ImagePositionPatient"][0][0]  # shape (3,)
        image_orientation = seq_info["ImageOrientationPatient"][0][0]  # shape (6,)

        row_cosines = np.array(image_orientation[:3]).flatten()
        col_cosines = np.array(image_orientation[3:]).flatten()
        normal = np.cross(row_cosines, col_cosines)

        # The slice location is the projection of IPP onto the normal vector
        slice_location = np.dot(np.array(image_position).flatten(), normal)

        return slice_location

    except KeyError as e:
        mylogger.error(f'Missing key in MATLAB data: {e}')
        sys.exit(1)


def flag_slice_location(mat_data, mylogger):
    """
    Determine if the slice location is inverted based on the calculated slice location and dicom header slice location.

    Check if the calculated slice location is equal within tolerance OR inverted within tolerance.

    Note: method could be improved - inversion causes issues with shear strain directions
    """
    try:
        sl_calculated = get_slice_location(mat_data, mylogger)
        sl_header = np.float64(mat_data["SequenceInfo"]["SliceLocation"][0][0])

        if np.isclose(sl_calculated, sl_header, atol=1e-3):
            return False
        elif np.isclose(sl_calculated, -sl_header, atol=1e-3):
            mylogger.warning('Inverted slice location detected.')
            return True
        else:
            mylogger.error(f'Calculated slice location is not equal/inverted to header: {sl_calculated} vs {sl_header}')
            sys.exit(1)

    except KeyError as e:
        mylogger.error(f'Missing key in MATLAB data: {e}')
        sys.exit(1)


def extract_contours(config, imaging_parameters, mat_data, mylogger):
    """
    Extract in-plane contours from DENSEanalysis data.

    Includes adjusting for half pixel shift, pixel spacing and adding slice location to define 3D coordinates.
    """

    # Get frame/s to be analysed
    if config["parameters"]["frames_to_fit"] == "single":
        # Get frame of interest from config
        frames = [config["parameters"]["frame_of_interest"]]
    elif config["parameters"]["frames_to_fit"] == "all":
        # Get all frames from config
        frames = list(range(0, imaging_parameters["Frames"]))

    # Get pixel spacing information
    pixel_spacing = imaging_parameters["PixelSpacing"]

    endo_contours = {}
    epi_contours = {}

    # Loop through frames to be analysed
    for frame_idx in frames:
        # Get the in-plane contours for the specified frame
        endo = mat_data["ROIInfo"]["Contour"][0][0][frame_idx][1]  # endo = second index [1]
        epi = mat_data["ROIInfo"]["Contour"][0][0][frame_idx][0]  # epi = first index [0]

        # Convert pixel coordinates to millimeters ([-] pixel shift - due to MATLAB one-indexing)
        endo[:, 0] = (endo[:, 0] - 0.5) * pixel_spacing[0]
        endo[:, 1] = (endo[:, 1] - 0.5) * pixel_spacing[1]
        epi[:, 0] = (epi[:, 0] - 0.5) * pixel_spacing[0]
        epi[:, 1] = (epi[:, 1] - 0.5) * pixel_spacing[1]

        # Add slice location to give 3D coordinates
        slice_location = get_slice_location(mat_data, mylogger)
        endo_contours[str(frame_idx)] = np.column_stack((endo, slice_location * np.ones(endo.shape[0])))
        epi_contours[str(frame_idx)] = np.column_stack((epi, slice_location * np.ones(epi.shape[0])))

    mylogger.info(f'Extracted {sum(arr.shape[0] for arr in endo_contours.values())} total endo and {sum(arr.shape[0] for arr in epi_contours.values())} total epi contour points from {len(frames)} frame/s.')
    return endo_contours, epi_contours


def extract_displacements(config, imaging_parameters, flagInverted, mat_data, mylogger):
    """
    Extract displacements from DENSEanalysis data.

    Includes defining locations from pixel rows and columns, adjusting for pixel size, inverting if necessary, and adding slice location to define 3D coordinates.
    """
    
    # Get frame/s to be analysed
    if config["parameters"]["frames_to_fit"] == "single":
        # Get frame of interest from config
        frames = [config["parameters"]["frame_of_interest"]]
    elif config["parameters"]["frames_to_fit"] == "all":
        # Get all frames from config
        frames = list(range(0, imaging_parameters["Frames"]))

    # Get pixel spacing information
    pixel_spacing = imaging_parameters["PixelSpacing"]

    # Get slice location information
    slice_location = get_slice_location(mat_data, mylogger)

    # Get image info multipliers
    multipliers = np.array(mat_data["ImageInfo"]["Multipliers"][0][0].flatten())

    locations = {}
    displacements = {}

    # Loop through frames to be analysed
    for frame_idx in frames:

        locs = []
        disps = []

        # Loop through pixel rows and columns to extract displacements
        for row in range(mat_data["ImageInfo"]["Xunwrap"][0][0][:,:,frame_idx].shape[0]):
            for col in range(mat_data["ImageInfo"]["Xunwrap"][0][0][:,:,frame_idx].shape[1]):

                # Skip pixel if NaN
                if np.isnan(mat_data["ImageInfo"]["Xunwrap"][0][0][:,:,frame_idx][row, col]):
                    continue

                # Locations

                # Convert pixel coordinates to millimeters ([+] pixel shift - due to Python zero-indexing)
                x_coord = (col + 0.5) * pixel_spacing[0]
                y_coord = (row + 0.5) * pixel_spacing[1]
                z_coord = slice_location

                # Append the 3D coordinates
                locs.append((x_coord, y_coord, z_coord))

                # Displacements

                # Get displacements from pixel values (in mm, apply flag)
                x_disp = pixel_spacing[0]*multipliers[0]*mat_data["ImageInfo"]["Xunwrap"][0][0][:,:,frame_idx][row, col]
                y_disp = pixel_spacing[1]*multipliers[1]*mat_data["ImageInfo"]["Yunwrap"][0][0][:,:,frame_idx][row, col]
                if flagInverted:
                    z_disp = -pixel_spacing[1]*multipliers[2]*mat_data["ImageInfo"]["Zunwrap"][0][0][:,:,frame_idx][row, col]
                else:
                    z_disp = pixel_spacing[1]*multipliers[2]*mat_data["ImageInfo"]["Zunwrap"][0][0][:,:,frame_idx][row, col]

                # Append the displacements
                disps.append((x_disp, y_disp, z_disp))


        locations[str(frame_idx)] = np.array(locs)
        displacements[str(frame_idx)] = np.array(disps)

    mylogger.info(f'Extracted {sum(arr.shape[0] for arr in displacements.values())} total displacements from {len(frames)} frame/s.')
    return locations, displacements


