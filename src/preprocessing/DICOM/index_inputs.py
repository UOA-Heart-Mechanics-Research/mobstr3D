import sys
from pathlib import Path
import numpy as np
import pydicom
import json



def get_slice_location(dicom_data, mylogger):
    """
    Get slice location for a given slice from the DICOM data.

    Slice location should be calculated from ImagePositionPatient and ImageOrientationPatient header info stored in the DICOM header.
    """
    try:
        # Get Image Position Patient and Image Orientation Patient from nifty header data
        image_position = dicom_data.ImagePositionPatient
        image_orientation = dicom_data.ImageOrientationPatient

        row_cosines = np.array(image_orientation[:3]).flatten()
        col_cosines = np.array(image_orientation[3:]).flatten()
        normal = np.cross(row_cosines, col_cosines)

        # The slice location is the projection of IPP onto the normal vector
        slice_location = np.dot(np.array(image_position).flatten(), normal)

        return slice_location

    except KeyError as e:
        mylogger.error(f'Missing key in DICOM header data: {e}')
        sys.exit(1)


def flag_slice_location(dicom_data, mylogger):
    """
    Determine if the slice location is inverted based on the calculated slice location and dicom header slice location.

    Check if the calculated slice location is equal within tolerance OR inverted within tolerance.

    Note: method could be improved - inversion causes issues with shear strain directions
    """
    try:
        sl_calculated = get_slice_location(dicom_data, mylogger)
        sl_header = np.float64(dicom_data.SliceLocation)

        if np.isclose(sl_calculated, sl_header, atol=1e-3):
            return False
        elif np.isclose(sl_calculated, -sl_header, atol=1e-3):
            mylogger.warning('Inverted slice location detected.')
            return True
        else:
            mylogger.error(f'Calculated slice location is not equal/inverted to header: {sl_calculated} vs {sl_header}')
            sys.exit(1)

    except KeyError as e:
        mylogger.error(f'Missing key in DICOM header data: {e}')
        sys.exit(1)



def collect_imaging_parameters(idx, imaging_parameters, dicom_data, mylogger):
    """
    Collect the imaging parameters associated with the input DENSE data, check consistency across slices, and return for further processing.
    """

    # If idx == -1, collect imaging parameters from the first slice
    if idx == -1:

        # Break down ImageComments into DENSE specific parameters
        image_comments = dicom_data.ImageComments
        split_comments = image_comments.split(" ")
        encfreq = split_comments[5].split(":")[1]
        scale = split_comments[4].split(":")[1]
        frames = split_comments[9].split("/")[1]

        imaging_parameters = {
            "SliceThickness": np.array(dicom_data.SliceThickness).flatten(),
            "Rows": np.array(dicom_data.Rows).flatten(),
            "Columns": np.array(dicom_data.Columns).flatten(),
            "PixelSpacing": np.array(dicom_data.PixelSpacing).flatten(),
            "EncodingFrequency": np.float64(encfreq),
            "Scale": np.float64(scale),
            "Frames": int(frames),
            "ImageComments": image_comments,
            "DENSE": {
                "RCswap": split_comments[10].split(":")[1],
                "RCSflip": split_comments[11].split(":")[1]
            }
        }
        mylogger.info(f'Collected imaging parameters.')
    else:
        for key in imaging_parameters.keys():
            try:
                # Get the reference value from imaging_parameters
                ref_value = imaging_parameters[key]
                # Get the value for the current slice
                if key == "EncodingFrequency":
                    image_comments = dicom_data.ImageComments
                    split_comments = image_comments.split(" ")
                    encfreq = split_comments[5].split(":")[1]
                    comp_value = np.float64(encfreq)
                elif key == "Scale":
                    image_comments = dicom_data.ImageComments
                    split_comments = image_comments.split(" ")
                    scale = split_comments[4].split(":")[1]
                    comp_value = np.float64(scale)
                elif key == "Frames":
                    image_comments = dicom_data.ImageComments
                    split_comments = image_comments.split(" ")
                    frames = split_comments[9].split("/")[1]
                    comp_value = int(frames)
                elif key == "ImageComments":
                    continue  # Skip comparison for ImageComments
                elif key == "DENSE":
                    image_comments = dicom_data.ImageComments
                    split_comments = image_comments.split(" ")
                    rc_swap = split_comments[10].split(":")[1]
                    rc_sflip = split_comments[11].split(":")[1]
                    comp_value = {
                        "RCswap": rc_swap,
                        "RCSflip": rc_sflip
                    }
                else:
                    comp_value = np.array(getattr(dicom_data, key)).flatten()
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



def check_DENSE_3D(input_files, mylogger):
    """
    Check if the all of the input DICOM files are DENSE with 3D displacement encoding images (Mag + X Y Z phase).
    """

    # Initialise counter
    DENSE_image_count = 0

    # Index all input files
    file_index = {}
    frame_index = {}
    series_index = {}

    for file in input_files:

        issue = False
        
        # Load DICOM header
        ds = pydicom.dcmread(file, stop_before_pixels=True)
        
        # Get DENSE-specific ImageComments tag
        image_comments = getattr(ds, "ImageComments", None)

        # Check for Image Comments indicating DENSE
        if image_comments is None or "DENSE" not in str(image_comments):
            mylogger.info(f'File {file} is not recognised as DENSE/missing DENSE Image Comments (ImageComments=\"{image_comments}\").')
            issue = True
        
        file_index[file] = {
            "File": file,
            "ImageComments": image_comments,
            "SliceLocation": get_slice_location(ds, mylogger),
            "FrameIndex": getattr(ds, "InstanceNumber", None),
            "SeriesInstanceUID": getattr(ds, "SeriesInstanceUID", None)
        }

        if issue:
            continue
        else:
            DENSE_image_count += 1

    mylogger.info(f'{DENSE_image_count}/{len(input_files)} input files are recognized as DENSE DICOM images.')
    if DENSE_image_count != len(input_files):
        return False

    return True, file_index


def get_number_of_slices(file_index, mylogger):
    """
    Get the number of unique slices from the file index, by counting unique slice locations.
    """
    slice_locations = set()
    for file, info in file_index.items():
        slice_locations.add(info["SliceLocation"])
    nSlices = len(slice_locations)
    mylogger.info(f'Number of unique slices: {nSlices}')
    return nSlices


def get_number_of_frames(file_index, mylogger):
    """
    Get the number of unique frames from the file index, by looking for max FrameIndex.
    """
    frame_indices = set()
    for file, info in file_index.items():
        frame_indices.add(info["FrameIndex"])
    nFrames = len(frame_indices)
    mylogger.info(f'Number of frames: {nFrames}')
    return nFrames

def index_DENSE_series(file_index, nSlices, mylogger):
    """
    Index the DENSE series by SeriesInstanceUID and FrameIndex for easier access later.
    """
    series_index = {}
    for file, info in file_index.items():
        series_uid = info["SeriesInstanceUID"]
        frame_idx = info["FrameIndex"]
        if series_uid not in series_index:
            series_index[series_uid] = {}
        series_index[series_uid][frame_idx] = file

    # Sort the series_index by FrameIndex
    for series_uid in series_index:
        series_index[series_uid] = dict(sorted(series_index[series_uid].items()))

    # Index into DENSE series using ImageComments to identify one of each encoding type per slice (Mag, X, Y, Z)

    # Define slice mapping based on SliceLocation
    slice_map = {}
    for file, info in file_index.items():
        slice_location = info["SliceLocation"]
        if slice_location not in slice_map:
            slice_map[slice_location] = []
        slice_map[slice_location].append(file)
    slice_map = sorted(slice_map.keys(), reverse=True) 

    # Assign one of each encoding type per slice (Mag, X, Y, Z) based on SliceLocation
    DENSE_series_index = {}
    for slice_idx, slice_location in enumerate(slice_map):
        slice_files = [f for f in file_index if file_index[f]["SliceLocation"] == slice_location]
        slice_entry = {}
        for file in slice_files:
            series_uid = file_index[file]["SeriesInstanceUID"]
            image_comments = file_index[file]["ImageComments"].upper()
            if "MAG" in image_comments:
                slice_entry.setdefault("MAG", {})[int(file_index[file]["FrameIndex"]) - 1] = file
            elif "X-ENC PHA" in image_comments: #or "X PHASE" in image_comments:
                slice_entry.setdefault("XPHA", {})[int(file_index[file]["FrameIndex"]) - 1] = file
            elif "Y-ENC PHA" in image_comments:
                slice_entry.setdefault("YPHA", {})[int(file_index[file]["FrameIndex"]) - 1] = file
            elif "Z-ENC PHA" in image_comments:
                slice_entry.setdefault("ZPHA", {})[int(file_index[file]["FrameIndex"]) - 1] = file
        DENSE_series_index[f'Slice_{slice_idx}'] = slice_entry

    # Sort DENSE_series_index by frame indices within each encoding type
    for slice_key, slice_info in DENSE_series_index.items():
        for tag in slice_info:
            if isinstance(slice_info[tag], dict):
                DENSE_series_index[slice_key][tag] = dict(sorted(slice_info[tag].items()))

    
    mylogger.info(f'Indexed DENSE series by SeriesInstanceUID and FrameIndex.')
    return DENSE_series_index


def find_dicom_for_nifti(index_json_path, nifti_query, mylogger):

    nifti_query = Path(nifti_query)
    with open(index_json_path, 'r', encoding='utf-8') as fh:
        data = json.load(fh)

    matches = []
    for entry in data:
        nifti_entry = entry.get("nifti_file")
        if not nifti_entry:
            continue
        p = Path(nifti_entry)
        # match by full path or by filename
        if p == nifti_query or p.name == nifti_query.name:
            matches.append(entry.get("dicom_file"))

    if not matches:
        mylogger.error(f'No corresponding "MAG" DICOM file found for NIfTI file "{nifti_query}".')
        sys.exit(1)
    if len(matches) == 1:
        return matches[0]
    return matches  # return all matches

def find_phasedicom_for_nifti(index_json_path, nifti_query, mylogger):

    nifti_query = Path(nifti_query)
    with open(index_json_path, 'r', encoding='utf-8') as fh:
        data = json.load(fh)

    matches = []
    for entry in data:
        nifti_entry = entry.get("nifti_file")
        if not nifti_entry:
            continue
        p = Path(nifti_entry)
        # match by full path or by filename
        if p == nifti_query or p.name == nifti_query.name:
            matches.append(entry.get("dicom_file_phase"))

    if not matches:
        mylogger.error(f'No corresponding "XPHA" DICOM file found for NIfTI file "{nifti_query}".')
        sys.exit(1)
    if len(matches) == 1:
        return matches[0]
    return matches  # return all matches


def find_nifti_for_mask(index_json_path, mask_query, mylogger):

    mask_query = Path(mask_query)
    with open(index_json_path, 'r', encoding='utf-8') as fh:
        data = json.load(fh)

    matches = []
    for entry in data:
        mask_entry = entry.get("mask_file")
        if not mask_entry:
            continue
        p = Path(mask_entry)
        # match by full path or by filename
        if p == mask_query or p.name == mask_query.name:
            matches.append(entry.get("nifti_file"))

    if not matches:
        mylogger.error(f'No corresponding NIfTI file found for mask file "{mask_query}".')
        sys.exit(1)
    if len(matches) == 1:
        return matches[0]
    return matches  # return all matches
