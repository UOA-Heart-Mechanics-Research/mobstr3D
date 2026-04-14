import json
import numpy as np
import sys



def load_contour(file_path: str, mylogger):
    """
    Load endo and epi contours from a .json file.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
        if not check_contour_structure(data, mylogger):
            sys.exit(1)

    # Given data["key"] is a dict of frames, each frame is a dict of slices, each slice is a (n, 3) np.array
    endo = data["endo"]
    epi = data["epi"]

    return endo, epi


def load_disp(file_path: str, mylogger):
    """
    Load displacements from a .json file.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
        if not check_disp_structure(data, mylogger):
            sys.exit(1)

    # Given data["key"] is a dict of frames, each frame is a dict of slices, each slice is a (n, 3) np.array
    loc = data["locations"]
    disp = data["displacements"]
    
    return loc, disp


def get_template_parameters(endo_f, epi_f, config):
    """
    Create a template mesh from the endo and epi contours.

    Note: template mesh generated before rtz2xyz transformation!
    Hence,
    x, y = in-plane
    z = long axis
    """

    # Loop through slices and calculate radii - in-plane only
    endo_f = np.vstack([arr for arr in endo_f.values() if arr is not None])  # shape (total_points, 3)
    lv_endo_radii = np.linalg.norm(endo_f[:, 1:3], axis=1)

    epi_f = np.vstack([arr for arr in epi_f.values() if arr is not None])  # shape (total_points, 3)
    lv_epi_radii = np.linalg.norm(epi_f[:, 1:3], axis=1)

    # Define inner and outer radii
    inner_radius = lv_endo_radii.min()
    outer_radius = lv_epi_radii.max()

    # Define cylinder height
    cylinder_height = np.abs(epi_f[:, 0].max() - epi_f[:, 0].min()) + (2.0 * config["model_parameters"]["dist_to_data"])  # Add padding to ensure full coverage
    cylinder_bot = (epi_f[:, 0].min() - config["model_parameters"]["dist_to_data"])

    template_params = {
        "inner_radius": inner_radius,
        "outer_radius": outer_radius,
        "cylinder_height": cylinder_height,
        "cylinder_bot": cylinder_bot,
        "translation_vector": [0.0, 0.0, cylinder_bot]  # Translation vector
    }

    return template_params


def check_contour_structure(data, mylogger):
    min_slices = 3
    warn_slices = 4
    for key in ["endo", "epi"]:
        if not isinstance(data[key], dict):
            mylogger.error(f"'{key}' is not a dict - should be a dict of frames.")
            return False
        for frame in data[key]:
            if not isinstance(data[key][frame], dict):
                mylogger.error(f"'{key}[{frame}]' is not a dict - should be dict of slices.")
                return False
            num_slices = len(data[key][frame])
            if num_slices < min_slices:
                mylogger.error(f"Frame {frame} in '{key}' has only {num_slices} slices (minimum required: {min_slices}).")
                return False
            elif num_slices < warn_slices:
                mylogger.warning(f"Frame {frame} in '{key}' has only {num_slices} slices. Fewer than {warn_slices} can cause issues and require smoothing.")
            for slice in data[key][frame]:
                if data[key][frame][slice] is not None:
                    arr_np = np.array(data[key][frame][slice])
                    if arr_np.ndim != 2 or arr_np.shape[1] != 3:
                        mylogger.error(
                            f"'{key}[{frame}][{slice}]' does not have shape (n, 3)."
                        )
                        return False
    return True

def check_disp_structure(data, mylogger):
    min_slices = 3
    warn_slices = 4
    for key in ["locations", "displacements"]:
        if not isinstance(data[key], dict):
            mylogger.error(f"'{key}' is not a dict - should be a dict of frames.")
            return False
        for frame in data[key]:
            if not isinstance(data[key][frame], dict):
                mylogger.error(f"'{key}[{frame}]' is not a dict - should be dict of slices.")
                return False
            num_slices = len(data[key][frame])
            if num_slices < min_slices:
                mylogger.error(f"Frame {frame} in '{key}' has only {num_slices} slices (minimum required: {min_slices}).")
                return False
            elif num_slices < warn_slices:
                mylogger.warning(f"Frame {frame} in '{key}' has only {num_slices} slices. Fewer than {warn_slices} can cause issues and require smoothing.")
            for slice in data[key][frame]:
                if data[key][frame][slice] is not None:
                    arr_np = np.array(data[key][frame][slice])
                    if arr_np.ndim != 2 or arr_np.shape[1] != 3:
                        mylogger.error(
                            f"'{key}[{frame}][{slice}]' does not have shape (n, 3)."
                        )
                        return False
    return True