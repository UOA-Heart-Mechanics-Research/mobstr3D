import sys
import numpy as np
import nibabel as nib
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pydicom
from pathlib import Path

from src.preprocessing.DICOM.index_inputs import find_dicom_for_nifti, find_nifti_for_mask, get_slice_location

def ray_march_endo(mask):
    """
    Perform ray marching to extract contour points from a binary mask - finding inside edge points for endocardium.
    Args:
        mask (np.ndarray): Binary mask of the structure.
    Returns:
        pts (np.ndarray): Extracted contour points.
    """
    # Compute center (image moments)
    M = cv2.moments(mask)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    center = (cx, cy)

    h, w = mask.shape
    num_angles = 360
    thetas = np.linspace(0, 2*np.pi, num_angles, endpoint=False)

    points = []  # will store contour points

    for t in thetas:
        # step size in pixels (small step = more accurate)
        for r in range(1, max(h, w)):
            x = int(cx + r * np.cos(t))
            y = int(cy + r * np.sin(t))

            # stop if we leave the image
            if x < 0 or x >= w or y < 0 or y >= h:
                break

            if mask[y, x] == 0:
                # last point inside the mask
                # use r-1 to get the boundary pixel
                # Add/subtract half pixel offset for endocardium depending on position from center
                if x < cx:
                    xr = int(cx + (r-1) * np.cos(t)) - 0.5
                else:
                    xr = int(cx + (r-1) * np.cos(t)) + 0.5
                if y < cy:
                    yr = int(cy + (r-1) * np.sin(t)) - 0.5
                else:
                    yr = int(cy + (r-1) * np.sin(t)) + 0.5
                points.append((xr, yr))
                break

    pts = np.array(points, dtype=np.float32)

    # Remove duplicate points
    pts = np.unique(pts, axis=0)

    # For each x, find the max and min y and remove other points
    for x in np.unique(pts[:, 0]):
        ys = pts[pts[:, 0] == x][:, 1]
        if len(ys) > 2:
            if len(ys[ys <= cy]) == 0:
                minmax_y = np.max(ys)
            else:
                minmax_y = np.max(ys[ys <= cy])
            if len(ys[ys >= cy]) == 0:
                maxmin_y = np.min(ys)
            else:
                maxmin_y = np.min(ys[ys >= cy])
            pts = pts[~((pts[:, 0] == x) & (pts[:, 1] != maxmin_y) & (pts[:, 1] != minmax_y))]
    # For each y, find the max and min x and remove other points
    for y in np.unique(pts[:, 1]):
        xs = pts[pts[:, 1] == y][:, 0]
        if len(xs) > 2:
            if len(xs[xs <= cx]) == 0:
                minmax_x = np.max(xs)
            else:
                minmax_x = np.max(xs[xs <= cx])
            if len(xs[xs >= cx]) == 0:
                maxmin_x = np.min(xs)
            else:
                maxmin_x = np.min(xs[xs >= cx])
            pts = pts[~((pts[:, 1] == y) & (pts[:, 0] != maxmin_x) & (pts[:, 0] != minmax_x))]

    return pts, center


def ray_march_epi(mask):
    """
    Perform ray marching to extract contour points from a binary mask - finding outside edge points for epicardium.
    Args:
        mask (np.ndarray): Binary mask of the structure.
    Returns:
        pts (np.ndarray): Extracted contour points.
    """
    # Compute center (image moments)
    M = cv2.moments(mask)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    center = (cx, cy)

    h, w = mask.shape
    num_angles = 360
    thetas = np.linspace(0, 2*np.pi, num_angles, endpoint=False)

    points = []  # will store contour points

    for t in thetas:
        # step size in pixels (small step = more accurate)
        for r in range(1, max(h, w)):
            x = int(cx + r * np.cos(t))
            y = int(cy + r * np.sin(t))

            # stop if we leave the image
            if x < 0 or x >= w or y < 0 or y >= h:
                break

            if mask[y, x] == 0:
                # last point inside the mask
                # use r-1 to get the boundary pixel
                # Add/subtract half pixel offset for epicardium depending on position from center
                if x < cx:
                    xr = int(cx + (r-1) * np.cos(t)) - 0.5
                else:
                    xr = int(cx + (r-1) * np.cos(t)) + 0.5
                if y < cy:
                    yr = int(cy + (r-1) * np.sin(t)) - 0.5
                else:
                    yr = int(cy + (r-1) * np.sin(t)) + 0.5
                points.append((xr, yr))
                break

    pts = np.array(points, dtype=np.float32)

    # For each x, find the max and min y and remove other points
    for x in np.unique(pts[:, 0]):
        ys = pts[pts[:, 0] == x][:, 1]
        if len(ys) > 2:
            max_y = np.max(ys)
            min_y = np.min(ys)
            pts = pts[~((pts[:, 0] == x) & (pts[:, 1] != max_y) & (pts[:, 1] != min_y))]
    # Remove duplicate points
    pts = np.unique(pts, axis=0)
    # For each y, find the max and min x and remove other points
    for y in np.unique(pts[:, 1]):
        xs = pts[pts[:, 1] == y][:, 0]
        if len(xs) > 2:
            max_x = np.max(xs)
            min_x = np.min(xs)
            pts = pts[~((pts[:, 1] == y) & (pts[:, 0] != max_x) & (pts[:, 0] != min_x))]

    return pts, center



def create_contours_from_labels(DENSE_series_index, mask, imaging_parameters, config, mylogger):
    """
    Creates endocardial and epicardial contours from a segmentation mask file.
    Args:
        mask (Path): Path to the segmentation mask NIfTI file.
            Assumes mask has labels:
                0: background
                1: myocardium
                2: bloodpool/lvcavity
        mylogger: Logger object for logging information.
    Returns:
        endo_contours (list): List of endocardial contours for each frame.
        epi_contours (list): List of epicardial contours for each frame.
    """

    endo_contours = []
    epi_contours = []

    # Load the segmentation mask
    seg_img = nib.load(str(mask))
    seg_data = seg_img.get_fdata().astype(np.uint8)

    # Break if mask is not 2D
    if seg_data.ndim != 2:
        mylogger.error(f'Segmentation mask "{mask}" has unexpected number of dimensions: {seg_data.ndim}. Expected 2D slice.')
        sys.exit(1)

    # Create binary masks for myocardium and bloodpool
    myocardium_mask = (seg_data == 1).astype(np.uint8) * 255
    bloodpool_mask = (seg_data == 2).astype(np.uint8) * 255

    # Extract contour points using ray marching
    endo_pts, endo_center = ray_march_endo(bloodpool_mask)
    epi_pts, epi_center = ray_march_epi(cv2.bitwise_or(myocardium_mask, bloodpool_mask))

    # Get slice index from mask filename "*_ID_'slice0''frame00'.nii.gz"
    slice_idx = int(mask.name.split("_")[3].split(".")[0][0])
    # Get frame index from mask filename "*_ID_'slice0''frame00'.nii.gz"
    frame_idx = int(mask.name.split("_")[3].split(".")[0][1:])


    # Debug: plot contours around mask
    if config["debug_flags"]["debug_contours"]:
        mylogger.info(f'Plotting contour overlays for debugging...')

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(seg_data, cmap='jet', alpha=1)
        ax[0].set_title('Segmentation Mask')
        ax[1].imshow(seg_data, cmap='jet', alpha=1)
        # Plot contours as circle, connected at the start and end
        if endo_pts is not None:
            endo_pts_closed = np.vstack([endo_pts, [endo_pts[0]]])
            x_endo = endo_pts_closed[:, 0]
            y_endo = endo_pts_closed[:, 1]
            ax[1].plot(x_endo, y_endo,'o',color='red', label='Endocardium')
        if epi_pts is not None:
            epi_pts_closed = np.vstack([epi_pts, [epi_pts[0]]])
            x_epi = epi_pts_closed[:, 0]
            y_epi = epi_pts_closed[:, 1]
            ax[1].plot(x_epi, y_epi, 'o', color='green', label='Epicardium')
        # Plot center used for ray marching
        ax[1].plot(endo_center[0], endo_center[1], 'x', color='yellow', label='Center_endo')
        ax[1].plot(epi_center[0], epi_center[1], 'x', color='cyan', label='Center_epi')
        # Create legend using proxy artists so labels appear
        handles = [
            mpatches.Patch(color='blue', label='0 = Background'),
            mpatches.Patch(color='green', label='1 = Myocardium'),
            mpatches.Patch(color='red', label='2 = Blood Pool'),
        ]
        ax[1].legend(handles=handles, loc='upper right', fontsize=8, framealpha=0.7)
        ax[1].set_title('Contour Overlay')
        plt.suptitle(f'{mask.name}')
        plt.show()


    # Debug: plot mask beside phase images (for confirming seed point selection frame)
    if config["debug_flags"]["debug_seedframe"] and int(frame_idx) == config["parameters"]["frame_of_seed"]:
        mylogger.info(f'Plotting segmentation mask alongside phase images (for confirming seed point selection frame: {config["parameters"]["frame_of_seed"]})...')

        fig, ax = plt.subplots(2, 2, figsize=(10, 10))

        # Get corresponding X, Y, Z phase DICOM files
        dicom = DENSE_series_index[f"Slice_{slice_idx}"]["MAG"][str(frame_idx)]
        dicom_xpha_file = DENSE_series_index[f"Slice_{slice_idx}"]["XPHA"][str(frame_idx)]
        dicom_ypha_file = DENSE_series_index[f"Slice_{slice_idx}"]["YPHA"][str(frame_idx)]
        dicom_zpha_file = DENSE_series_index[f"Slice_{slice_idx}"]["ZPHA"][str(frame_idx)]

        dicom_mag = pydicom.dcmread(str(dicom))
        mag_data = dicom_mag.pixel_array

        xpha_path = Path(dicom_xpha_file)
        dicom_xpha = pydicom.dcmread(str(xpha_path))
        xpha_data = dicom_xpha.pixel_array

        ypha_path = Path(dicom_ypha_file)   
        dicom_ypha = pydicom.dcmread(str(ypha_path))
        ypha_data = dicom_ypha.pixel_array

        zpha_path = Path(dicom_zpha_file)
        dicom_zpha = pydicom.dcmread(str(zpha_path))
        zpha_data = dicom_zpha.pixel_array

        # a. plot segmentation mask and 'MAG' image
        ax[0,0].imshow(mag_data, cmap='gray', alpha=1)
        ax[0,0].imshow(seg_data, cmap='jet', alpha=0.15)
        ax[0,0].set_title('Segmentation Mask')

        # b. plot XPHA
        ax[0,1].imshow(xpha_data, cmap='gray', alpha=1)
        ax[0,1].set_title('XPHA Image')
        # c. plot YPHA
        ax[1,0].imshow(ypha_data, cmap='gray', alpha=1)
        ax[1,0].set_title('YPHA Image')
        # d. plot ZPHA
        ax[1,1].imshow(zpha_data, cmap='gray', alpha=1)
        ax[1,1].set_title('ZPHA Image')

        # Create legend using proxy artists so labels appear
        handles = [
            mpatches.Patch(color='blue', label='0 = Background'),
            mpatches.Patch(color='green', label='1 = Myocardium'),
            mpatches.Patch(color='red', label='2 = Blood Pool'),
        ]
        ax[0,0].legend(handles=handles, loc='upper right', fontsize=8, framealpha=0.7)
        ax[0,0].set_title('Segmentation Overlay')
        plt.suptitle(f'{mask.name}')
        plt.show()


    # Define X and Y in mm
    endo_pts[:, 0] = (endo_pts[:, 0] + 0.5) * imaging_parameters["PixelSpacing"][0]
    endo_pts[:, 1] = (endo_pts[:, 1] + 0.5) * imaging_parameters["PixelSpacing"][1]

    epi_pts[:, 0] = (epi_pts[:, 0] + 0.5) * imaging_parameters["PixelSpacing"][0]
    epi_pts[:, 1] = (epi_pts[:, 1] + 0.5) * imaging_parameters["PixelSpacing"][1]

    # Get slice location from corresponding DICOM
    slice_location = get_slice_location(pydicom.dcmread(DENSE_series_index[f"Slice_{slice_idx}"]["MAG"][str(frame_idx)]), mylogger)

    # Add slice location as Z coordinate of contour points
    endo_pts = np.hstack([endo_pts.astype(np.float64), np.full((len(endo_pts),1), slice_location)])
    epi_pts = np.hstack([epi_pts.astype(np.float64), np.full((len(epi_pts),1), slice_location)])

    endo_contours = endo_pts
    epi_contours = epi_pts     

    return endo_contours, epi_contours

