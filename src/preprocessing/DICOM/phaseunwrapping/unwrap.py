import numpy as np
import sys
import matplotlib.pyplot as plt
import json
from pathlib import Path
import pydicom
from scipy.signal import convolve2d
import nibabel as nib

from src.preprocessing.DICOM.index_inputs import get_slice_location



def calc_phase_quality_2D(w_phase, pixel_size, mask, connectivity, config, mylogger):
        
    """
    Phase Quality 2D

    Measured as the root-mean-square of the variances, within a local nxn region, of the partial derivatives in the x1, x2 and x3 directions.

    REFERENCE
    Quality-guided path following phase unwrapping:
    D. C. Ghiglia and M. D. Pritt, Two-Dimensional Phase Unwrapping:
    Theory, Algorithms and Software. New York: Wiley-Interscience, 1998.

    Repeated in:
    B. S. Spottiswoode, X. Zhong, A. T. Hess, C. M. Kramer,
    E. M. Meintjes, B. M. Mayosi, and F. H. Epstein,
    "Tracking Myocardial Motion From Cine DENSE Images Using
    Spatiotemporal Phase Unwrapping and Temporal Fitting,"
    Medical Imaging, IEEE Transactions on, vol. 26, pp. 15, 2007.

    Inputs:
    w_phase : 2D array [MxN]
        Wrapped phase image to calculate quality from.
    pixel_size : array-like [pxlx, pxly]
        Pixel size in each dimension (mm).
    mask : 2D array [MxN] logical
        Logical mask specifying valid pixels for quality calculation.
    connectivity : int
        Connectivity for neighbourhood (4 or 8).
    
    Outputs:
    map_quality : 2D array [0,1]
        Phase quality map.
    """

    # Normalise wrapped phase image from [0, max(w_phase)] to [-pi, pi]
    w_phase = (w_phase / np.max(w_phase)) * (2 * np.pi) - np.pi

    # Define search neighbourhood size based on connectivity
    if connectivity == 4:
        h = np.array([[0, 1, 0],
                      [1, 0, 1],
                      [0, 1, 0]])
    elif connectivity == 8:
        h = np.array([[1, 1, 1],
                      [1, 0, 1],
                      [1, 1, 1]])
    else:
        mylogger.error('Connectivity must be 4 or 8.')
        sys.exit(1)
    
    # Index neighbourhood offsets
    nhood = np.argwhere(h == 1) - 1  #subtract 1 to center
    
    # Define distance weights based on pixel size
    weight = nhood*np.array([pixel_size[0], pixel_size[1]])
    weight = 1/np.sqrt(np.sum(weight**2, axis=1))

    # Index voxels within mask
    inds = np.argwhere(mask)

    # Get indexed voxel coordinates
    # inds is (N,2): (row, col)
    I = inds[:, 0]
    J = inds[:, 1]

    # Initalise gradients and distance weights
    G = np.zeros((len(inds), len(nhood)))
    W = np.zeros((len(inds), len(nhood)))

    # Loop through neighbourhood
    for n in range(len(nhood)):
        # Get neighbour coordinates
        I_n = I - nhood[n, 0]
        J_n = J - nhood[n, 1]
        ind_n = I_n * w_phase.shape[1] + (J_n) #linear index of neighbour

        # Check neighbour is within image bounds
        valid = (I_n >= 0) & (I_n < w_phase.shape[0]) & (J_n >= 0) & (J_n < w_phase.shape[1])

        # Check neighbour is within mask
        # Safely check neighbour is within mask by only indexing mask at in-range positions
        mask_valid = np.zeros_like(valid, dtype=bool)
        if np.any(valid):
            mask_valid[valid] = mask[I_n[valid], J_n[valid]]
        valid = valid & mask_valid
        
        # Store gradient and weight
        G[valid, n] = w_phase[I_n[valid], J_n[valid]] - w_phase[I[valid], J[valid]]
        W[valid, n] = weight[n]

    # Calculate locally unwrapped gradients
    G = (G + np.pi) % (2 * np.pi) - np.pi #wrap to [-pi, pi]
    # Calculate normalized weights
    wsum = np.sum(W, axis=1, keepdims=True)
    wsum[wsum == 0] = 1.0
    W = W / wsum
    # Calculate weighted mean of gradients
    Gmean = np.sum(W * G, axis=1)
    # Calculate weighted variance of gradients
    G = G - Gmean[:, np.newaxis]
    Gvar = np.sum(W * G**2, axis=1)

    # Calculate phase quality map
    qual = np.full(w_phase.shape, np.nan)
    qual[mask] = np.exp(-np.sqrt(Gvar))

    # Debug: plot map of phase quality
    if config["debug_flags"]["debug_phasequality"]:
        mylogger.info('Plotting phase unwrapping phase quality maps for debugging...')
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        im0 = axs[0].imshow(w_phase, cmap='gray')
        plt.colorbar(im0, ax=axs[0])
        axs[0].set_title('Wrapped Phase Image')

        im1 = axs[1].imshow(qual, cmap='gray', vmin=0, vmax=1)
        plt.colorbar(im1, ax=axs[1])
        axs[1].set_title('Phase Quality Map')

        plt.suptitle('Initial Phase Quality Calculation [DEBUG]')
        plt.show()

    return qual


def quickfind(adjoin, qual_idx):
    """
    Find index of highest-quality pixel that is True in adjoin.
    qual_idx must be sorted descending by quality.
    """
    adjoin_flat = adjoin.ravel()

    for idx in qual_idx:
        if adjoin_flat[idx]:
            return idx

    return None  # no valid pixel found


def unwrap_2element_scalar(p0, p1):
    dp = p1 - p0
    dps = dp - np.floor((dp + np.pi) / (2 * np.pi)) * (2 * np.pi)

    if dps == -np.pi and dp > 0:
        dps = np.pi

    if abs(dp) < np.pi:
        return p1

    return p1 + (dps - dp)



def unwrap_phase_2d_floodfill(w_phase, args, config, mylogger, seed_point=None):
    
    """
    Phase Unwrapping using Flood-Fill Algorithm

    REFERENCE
    Quality-guided path following phase unwrapping:
    D. C. Ghiglia and M. D. Pritt, Two-Dimensional Phase Unwrapping:
    Theory, Algorithms and Software. New York: Wiley-Interscience, 1998.

    Repeated in:
    B. S. Spottiswoode, X. Zhong, A. T. Hess, C. M. Kramer,
    E. M. Meintjes, B. M. Mayosi, and F. H. Epstein,
    "Tracking Myocardial Motion From Cine DENSE Images Using
    Spatiotemporal Phase Unwrapping and Temporal Fitting,"
    Medical Imaging, IEEE Transactions on, vol. 26, pp. 15, 2007.
    """

    ### SETUP ###

    # Check input phase image is 2D
    if w_phase.ndim != 2:
        mylogger.error('Input phase image must be 2D.')
        sys.exit(1)

    # Define mask
    mask = args["mask"]

    # Convert mask to logical
    if mask is not None:
        # Assume mask is 0 = background, 1 = myocardium, 2 = bloodpool, keep 1 = myocardium as true
        mask = (mask == 1)
    else:
        # Specify entire image as valid
        mask = np.ones(w_phase.shape, dtype=bool)

    # Check mask size matches phase image
    if mask.shape != w_phase.shape:
        mylogger.error('Mask size does not match phase image size.')
        breakpoint()
        sys.exit(1)
    

    ### CALCULATE PHASE QUALITY ###
    qual = calc_phase_quality_2D(w_phase, args["pixel_size"], mask, args["connectivity"], config, mylogger)

    ### SELECT SEED POINT ###
    if seed_point is None:
        if args["seed"] == 'auto':
            # Find maximum quality point within mask
            qual_masked = np.copy(qual)
            qual_masked[~mask] = -np.inf
            seed_idx = np.unravel_index(np.argmax(qual_masked), qual_masked.shape)
            seed_point = (seed_idx[0], seed_idx[1])  # (row, col)
            mylogger.info(f'Automatically selected seed point at {(int(round(seed_point[1])), int(round(seed_point[0])))} with quality {qual[seed_point]:.4f}.')
        elif args["seed"] == 'manual':
            # Plot phase quality map for manual seed selection
            fig, ax = plt.subplots()
            im = ax.imshow(qual, cmap='viridis', vmin=0, vmax=1)
            plt.colorbar(im, ax=ax)
            ax.set_title('Phase Quality Map - Select Seed Point')
            seed_point = plt.ginput(1)[0]  # Get one point from user
            seed_point = (int(round(seed_point[1])), int(round(seed_point[0])))  # Convert to (row, col)
            plt.close(fig)
            mylogger.info(f'Manually selected seed point at {seed_point} with quality {qual[seed_point]:.4f}.')
        else:
            mylogger.error('Seed selection method must be "auto" or "manual".')
            sys.exit(1)

    ### UNWRAP PHASE USING FLOOD-FILL ALGORITHM ###

    # Intialise unwrapped phase seed point
    uw_phase = np.full(w_phase.shape, np.nan)
    uw_phase[seed_point] = w_phase[seed_point]

    # Only consider pixels within mask
    uw_phase[~mask] = np.nan

    # Logical map of unwrapped pixels
    is_unwrap = ~np.isnan(uw_phase)

    # Check that seed point is within mask
    if not mask[seed_point[0], seed_point[1]]:
        mylogger.error('Seed point is outside of the mask. No valid unwrapped phase pixel exists to start flood-fill unwrapping.')
        sys.exit(1)
    
    # Define search neighbourhood size based on connectivity
    connectivity = args["connectivity"]
    if connectivity == 4:
        h = np.array([[0, 1, 0],
                      [1, 0, 1],
                      [0, 1, 0]])
    elif connectivity == 8:
        h = np.array([[1, 1, 1],
                      [1, 0, 1],
                      [1, 1, 1]])
    else:
        mylogger.error('Connectivity must be 4 or 8.')
        sys.exit(1)

    # Index neighbourhood offsets
    offsets = np.argwhere(h == 1) - 1  #subtract 1 to center
    nhood_ioffset = offsets[:, 0]
    nhood_joffset = offsets[:, 1]

    # Quality multiplier: prioritise neighbours that are closer to the current pixel
    px = float(args["pixel_size"][0])
    py = float(args["pixel_size"][1])
    distances = np.sqrt((nhood_ioffset * px) ** 2 + (nhood_joffset * py) ** 2)
    qual_factor = 1.0 / distances

    # Define adjoin matrix - finding neighbours not yet unwrapped
    tmp = convolve2d(is_unwrap.astype(float), h, mode='same')
    adjoin = (tmp.astype(bool)) & (~is_unwrap) & mask

    # Sort quality map indices in descending order (once for efficiency)
    qual_idx = np.argsort(qual.ravel())[::-1]
    qual_idx = qual_idx.astype(np.uint32)

    # Debug: prepare persistent plot to update each iteration
    fig = None
    ax = None
    im = None
    if config["debug_flags"]["debug_phaseunwrapping"]:
        mylogger.info('Plotting phase unwrapping progress for debugging...')
        plt.ion()
        fig, ax = plt.subplots(1, 2, figsize=(10, 6))
        im = ax[0].imshow(uw_phase, cmap='gray')
        fig.colorbar(im, ax=ax[0])
        ax[0].set_title('Phase Unwrapping Progress [DEBUG]')
        # Define new image for progress quality map
        # Background set to black, masked pixels set to blue, adjoin pixels set to red, unwrapped pixels set to white
        uw_progress_img = np.zeros((*w_phase.shape, 3), dtype=np.float32)
        uw_progress_img[:, :, 0] = adjoin.astype(np.float32) * 1.0  # Red channel for adjoin
        uw_progress_img[:, :, 1] = is_unwrap.astype(np.float32) * 1.0  # Green channel for unwrapped
        uw_progress_img[:, :, 2] = (~mask).astype(np.float32) * 0.5  # Blue channel for masked
        im_p = ax[1].imshow(uw_progress_img)
        fig.colorbar(im_p, ax=ax[1])
        ax[1].set_title('Phase Quality Map [DEBUG]')
        plt.show()

    # Flood-fill unwrapping loop, loop until no adjoined pixels remain
    while np.any(adjoin):

        # Find index of highest quality adjoined pixel
        idx = quickfind(adjoin, qual_idx)

        # Linear → (i0, j0)
        i0 = idx % w_phase.shape[0]
        j0 = idx // w_phase.shape[0]

        # Neighbours
        i = i0 + nhood_ioffset
        j = j0 + nhood_joffset

        # Valid neighbours
        valid = (
            (i >= 0) & (i < w_phase.shape[0]) &
            (j >= 0) & (j < w_phase.shape[1])
        )

        i_v = i[valid]
        j_v = j[valid]

        # Linear indices
        nhood = i_v + w_phase.shape[0] * j_v
        fac = qual_factor[valid]

        # Unwrapped neighbours
        is_unwrap_flat = is_unwrap.ravel()
        tf = is_unwrap_flat[nhood]

        nhood_unwrapped = nhood[tf]
        fac = fac[tf]

        num = len(nhood_unwrapped)

        if num == 0:
            adjoin.ravel()[idx] = False
            continue

        elif num == 1:
            phase_ref = uw_phase.ravel()[nhood_unwrapped[0]]
            p = unwrap_2element_scalar(phase_ref, w_phase.ravel()[idx])
            #print(f'Unwrapped value: {p} - for phase ref {phase_ref}, and wrapped value {w_phase.ravel()[idx]}')

        else:
            f = fac * qual.ravel()[nhood_unwrapped]
            nhood_max = nhood_unwrapped[np.argmax(f)]
            phase_ref = uw_phase.ravel()[nhood_max]
            p = unwrap_2element_scalar(phase_ref, w_phase.ravel()[idx])

        # Unwrap pixel
        uw_phase.ravel()[idx] = p

        # Update is_unwrap map
        is_unwrap.ravel()[idx] = True

        # Update adjoin matrix
        adjoin.ravel()[idx] = False
        adjoin.ravel()[nhood] = (~is_unwrap.ravel()[nhood]) & mask.ravel()[nhood]

        # Debug: update persistent unwrapped phase plot (if enabled)
        if config["debug_flags"]["debug_phaseunwrapping"] and im is not None:
            im.set_data(uw_phase)
            # Define new image for progress quality map
            # Background set to black, masked pixels set to blue, adjoin pixels set to red, unwrapped pixels set to white
            uw_progress_img = np.zeros((*w_phase.shape, 3), dtype=np.float32)
            uw_progress_img[:, :, 0] = adjoin.astype(np.float32) * 1.0  # Red channel for adjoin
            uw_progress_img[:, :, 1] = is_unwrap.astype(np.float32) * 1.0  # Green channel for unwrapped
            uw_progress_img[:, :, 2] = (~mask).astype(np.float32) * 0.5  # Blue channel for masked
            im_p.set_data(uw_progress_img)
            # Update color limits if needed to keep contrast reasonable
            finite = np.isfinite(uw_phase)
            if np.any(finite):
                im.set_clim(np.nanmin(uw_phase[finite]), np.nanmax(uw_phase[finite]))
            fig.canvas.draw_idle()
            plt.pause(0.01)

    
    # Confirm all masked values were unwrapped
    if any(mask.ravel() != ~np.isnan(uw_phase.ravel())):
        mylogger.warning('Not all masked pixels were unwrapped.')

    # Debug: close persistent plot
    if config["debug_flags"]["debug_phaseunwrapping"] and fig is not None:
        plt.ioff()
        plt.close(fig)

    return seed_point, qual, uw_phase




def unwrap_phase_3d_floodfill(w_phase_3D, args, config, mylogger, seed_point_3D=None):
    
    """
    Phase Unwrapping using Flood-Fill Algorithm

    REFERENCE
    Quality-guided path following phase unwrapping:
    D. C. Ghiglia and M. D. Pritt, Two-Dimensional Phase Unwrapping:
    Theory, Algorithms and Software. New York: Wiley-Interscience, 1998.

    Repeated in:
    B. S. Spottiswoode, X. Zhong, A. T. Hess, C. M. Kramer,
    E. M. Meintjes, B. M. Mayosi, and F. H. Epstein,
    "Tracking Myocardial Motion From Cine DENSE Images Using
    Spatiotemporal Phase Unwrapping and Temporal Fitting,"
    Medical Imaging, IEEE Transactions on, vol. 26, pp. 15, 2007.
    """

    ### SETUP ###

    # Check input phase image is 3D
    if w_phase_3D.ndim != 3:
        mylogger.error('Input phase image for "unwrap_phase_3d_floodfill" must be 3D.')
        sys.exit(1)

    # Define mask
    mask_3D = args["mask_3D"]

    # Convert mask to logical
    if mask_3D is not None:
        # Assume mask is 0 = background, 1 = myocardium, 2 = bloodpool, keep 1 = myocardium as true
        mask_3D = (mask_3D == 1)
    else:
        # Specify entire image as valid
        mask_3D = np.ones(w_phase_3D.shape, dtype=bool)

    # Check mask size matches phase image
    if mask_3D.shape != w_phase_3D.shape:
        mylogger.error('Mask size does not match phase image size.')
        sys.exit(1)
    
    # Define frames to be unwrapped
    frames = range(config["parameters"]["frame_of_seed"], config["parameters"]["frame_of_seed"] + w_phase_3D.shape[2])

    # Predefine 3D qual array
    qual_3D = np.full(w_phase_3D.shape, 0.0)


    ### CALCULATE PHASE QUALITY ###

    # Loop through frames
    for frame_idx, frame in enumerate(frames):

        # Extract 2D phase and mask for current frame
        w_phase_2D = w_phase_3D[:, :, frame_idx]
        mask_2D = mask_3D[:, :, frame_idx]

        # Convert mask to logical
        if mask_2D is not None:
            # Assume mask is 0 = background, 1 = myocardium, 2 = bloodpool, keep 1 = myocardium as true
            mask_2D = (mask_2D == 1)
        else:
            # Specify entire image as valid
            mask_2D = np.ones(w_phase_2D.shape, dtype=bool)

        # Calculate phase quality for current frame
        qual = calc_phase_quality_2D(w_phase_2D, args["pixel_size"], mask_2D, args["connectivity"], config, mylogger)

        # Store unwrapped 2D qual back into 3D array
        qual_3D[:, :, frame_idx] = qual


    ### SELECT SEED POINT ###

    if seed_point_3D is None:
        
        # Get first frame for seed selection - index [0]
        qual = qual_3D[:, :, 0]
        mask = mask_3D[:, :, 0]
        
        if args["seed"] == 'auto':
            # Find maximum quality point within mask
            qual_masked = np.copy(qual)
            qual_masked[~mask] = -np.inf
            seed_idx = np.unravel_index(np.argmax(qual_masked), qual_masked.shape)
            seed_point = (int(round(seed_idx[1])), int(round(seed_idx[0])))  # Convert to (row, col)
            mylogger.info(f'Automatically selected seed point at {seed_point} on frame {frames[0]} with quality {qual[seed_point]:.4f}.')
        elif args["seed"] == 'manual':
            # Plot phase quality map for manual seed selection
            fig, ax = plt.subplots()
            im = ax.imshow(qual, cmap='viridis', vmin=0, vmax=1)
            plt.colorbar(im, ax=ax)
            ax.set_title('Phase Quality Map - Select Seed Point')
            seed_point = plt.ginput(1)[0]  # Get one point from user
            seed_point = (int(round(seed_point[1])), int(round(seed_point[0])))  # Convert to (row, col)
            plt.close(fig)
            mylogger.info(f'Manually selected seed point at {seed_point} on frame {frames[0]} with quality {qual[seed_point]:.4f}.')
        else:
            mylogger.error('Seed selection method must be "auto" or "manual".')
            sys.exit(1)

        seed_point_3D = (seed_point[0], seed_point[1], 0)  # (row, col, frame_index=0)

    else:
        seed_point = (seed_point_3D[0][0], seed_point_3D[0][1])  # (row, col)
    
    # Setup storage for seed points for all frames
    all_seed_points = [None] * len(frames)
    all_seed_points[0] = [seed_point[0], seed_point[1]]


    ### INITIAL UNWRAP PHASE USING 3D FLOOD-FILL ALGORITHM ###

    # Intialise unwrapped phase seed point
    temp_uw_phase_3D = np.full(w_phase_3D.shape, np.nan)
    temp_uw_phase_3D[seed_point[0], seed_point[1], 0] = w_phase_3D[seed_point[0], seed_point[1], 0]

    # Only consider pixels within mask
    temp_uw_phase_3D[~mask_3D] = np.nan

    # Logical map of unwrapped pixels
    is_unwrap = ~np.isnan(temp_uw_phase_3D)

    # Check that seed point is within mask
    if not mask_3D[seed_point[0], seed_point[1], 0]:
        mylogger.error('Seed point is outside of the mask. No valid unwrapped phase pixel exists to start flood-fill unwrapping.')
        sys.exit(1)

    # Check that the seed point has been unwrapped
    if not is_unwrap[seed_point[0], seed_point[1], 0]:
        mylogger.error('Seed point has not been unwrapped. Cannot start flood-fill unwrapping.')
        sys.exit(1)


    # Predefine adjoin and islocal arrays
    adjoin = np.zeros(w_phase_3D.shape, dtype=bool)
    islocal = np.zeros(w_phase_3D.shape, dtype=bool)

    # Get searchradius from args
    searchradius = args["searchradius"]

    # Define local region around seed point
    I, J = np.meshgrid(
        np.arange(-searchradius, searchradius+1),
        np.arange(-searchradius, searchradius+1),
        indexing='ij'   # IMPORTANT: matches ndgrid, not meshgrid default
    )

    D = I**2 + J**2
    hlocal = D <= searchradius**2

    i, j = np.nonzero(hlocal)
    local_ioffset = i - searchradius
    local_joffset = j - searchradius

    
    # Define search neighbourhood size based on connectivity
    connectivity = args["connectivity"]
    if connectivity == 4:
        hnhood = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ], dtype=bool)

        hnhood_time = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ], dtype=bool)

    elif connectivity == 8:
        hnhood = np.array([
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ], dtype=bool)

        hnhood_time = np.array([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]
        ], dtype=bool)
    else:
        mylogger.error('Connectivity must be 4 or 8.')
        sys.exit(1)

    # Index spatial neighbourhood offsets
    offsets = np.argwhere(hnhood == 1) - 1  #subtract 1 to center
    nhood_ioffset = offsets[:, 0]
    nhood_joffset = offsets[:, 1]

    # Index temporal neighbourhood offsets
    offsets = np.argwhere(hnhood_time == 1) - 1  #subtract 1 to center
    nhood_time_ioffset = offsets[:, 0]
    nhood_time_joffset = offsets[:, 1]

    # Quality multiplier: prioritise neighbours that are closer to the current pixel
    px = float(args["pixel_size"][0])
    py = float(args["pixel_size"][1])
    distances = np.sqrt((nhood_ioffset * px) ** 2 + (nhood_joffset * py) ** 2)
    qual_factor = 1.0 / distances

    # Define range of frames to iterate through
    frseed = 0  # seed frame index

    # Note: we need to iterate both forwards and backwards from the seed frame (if seed frame is not on first or last frame)
    frrng = [
        np.arange(frseed, len(frames)),     # forward
        np.arange(frseed, -1, -1)           # backward
        ]

    # Loop through frame directions
    for ci in range(len(frrng)):

        # Initialise starting point as the seed point
        i0 = seed_point[0]
        j0 = seed_point[1]
        
        # Loop through frames in current direction
        for fi in range(len(frrng[ci]) - 1):

            flag_nextframe = False

            # Current frame index
            fr0 = frrng[ci][fi]
            fr1 = frrng[ci][fi + 1]

            # Determine pixels in search region around current starting point
            i = i0 + local_ioffset
            j = j0 + local_joffset
            valid = (
                (i >= 0) & (i < w_phase_3D.shape[0]) &
                (j >= 0) & (j < w_phase_3D.shape[1])
            )
            i = i[valid]
            j = j[valid]

            # Search mask
            tmp = np.zeros(w_phase_3D.shape[0:2], dtype=bool)
            tmp[i, j] = True

            islocal[:] = False
            islocal[:, :, fr0] = tmp
            islocal[:, :, fr1] = tmp

            islocal &= mask_3D

            # Check for overlapping search region after masking
            tmp = convolve2d(
                islocal[:, :, fr0].astype(float),
                hnhood_time,
                mode='same'
            )

            overlap = (tmp > 0) & islocal[:, :, fr1]

            if not np.any(overlap):
                mylogger.warning(f'No overlapping search region found between frames {frames[fr0]} and {frames[fr1]}. Stopping unwrapping in this direction.')
                break
            
            # Intialise spatial adjoin for current frame pair
            tmp = convolve2d(
                is_unwrap[:, :, fr0].astype(float),
                hnhood,
                mode='same'
            )
            ad0 = (tmp > 0) & (~is_unwrap[:, :, fr0]) & islocal[:, :, fr0]

            tmp = convolve2d(
                is_unwrap[:, :, fr0].astype(float),
                hnhood_time,
                mode='same'
            )
            ad1 = (tmp > 0) & islocal[:, :, fr1]

            # Fill adjoin for current frame pair
            adjoin[:] = False
            adjoin[:, :, fr0] = ad0
            adjoin[:, :, fr1] = ad1

            # Debug: prepare persistent plot to update each iteration
            fig = None
            ax = None
            im = None
            if config["debug_flags"]["debug_phaseunwrapping"]:
                mylogger.info('Plotting phase unwrapping progress for debugging...')
                plt.ion()
                fig, ax = plt.subplots(1, 3, figsize=(15, 6))
                im = ax[0].imshow(temp_uw_phase_3D[:, :, fr0], cmap='gray')
                fig.colorbar(im, ax=ax[0])
                ax[0].set_title('Phase Unwrapping Progress [DEBUG]')
                # Define new image for progress quality map
                # Background set to black, masked pixels set to blue, adjoin pixels set to red, unwrapped pixels set to white
                uw_progress_img_fr0 = np.zeros((*w_phase_3D.shape[0:2], 3), dtype=np.float32)
                uw_progress_img_fr0[:, :, 0] = adjoin[:, :, fr0].astype(np.float32) * 1.0  # Red channel for adjoin
                uw_progress_img_fr0[:, :, 1] = is_unwrap[:, :, fr0].astype(np.float32) * 1.0  # Green channel for unwrapped
                uw_progress_img_fr0[:, :, 2] = (~mask_3D[:, :, fr0]).astype(np.float32) * 0.5  # Blue channel for masked
                im_p0 = ax[1].imshow(uw_progress_img_fr0)
                fig.colorbar(im_p0, ax=ax[1])
                ax[1].set_title('Phase Quality Map [DEBUG]')
                # Define new image for progress quality map
                # Background set to black, masked pixels set to blue, adjoin pixels set to red, unwrapped pixels set to white
                uw_progress_img_fr1 = np.zeros((*w_phase_3D.shape[0:2], 3), dtype=np.float32)
                uw_progress_img_fr1[:, :, 0] = adjoin[:, :, fr1].astype(np.float32) * 1.0  # Red channel for adjoin
                uw_progress_img_fr1[:, :, 1] = is_unwrap[:, :, fr1].astype(np.float32) * 1.0  # Green channel for unwrapped
                uw_progress_img_fr1[:, :, 2] = (~mask_3D[:, :, fr1]).astype(np.float32) * 0.5  # Blue channel for masked
                im_p1 = ax[2].imshow(uw_progress_img_fr1)
                fig.colorbar(im_p1, ax=ax[2])
                ax[2].set_title('Phase Quality Map [DEBUG]')

                plt.show()

            # Loop through flood-fill unwrapping until a voxel in the next frame is unwrapped
            while True:

                if (not np.any(adjoin[:, :, fr0])) and (not np.any(adjoin[:, :, fr1])):
                    mylogger.warning(f'No more adjoined pixels to unwrap between frames {frames[fr0]} and {frames[fr1]}, but no unwrapped pixels found in frame {frames[fr1]}. Stopping unwrapping in this direction.')
                    break

                # Search for any unwrapped phase within the adjoin matrix of the next frame, get it, and move to next frame pair/loop
                tf = is_unwrap[:, :, fr1] & adjoin[:, :, fr1]

                if np.any(tf):
                    # Find max-quality pixel within tf
                    q = qual_3D[:, :, fr1]
                    masked_q = np.where(tf, q, -np.inf)

                    ind = np.argmax(masked_q)
                    val = masked_q.flat[ind]

                    # Convert to linear index -> (i0, j0)
                    i0, j0 = np.unravel_index(ind, w_phase_3D.shape[0:2])

                    break
                
                # Find index of highest quality adjoined pixel, in either frame
                # frame fr0
                q0 = qual_3D[:, :, fr0]
                mask0 = adjoin[:, :, fr0]
                masked_q0 = np.where(mask0, q0, -np.inf)
                ind0 = np.argmax(masked_q0)
                val0 = masked_q0.flat[ind0]

                # frame fr1
                q1 = qual_3D[:, :, fr1]
                mask1 = adjoin[:, :, fr1]
                masked_q1 = np.where(mask1, q1, -np.inf)
                ind1 = np.argmax(masked_q1)
                val1 = masked_q1.flat[ind1]

                # a) Current frame has highest quality adjoined pixel
                if val0 >= val1:
                    
                    # Linear → (i0, j0)
                    ind = np.ravel_multi_index(
                        (*np.unravel_index(ind0, w_phase_3D.shape[0:2]), fr0),
                        dims=(*w_phase_3D.shape[0:2], len(frames))
                    )
                    i, j = np.unravel_index(ind0, w_phase_3D.shape[0:2])

                    # Get spatial neighbours in current frame
                    inh = i + nhood_ioffset
                    jnh = j + nhood_joffset

                    valid = (
                        (inh >= 0) & (inh < w_phase_3D.shape[0]) &
                        (jnh >= 0) & (jnh < w_phase_3D.shape[1])
                    )

                    inh = inh[valid]
                    jnh = jnh[valid]

                    nhood = (inh, jnh, np.full_like(inh, fr0))

                    fac = qual_factor[valid]

                    # Get temporal neighbours in next frame
                    inh = i + nhood_time_ioffset
                    jnh = j + nhood_time_joffset

                    valid = (
                        (inh >= 0) & (inh < w_phase_3D.shape[0]) &
                        (jnh >= 0) & (jnh < w_phase_3D.shape[1])
                    )

                    inh = inh[valid]
                    jnh = jnh[valid]

                    nhood_time = (inh, jnh, np.full_like(inh, fr1))

                    # Define FLAG
                    flag_nextframe = False


                # b) Next frame has highest quality adjoined pixel
                else:

                    # Linear → (i0, j0)
                    ind = np.ravel_multi_index(
                        (*np.unravel_index(ind1, w_phase_3D.shape[0:2]), fr1),
                        dims=(*w_phase_3D.shape[0:2], len(frames))
                    )

                    i, j = np.unravel_index(ind1, w_phase_3D.shape[0:2])

                    # Get temporal neighbours in current frame (second frame in pair)
                    inh = i + nhood_time_ioffset
                    jnh = j + nhood_time_joffset

                    valid = (
                        (inh >= 0) & (inh < w_phase_3D.shape[0]) &
                        (jnh >= 0) & (jnh < w_phase_3D.shape[1])
                    )

                    inh = inh[valid]
                    jnh = jnh[valid]

                    nhood = (inh, jnh, np.full_like(inh, fr0))


                    fac = np.ones(len(inh))    # equal weighting for temporal neighbours

                    # Define FLAG
                    flag_nextframe = True

                    # Store fr1 pixel as seed point
                    all_seed_points[fr1] = (i, j)  # (row, col)

                # Get unwrapped neighbours
                nhood_flat = np.ravel_multi_index(nhood, dims=(w_phase_3D.shape[0], w_phase_3D.shape[1], len(frames)))
                tf = is_unwrap.ravel()[nhood_flat] & islocal.ravel()[nhood_flat]
                nhood_unwrapped = nhood_flat[tf]
                fac = fac[tf]

                num_neighbours = len(nhood_unwrapped)

                # No unwrapped neighbours (should not happen)
                if num_neighbours == 0:
                    adjoin.ravel()[ind] = False
                    continue

                # Single unwrapped neighbour
                elif num_neighbours == 1:
                    phase_ref = temp_uw_phase_3D.ravel()[nhood_unwrapped[0]]
                    p = unwrap_2element_scalar(phase_ref, w_phase_3D.ravel()[ind])

                # Multiple unwrapped neighbours
                else:
                    f = fac * qual_3D.ravel()[nhood_unwrapped]  # quality factor

                    ind_pr = np.argmax(f)                  # neighbour with max weighted quality
                    neighbour = nhood_unwrapped[ind_pr]
                    phase_ref = temp_uw_phase_3D.ravel()[neighbour]
                    p = unwrap_2element_scalar(phase_ref, w_phase_3D.ravel()[ind])  

                # Unwrap current pixel
                temp_uw_phase_3D.ravel()[ind] = p

                # Update unwrap mask
                is_unwrap.ravel()[ind] = True

                # Update adjoin matrix
                if not flag_nextframe:
                    # Remove current pixel from adjoin
                    adjoin.ravel()[ind] = False

                    # Spatial neighbors (same frame)
                    adjoin.ravel()[nhood_flat] = (~is_unwrap.ravel()[nhood_flat]) & islocal.ravel()[nhood_flat]

                    # Temporal neighbors (next frame)
                    nhood_time_flat = np.ravel_multi_index(nhood_time, dims=(w_phase_3D.shape[0], w_phase_3D.shape[1], len(frames)))
                    adjoin.ravel()[nhood_time_flat] = islocal.ravel()[nhood_time_flat]

                # Debug: update persistent unwrapped phase plot (if enabled)
                if config["debug_flags"]["debug_phaseunwrapping"] and im is not None:
                    im.set_data(temp_uw_phase_3D[:, :, fr0])
                    # Define new image for progress quality map
                    # Background set to black, masked pixels set to blue, adjoin pixels set to red, unwrapped pixels set to white
                    uw_progress_img_fr0 = np.zeros((*w_phase_3D.shape[0:2], 3), dtype=np.float32)
                    uw_progress_img_fr0[:, :, 0] = adjoin[:, :, fr0].astype(np.float32) * 1.0  # Red channel for adjoin
                    uw_progress_img_fr0[:, :, 1] = is_unwrap[:, :, fr0].astype(np.float32) * 1.0  # Green channel for unwrapped
                    uw_progress_img_fr0[:, :, 2] = (~mask_3D[:, :, fr0]).astype(np.float32) * 0.5  # Blue channel for masked
                    im_p0.set_data(uw_progress_img_fr0)
                    # Define new image for progress quality map
                    # Background set to black, masked pixels set to blue, adjoin pixels set to red, unwrapped pixels set to white
                    uw_progress_img_fr1 = np.zeros((*w_phase_3D.shape[0:2], 3), dtype=np.float32)
                    uw_progress_img_fr1[:, :, 0] = adjoin[:, :, fr1].astype(np.float32) * 1.0  # Red channel for adjoin
                    uw_progress_img_fr1[:, :, 1] = is_unwrap[:, :, fr1].astype(np.float32) * 1.0  # Green channel for unwrapped
                    uw_progress_img_fr1[:, :, 2] = (~mask_3D[:, :, fr1]).astype(np.float32) * 0.5  # Blue channel for masked
                    im_p1.set_data(uw_progress_img_fr1)
                    # Update color limits if needed to keep contrast reasonable
                    finite = np.isfinite(temp_uw_phase_3D[:, :, fr0])
                    if np.any(finite):
                        im.set_clim(np.nanmin(temp_uw_phase_3D[:, :, fr0][finite]), np.nanmax(temp_uw_phase_3D[:, :, fr0][finite]))
                    fig.canvas.draw_idle()
                    plt.pause(0.5)

            # Debug: close persistent plot
            if config["debug_flags"]["debug_phaseunwrapping"] and fig is not None:
                plt.ioff()
                plt.close(fig)   
    
    # Check all frames have some unwrapped pixels
    has_unwrapped = np.any(~np.isnan(temp_uw_phase_3D), axis=(0, 1))

    missing_frames = np.where(~has_unwrapped)[0]

    if missing_frames.size > 0:
        mylogger.error(f"Unwrapping failed: no unwrapped pixels in frames {missing_frames}")
        sys.exit(1)

    mylogger.info('Unwrapped pixels found on all frames! Initial 3D flood-fill phase unwrapping completed successfully.')


    ### FULL UNWRAP PHASE USING 2D FLOOD-FILL ALGORITHM ###

    # Following initial 3D unwrapping to obtain a successfully unwrapped pixel on each frame, perform full 2D flood-fill unwrapping on each frame individually to ensure all pixels are unwrapped

    # Predefine final 3D unwrapped phase and quality arrays
    uw_phase_3D = np.full(w_phase_3D.shape, np.nan)
    qual_3D = np.full(w_phase_3D.shape, 0.0)

    # Loop through frames
    for frame_idx, frame in enumerate(frames):

        # mylogger.info(f'Performing final 2D flood-fill phase unwrapping on frame {frame}...')

        # Extract 2D phase and mask for current frame
        w_phase = w_phase_3D[:, :, frame_idx]
        mask = mask_3D[:, :, frame_idx]
        seed_point = all_seed_points[frame_idx]

        # Convert mask to logical
        if mask is not None:
            # Assume mask is 0 = background, 1 = myocardium, 2 = bloodpool, keep 1 = myocardium as true
            mask = (mask == 1)
        else:
            # Specify entire image as valid
            mask = np.ones(w_phase.shape, dtype=bool)

        # Extract initial unwrapped phase for current frame from 3D unwrapping using the seed point and add to w_phase
        w_phase[seed_point[0], seed_point[1]] = temp_uw_phase_3D[:, :, frame_idx][seed_point[0], seed_point[1]]

        # Update args mask for 2D unwrapping
        args["mask"] = mask

        # Unwrap phase using 2D flood-fill algorithm
        seed_point, qual, uw_phase = unwrap_phase_2d_floodfill(w_phase, args, config, mylogger, seed_point)

        # Store unwrapped 2D phase and quality back into 3D arrays
        uw_phase_3D[:, :, frame_idx] = uw_phase
        qual_3D[:, :, frame_idx] = qual

    return all_seed_points, qual_3D, uw_phase_3D




def extract_displacements(config, imaging_parameters, flagInverted, nSlices, nFrames, DENSE_series_index, sorted_dicoms, sorted_masks, mylogger):
    """
    Extract displacement data from DICOM file after phase unwrapping.
    Args:
        config (dict): Configuration parameters.
        imaging_parameters (dict): Imaging parameters for the slice.
        flagInverted (bool): Flag indicating if slice locations are inverted.
        dicom_data (pydicom.Dataset): DICOM dataset for the slice.
        mylogger (Logger): Logger for logging information.
    Returns:
        locations_ics (np.ndarray): Locations in image coordinate system.
        displacements_ics (np.ndarray): Displacements in image coordinate system.
    """

    # Initialise dicts
    locations = {}
    displacements = {}

    # Intialise dicts to store slice arrays
    w_phase_3D_xpha_slices = {}
    w_phase_3D_ypha_slices = {}
    w_phase_3D_zpha_slices = {}
    mask_3D_slices = {}

    # Adjust nFrames based on frame_of_seed
    nFrames = nFrames - config["parameters"]["frame_of_seed"]

    # Loop through sorted masks to create unwrapping inputs
    for mask_idx, mask in enumerate(sorted_masks):
        
        # Get slice index from mask filename "*_ID_'slice0''frame00'.nii.gz"
        slice_idx = int(mask.name.split("_")[3].split(".")[0][0])
        # Get frame index from mask filename "*_ID_'slice0''frame00'.nii.gz"
        frame_idx = int(mask.name.split("_")[3].split(".")[0][1:])

        if mask_idx == 0:
            # Get array shape from mask
            mask_data = nib.load(str(mask)).get_fdata().astype(np.uint8)
            mask_shape = mask_data.shape
        
        # Ensure dicts have initialized dicts for this slice
        slice_key = f"Slice_{slice_idx}"
        if slice_key not in locations:
            locations[slice_key] = {}
            displacements[slice_key] = {}

            # Initialize 3D arrays for phase images and mask
            mask_3D_slices[slice_key] = np.zeros((mask_shape[0], mask_shape[1], nFrames))
            w_phase_3D_xpha_slices[slice_key] = np.zeros((mask_shape[0], mask_shape[1], nFrames))
            w_phase_3D_ypha_slices[slice_key] = np.zeros((mask_shape[0], mask_shape[1], nFrames))
            w_phase_3D_zpha_slices[slice_key] = np.zeros((mask_shape[0], mask_shape[1], nFrames))

        # Define dicom_file corresponding to this mask
        dicom_file = sorted_dicoms[mask_idx]

        # Find the 3 phase images using the DENSE series index given MAG DICOM input

        # Find dicom_file in DENSE_series_index to get slice and frame indices
        dicom_file_str = str(dicom_file)
        search_slice_idx = None
        search_frame_idx = None
        for s_idx, s_key in enumerate(DENSE_series_index):
            for f_idx, f_key in enumerate(DENSE_series_index[s_key]["MAG"]):
                if DENSE_series_index[s_key]["MAG"][f_key] == dicom_file_str:
                    search_slice_idx = s_idx
                    search_frame_idx = f_idx
                    break
            if search_slice_idx is not None:
                break
        if search_slice_idx is None or search_frame_idx is None:
            mylogger.error(f'Could not find DICOM file in DENSE series index: {dicom_file_str}')
            sys.exit(1)
        # Get corresponding X, Y, Z phase DICOM files
        dicom_mag_file = DENSE_series_index[list(DENSE_series_index.keys())[search_slice_idx]]["MAG"][list(DENSE_series_index[list(DENSE_series_index.keys())[search_slice_idx]]["MAG"].keys())[search_frame_idx]]
        dicom_xpha_file = DENSE_series_index[list(DENSE_series_index.keys())[search_slice_idx]]["XPHA"][list(DENSE_series_index[list(DENSE_series_index.keys())[search_slice_idx]]["XPHA"].keys())[search_frame_idx]]
        dicom_ypha_file = DENSE_series_index[list(DENSE_series_index.keys())[search_slice_idx]]["YPHA"][list(DENSE_series_index[list(DENSE_series_index.keys())[search_slice_idx]]["YPHA"].keys())[search_frame_idx]]
        dicom_zpha_file = DENSE_series_index[list(DENSE_series_index.keys())[search_slice_idx]]["ZPHA"][list(DENSE_series_index[list(DENSE_series_index.keys())[search_slice_idx]]["ZPHA"].keys())[search_frame_idx]]
        # Load phase DICOM images
        img_xpha = (pydicom.dcmread(dicom_xpha_file).pixel_array / 4095.0) * (2 * np.pi) - np.pi
        img_ypha = (pydicom.dcmread(dicom_ypha_file).pixel_array / 4095.0) * (2 * np.pi) - np.pi
        img_zpha = (pydicom.dcmread(dicom_zpha_file).pixel_array / 4095.0) * (2 * np.pi) - np.pi

        # Check frame indices match
        if frame_idx != search_frame_idx:
            mylogger.error(f'Frame index mismatch between mask and DENSE series index for slice {slice_idx}, frame {frame_idx}.')
            sys.exit(1)

        frame_idx_array = frame_idx - config["parameters"]["frame_of_seed"]

        # Store phase images in dicts
        w_phase_3D_xpha_slices[slice_key][:, :, frame_idx_array] = img_xpha
        w_phase_3D_ypha_slices[slice_key][:, :, frame_idx_array] = img_ypha
        w_phase_3D_zpha_slices[slice_key][:, :, frame_idx_array] = img_zpha

        # Load and store mask data in 3D array
        mask_data = nib.load(str(mask)).get_fdata().astype(np.uint8)
        mask_3D_slices[slice_key][:, :, frame_idx_array] = mask_data

    
    # Loop through slices to perform phase unwrapping
    for slice_idx in range(len(w_phase_3D_xpha_slices.keys())):

        slice_key = f"Slice_{slice_idx}"

        # Get 3D wrapped phase images and mask for this slice
        img_xpha_3D = w_phase_3D_xpha_slices[slice_key]
        img_ypha_3D = w_phase_3D_ypha_slices[slice_key]
        img_zpha_3D = w_phase_3D_zpha_slices[slice_key]
        mask_3D = mask_3D_slices[slice_key]

        mylogger.info(f'Unwrapping phase images for {slice_key} using 3D flood-fill algorithm...')

        # Define unwrapping arguments
        if config["preprocessing_unwrapping"]["mask_flag"]:
            mask_data = mask_3D
        else:
            mask_data = None

        args = {
            "mask": None,
            "mask_3D": mask_data,
            "pixel_size": np.array(imaging_parameters["PixelSpacing"]),
            "seed": config["preprocessing_unwrapping"]["seed_point_selection"],  #default to manual seed selection for DENSE phase images - auto|manual
            "connectivity": 4,  #default to 4-connectivity for 2D unwrapping of DENSE phase images
            "searchradius": 2  #pixels - only used for 3D unwrapping
        }

        # Initialise seed point
        seed_point_3D = None

        # Initial phase unwrap
        seed_point_3D, qual_xpha_3D, uw_phase_xpha_3D = unwrap_phase_3d_floodfill(img_xpha_3D, args, config, mylogger, seed_point_3D)
        seed_point_3D, qual_ypha_3D, uw_phase_ypha_3D = unwrap_phase_3d_floodfill(img_ypha_3D, args, config, mylogger, seed_point_3D)
        seed_point_3D, qual_zpha_3D, uw_phase_zpha_3D = unwrap_phase_3d_floodfill(img_zpha_3D, args, config, mylogger, seed_point_3D)

        # Loop through frames to process 3D unwrapped phase images
        for frame_idx, frame in enumerate(range(config["parameters"]["frame_of_seed"], nFrames + config["parameters"]["frame_of_seed"])):

            # mylogger.info(f'Processing unwrapped phase images for {slice_key}, frame {frame}...')

            # Extract 2D unwrapped phase images for current frame
            uw_phase_xpha = uw_phase_xpha_3D[:, :, frame_idx]
            uw_phase_ypha = uw_phase_ypha_3D[:, :, frame_idx]
            uw_phase_zpha = uw_phase_zpha_3D[:, :, frame_idx]

            # Extract 2D quality maps for current frame
            qual_xpha = qual_xpha_3D[:, :, frame_idx]
            qual_ypha = qual_ypha_3D[:, :, frame_idx]
            qual_zpha = qual_zpha_3D[:, :, frame_idx]

            # Get corresponding DICOM file for this slice and frame
            dicom_file = DENSE_series_index[list(DENSE_series_index.keys())[slice_idx]]["MAG"][list(DENSE_series_index[list(DENSE_series_index.keys())[slice_idx]]["MAG"].keys())[frame_idx]]

            # Debug: plot unwrapped phase images
            if config["debug_flags"]["debug_phaseunwrapping"]:
                mylogger.info('Plotting unwrapped phase images for debugging...')
                fig, axs = plt.subplots(2, 3, figsize=(15, 10))
                im0 = axs[0, 0].imshow(uw_phase_xpha, cmap='gray')
                plt.colorbar(im0, ax=axs[0, 0])
                axs[0, 0].set_title('Unwrapped X Phase Image')

                im1 = axs[0, 1].imshow(uw_phase_ypha, cmap='gray')
                plt.colorbar(im1, ax=axs[0, 1])
                axs[0, 1].set_title('Unwrapped Y Phase Image')

                im2 = axs[0, 2].imshow(uw_phase_zpha, cmap='gray')
                plt.colorbar(im2, ax=axs[0, 2])
                axs[0, 2].set_title('Unwrapped Z Phase Image')

                im3 = axs[1, 0].imshow(qual_xpha, cmap='viridis')
                plt.colorbar(im3, ax=axs[1, 0])
                axs[1, 0].set_title('X Phase Quality Map')

                im4 = axs[1, 1].imshow(qual_ypha, cmap='viridis')
                plt.colorbar(im4, ax=axs[1, 1])
                axs[1, 1].set_title('Y Phase Quality Map')

                im5 = axs[1, 2].imshow(qual_zpha, cmap='viridis')
                plt.colorbar(im5, ax=axs[1, 2])
                axs[1, 2].set_title('Z Phase Quality Map')

                plt.suptitle('Phase Unwrapping Results [DEBUG]')
                plt.show()
            
            # Define locations and displacements in image coordinate system
            locs = []
            disps = []

            # Calculate displacement scaling factors
            xfac = 1 / (2 * np.pi * imaging_parameters["EncodingFrequency"] * imaging_parameters["Scale"] * imaging_parameters["PixelSpacing"][0])
            yfac = 1 / (2 * np.pi * imaging_parameters["EncodingFrequency"] * imaging_parameters["Scale"] * imaging_parameters["PixelSpacing"][1])
            zfac = 1 / (2 * np.pi * imaging_parameters["EncodingFrequency"] * imaging_parameters["Scale"] * imaging_parameters["PixelSpacing"][0])

            # Loop through each pixel in mask
            for i in range(uw_phase_xpha.shape[0]):
                for j in range(uw_phase_xpha.shape[1]):
                    if not np.isnan(uw_phase_xpha[i, j]) and not np.isnan(uw_phase_ypha[i, j]) and not np.isnan(uw_phase_zpha[i, j]):

                        # Define loc_x, loc_y, loc_z in mm
                        loc_x = (j + 0.5) * imaging_parameters["PixelSpacing"][0]
                        loc_y = (i + 0.5) * imaging_parameters["PixelSpacing"][1]
                        loc_z = get_slice_location(pydicom.dcmread(str(dicom_file)), mylogger)
                        
                        # Define disp_x, disp_y, disp_z in mm
                        # According to DENSE imaging flags RCswap and RCSflip

                        RCswap = int(imaging_parameters["DENSE"]['RCswap'])
                        Rflip = int(imaging_parameters["DENSE"]['RCSflip'].split('/')[0])
                        Cflip = int(imaging_parameters["DENSE"]['RCSflip'].split('/')[1])
                        Sflip = int(imaging_parameters["DENSE"]['RCSflip'].split('/')[2])

                        if RCswap == 0:
                            if Rflip == 0:
                                disp_x = uw_phase_xpha[i, j] * xfac * imaging_parameters["PixelSpacing"][0]
                            else:
                                disp_x = -uw_phase_xpha[i, j] * xfac * imaging_parameters["PixelSpacing"][0]
                            if Cflip == 0:
                                disp_y = uw_phase_ypha[i, j] * yfac * imaging_parameters["PixelSpacing"][1]
                            else:
                                disp_y = -uw_phase_ypha[i, j] * yfac * imaging_parameters["PixelSpacing"][1]
                            if Sflip == 0:
                                disp_z = uw_phase_zpha[i, j] * zfac * imaging_parameters["PixelSpacing"][0]
                            else:
                                disp_z = -uw_phase_zpha[i, j] * zfac * imaging_parameters["PixelSpacing"][0]
                        elif RCswap == 1:
                            if Rflip == 0:
                                disp_x = uw_phase_ypha[i, j] * xfac * imaging_parameters["PixelSpacing"][0]
                            else:
                                disp_x = -uw_phase_ypha[i, j] * xfac * imaging_parameters["PixelSpacing"][0]
                            if Cflip == 0:
                                disp_y = uw_phase_xpha[i, j] * yfac * imaging_parameters["PixelSpacing"][1]
                            else:
                                disp_y = -uw_phase_xpha[i, j] * yfac * imaging_parameters["PixelSpacing"][1]
                            if Sflip == 0:
                                disp_z = uw_phase_zpha[i, j] * zfac * imaging_parameters["PixelSpacing"][0]
                            else:
                                disp_z = -uw_phase_zpha[i, j] * zfac * imaging_parameters["PixelSpacing"][0]

                        # Append loc and disp tuples
                        locs.append((loc_x, loc_y, loc_z))
                        disps.append((disp_x, disp_y, disp_z))

            # Store in dicts
            locations[slice_key][str(frame)] = np.array(locs)
            displacements[slice_key][str(frame)] = np.array(disps)

            # Debug: plot scaled displacements
            if config["debug_flags"]["debug_displacements"]:
                mylogger.info('Plotting extracted displacements for debugging...')
                # Plot as quiver plot
                locs = np.array(locs)
                disps = np.array(disps)
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                ax.quiver(locs[:, 0], locs[:, 1], locs[:, 2],
                        -disps[:, 0], -disps[:, 1], -disps[:, 2])
                ax.set_xlabel('X (mm)')
                ax.set_ylabel('Y (mm)')
                ax.set_zlabel('Z (mm)')
                ax.set_title('Extracted Displacements in Image Coordinate System [DEBUG]')
                # Set axis equal
                plt.axis('equal')
                plt.show()


    # # Verification: dump extracted locations and displacements to matfile for denseanalysis comparison
    # from scipy.io import savemat
    # savemat('dev/testing_2026_1jan/2026_01_22_verifying-displacements-denseanalysis/debug_extracted_displacements.mat', {
    #     'locations_ics': np.array(locs),
    #     'displacements_ics': np.array(disps),
    #     'uw_phase_xpha': uw_phase_xpha,
    #     'uw_phase_ypha': uw_phase_ypha,
    #     'uw_phase_zpha': uw_phase_zpha
    # })
    # breakpoint()

    mylogger.info('Extracted displacements from DICOM data.')

    return locations, displacements