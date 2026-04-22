import os
from pathlib import Path
import pyvista as pv
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from matplotlib.colors import LogNorm


def plot_geofit_meshes_with_slider(models):
    """
    models: dict of frame indices to CylinderModel class objects
    """
    scene = pv.Plotter()
    text_actor = [None]
    mesh_actor = [None]
    frame_keys = sorted(models.keys())
    current_idx = [0]  # mutable container to track current frame

    def update_mesh(idx):
        frame_idx = frame_keys[int(idx)]
        # Remove previous mesh actor if it exists
        if mesh_actor[0] is not None:
            scene.remove_actor(mesh_actor[0])
        # Remove previous text
        if text_actor[0] is not None:
            scene.remove_actor(text_actor[0])
        # Add the new mesh and store the actor
        mesh_actor[0] = models[frame_idx].geofit_mesh.plot(scene)
        # Add new text and store the actor
        text_actor[0] = scene.add_text(f"Frame: {frame_idx}", position='upper_left', font_size=12)
        scene.render()
        current_idx[0] = int(idx)

    # Add the first mesh and text
    mesh_actor[0] = models[frame_keys[0]].geofit_mesh.plot(scene)
    text_actor[0] = scene.add_text(f"Frame: {frame_keys[0]}", position='upper_left', font_size=12)

    # Add slider and keep a reference to the widget
    slider_widget = scene.add_slider_widget(
        callback=update_mesh,
        rng=[0, len(frame_keys) - 1],
        value=0,
        title="Frame",
        style='modern',
        fmt='%0.0f',
        pointa=(.025, .1), pointb=(.225, .1),
    )

    # Arrow key handlers
    def step_left():
        idx = (current_idx[0] - 1) % len(frame_keys)
        slider_widget.GetRepresentation().SetValue(idx)
        update_mesh(idx)

    def step_right():
        idx = (current_idx[0] + 1) % len(frame_keys)
        slider_widget.GetRepresentation().SetValue(idx)
        update_mesh(idx)

    # Add keyboard shortcuts
    scene.add_key_event("Left", step_left)
    scene.add_key_event("Right", step_right)
    scene.add_text("Use left and right arrow keys to step frames", position='lower_left', font_size=10, color='orange')

    scene.camera.SetParallelProjection(True)
    scene.show()



def plot_meshes_with_slider(models):
    """
    models: dict of frame indices to CylinderModel class objects
    """
    scene = pv.Plotter(shape=(1,2))
    text_actor = [None]
    mesh_actor = [None]
    mesh_actor_ffd = [None]
    frame_keys = sorted(models.keys())
    current_idx = [0]  # mutable container to track current frame

    def update_mesh(idx):
        frame_idx = frame_keys[int(idx)]
        for tval in range(2):
            scene.subplot(0,tval)
            if tval == 0:
                # Remove previous mesh actor if it exists
                if mesh_actor[0] is not None:
                    scene.remove_actor(mesh_actor[0])
                # Remove previous text
                if text_actor[0] is not None:
                    scene.remove_actor(text_actor[0])
                # Add the new mesh and store the actor
                mesh_actor[0] = models[frame_idx].geofit_mesh.plot(scene)
                # Add new text and store the actor
                text_actor[0] = scene.add_text(f"Frame: {frame_idx}", position='upper_left', font_size=12)
            elif tval == 1:
                # Remove previous fitted mesh actor if it exists
                if mesh_actor_ffd[0] is not None:
                    scene.remove_actor(mesh_actor_ffd[0])
                # Add the new fitted mesh and store the actor
                mesh_actor_ffd[0] = models[frame_idx].fitted_mesh.plot(scene)
        scene.render()
        current_idx[0] = int(idx)

    # Add the first mesh and text
    for tval in range(2):
        scene.subplot(0,tval)
        if tval == 0:
            mesh_actor[0] = models[frame_keys[0]].geofit_mesh.plot(scene)
            text_actor[0] = scene.add_text(f"Frame: {frame_keys[0]}", position='upper_left', font_size=12)
        elif tval == 1:
            mesh_actor_ffd[0] = models[frame_keys[0]].fitted_mesh.plot(scene)
    

    # Add slider and keep a reference to the widget
    slider_widget = scene.add_slider_widget(
        callback=update_mesh,
        rng=[0, len(frame_keys) - 1],
        value=0,
        title="Frame",
        style='modern',
        fmt='%0.0f',
        pointa=(.025, .1), pointb=(.225, .1),
    )

    # Arrow key handlers
    def step_left():
        idx = (current_idx[0] - 1) % len(frame_keys)
        slider_widget.GetRepresentation().SetValue(idx)
        update_mesh(idx)

    def step_right():
        idx = (current_idx[0] + 1) % len(frame_keys)
        slider_widget.GetRepresentation().SetValue(idx)
        update_mesh(idx)

    # Add keyboard shortcuts
    scene.add_key_event("Left", step_left)
    scene.add_key_event("Right", step_right)
    scene.add_text("Use left and right arrow keys to step frames", position='lower_left', font_size=10, color='orange')

    scene.link_views()
    scene.camera.SetParallelProjection(True)
    scene.show()


def plot_local_coordinate_axes_strain_points(model):
    """
    Plot the local coordinate axes in the mesh.
    
    This function plots the local coordinate axes in the mesh using pyvista.
    It uses the get_vectors function to get the local vectors in the mesh,
    and then plots the local coordinate axes in the world coordinates.
    """

    def v_map(xi1, xi2, xi3):
            """
            Define orthogonal basis vectors v0, v1, v2 based on the material coordinate system defined by xi1, xi2, xi3.

            Local wall cooordinate system defines strains in a local cardiac coordinate system.
            """

            # v0 = xi1 = circumferential direction (theta)
            v0 = (xi1 / jnp.linalg.norm(xi1, axis=-1, keepdims=True))  # normalize

            # v1 = ~xi2 = longitudinal direction (z) - perpendicular to xi1, resticted to the xi1-xi2 plane
            xi2_orth = xi2 - jnp.sum(xi2 * xi1, axis=-1, keepdims=True) / jnp.sum(xi1 * xi1, axis=-1, keepdims=True) * xi1
            v1 = (xi2_orth / jnp.linalg.norm(xi2_orth, axis=-1, keepdims=True))  # normalize

            # v2 = ~xi3 = radial direction (r) - perpendicular to xi1 and xi2
            v2 = jnp.cross(v1, v0)  # cross product to get radial direction
            v2 = (v2 / jnp.linalg.norm(v2, axis=-1, keepdims=True))  # normalize

            return v0, v1, v2

    def get_clr_basis(mesh, eles, xis, ele_xi_pair=True):
            """ 
            Get the local vectors within the cylinder mesh at each embedded point.
            
            This function evaluates the embeddings of the mesh at the given elements and xi coordinates,
            and computes the vectors in world coordinates.
            It returns the world locations and the vectors v0, v1, and v2.

            This is used to define the coordinate system for which strain tensors are defined at each local coordinate.

            Material coordinate system (xi):
            xi1 = material circumferential direction (theta) - v
            xi2 = material longitudinal direction (z) - u
            xi3 = material radial direction (r) - w

            Local wall coordinate system (v):
            v0 = xi1 = circumferential direction (theta)
            v1 = longitudinal direction (z) - perpendicular to xi1, resticted to the xi1-xi2 plane
            v2 = radial direction (r) - perpendicular to xi1 and xi2
            
            """
            if ele_xi_pair:
                world_locs = mesh.evaluate_embeddings_ele_xi_pair(eles, xis)  # world locations of the embedded points
                # define xi directions based on deriv embeddings
                xi1 = mesh.evaluate_deriv_embeddings_ele_xi_pair(eles, xis, derivs=(0, 1, 0))  # derivative in xi1 direction (C) wrt. v
                xi2 = mesh.evaluate_deriv_embeddings_ele_xi_pair(eles, xis, derivs=(1, 0, 0))  # derivative in xi2 direction (L) wrt. u
                xi3 = mesh.evaluate_deriv_embeddings_ele_xi_pair(eles, xis, derivs=(0, 0, 1))  # derivative in xi3 direction (R) wrt. w
                v0, v1, v2 = v_map(xi1, xi2, xi3)
            else:
                world_locs = mesh.evaluate_embeddings(eles, xis)  # world locations of the embedded points
                # define xi directions based on deriv embeddings
                xi1 = mesh.evaluate_deriv_embeddings(eles, xis, derivs=(0, 1, 0))  # derivative in xi1 direction (C) wrt. v
                xi2 = mesh.evaluate_deriv_embeddings(eles, xis, derivs=(1, 0, 0))  # derivative in xi2 direction (L) wrt. u
                xi3 = mesh.evaluate_deriv_embeddings(eles, xis, derivs=(0, 0, 1))  # derivative in xi3 direction (R) wrt. w
                v0, v1, v2 = v_map(xi1, xi2, xi3)

            return world_locs, v0, v1, v2

    eles, xis = model.fitted_mesh.embed_points(jnp.array(model.strain_points))

    world_locs, v0, v1, v2 = get_clr_basis(model.fitted_mesh, eles, xis)
    scene = pv.Plotter()
    model.fitted_mesh.plot(scene)
    pts = pv.PolyData(np.array(world_locs))
    pts["v0"] = np.array(v0)
    pts["v1"] = np.array(v1)
    pts["v2"] = np.array(v2)
    scene.add_mesh(pts.glyph(orient="v0", factor=3.0, scale=False), color="red", label="v0 (C - Red)")
    scene.add_mesh(pts.glyph(orient="v1", factor=3.0, scale=False), color="green", label="v1 (L - Green)")
    scene.add_mesh(pts.glyph(orient="v2", factor=3.0, scale=False), color="blue", label="v2 (R - Blue)")

    scene.add_axes()
    scene.add_legend()
    scene.show()


def plot_meshes(model):
    """
    Plot the undef (fitted) and def (geofit) meshes together.
    """

    scene = pv.Plotter()
    model.fitted_mesh.plot(scene, node_colour='g')
    model.geofit_mesh.plot(scene, node_colour='b')
    scene.show()


def plot_strains_colourmaps(model):
    """
    Plot the strains in the mesh using model.strains dict keys.
    """
    sn = ['Circumferential (Theta)', 'Longitudinal (Z)','Radial (R)', 'Circ-Long','Circ-Rad', 'Long-Rad']
    strain_keys = ["E_cc", "E_ll", "E_rr", "E_cl", "E_cr", "E_lr"]  # Use dict keys

    coords = model.strain_points

    scene = pv.Plotter(shape=(2,3))
    for tval in range(6):
        scene.subplot(tval // 3, tval % 3)
        model.fitted_mesh.plot(scene, mesh_opacity=0.01)
        new_data = pv.PolyData(coords)
        new_data[f'{sn[tval]} Strain'] = model.strains[strain_keys[tval]]
        
        # # v1 calculate the color range for the strain values (different for each strain component)
        # ms = np.median(model.strains[strain_keys[tval]])
        # sds = np.median(np.abs(model.strains[strain_keys[tval]] - ms))
        # sds = max(0.01, sds)  # sanity check on plotting
        # clim = [ms - 3 * sds, ms + 3 * sds]
        # scene.add_mesh(
        #     new_data,
        #     render_points_as_spheres=True,
        #     point_size=20,
        #     clim=clim,
        #     cmap='viridis',
        #     scalar_bar_args={
        #         'title': f'{sn[tval]} Strain',
        #         'n_labels': 5,
        #         'position_x': 0.25,
        #         'width': 0.5,
        #         'title_font_size': 30
        #     }
        
        # # v2 - define cmaps - viridis
        # if tval // 3 == 0:
        #     clim = [-0.25, 0.5]
        # elif tval // 3 == 1:
        #     clim = [-0.02, 0.06]
        # scene.add_mesh(
        #     new_data,
        #     render_points_as_spheres=True,
        #     point_size=20,
        #     clim=clim,
        #     cmap='viridis',
        #     scalar_bar_args={
        #         'title': f'{sn[tval]} Strain',
        #         'n_labels': 5,
        #         'position_x': 0.25,
        #         'width': 0.5,
        #         'title_font_size': 30
        #     }

        # v3 - define mirrored cmaps - diverging colormap - coolwarm
        if tval // 3 == 0:
            clim = [-0.5, 0.5]
        elif tval // 3 == 1:
            clim = [-0.06, 0.06]
        scene.add_mesh(
            new_data,
            render_points_as_spheres=True,
            point_size=20,
            clim=clim,
            cmap='coolwarm',
            scalar_bar_args={
                'title': f'{sn[tval]} Strain',
                'n_labels': 5,
                'position_x': 0.25,
                'width': 0.5,
                'title_font_size': 30
            }
        )

    scene.link_views()
    scene.show()


def plot_strains_2D_transmural(model):
    """
    Plot all 6 strain components transmurally.

    The function generates 6 subplots, each showing one of the strain components.
    The strains are plotted in the world coordinates, with respect to radius.

    """

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))  # 2 rows, 3 columns

    radii = np.sqrt(
        (model.strains["Y"] - model.inputs.translation_vector[0]) ** 2 +
        (model.strains["Z"] - model.inputs.translation_vector[1]) ** 2
    )  # Assuming the first column is the radius

    strain_names = ['Circumferential (Theta)', 'Longitudinal (Z)', 'Radial (R)', 'Circ-Long','Circ-Rad', 'Long-Rad']
    strain_keys = ["E_cc", "E_ll", "E_rr", "E_cl", "E_cr", "E_lr"]  # Use dict keys

    count = 0

    for i in range(2):
        for j in range(3):
            ax = axes[i, j]
            x = radii
            y = model.strains[strain_keys[count]]  # Extract the strain component by key
            ax.grid(True, alpha=0.5)
            ax.scatter(x, y, c='blue', s=10)
            ax.set_xlabel('Radius (mm)')
            if i == 0:
                ax.set_ylim(-0.4, 1.0)  # Set y-limits
            else:
                ax.set_ylim(-0.2, 0.2)
            ax.set_title(f'{strain_names[count]} Strain')
            count += 1

    plt.tight_layout()
    plt.show()


def plot_strains_2D_transmural_heatmap(model):
    """
    Plot all 6 strain components transmurally using a heatmap.

    The function generates 6 subplots, each showing one of the strain components.
    The strains are plotted in the world coordinates, with respect to radius.
    The strains are visualised using a density heatmap, given large numbers of data points.

    """

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))  # 2 rows, 3 columns

    radii = np.sqrt(
        (model.strains["Y"] - 0) ** 2 +
        (model.strains["Z"] - 0) ** 2
    )  # Assuming the first column is the radius

    strain_names = ['Circumferential (Theta)', 'Longitudinal (Z)', 'Radial (R)', 'Circ-Long','Circ-Rad', 'Long-Rad']
    strain_keys = ["E_cc", "E_ll", "E_rr", "E_cl", "E_cr", "E_lr"]  # Use dict keys

    count = 0

    for i in range(2):
        for j in range(3):
            ax = axes[i, j]
            x = radii
            y = model.strains[strain_keys[count]]  # Extract the strain component by key
            gridsize = 50
            cmap = plt.get_cmap('YlOrBr')
            bgc = cmap(0.0)
            ax.set_facecolor(bgc)
            ax.grid(True, alpha=0.5)
            if i == 0:
                hb = ax.hexbin(x, y, gridsize=gridsize, cmap=cmap, vmax=10, extent=[np.min(radii), np.max(radii), -0.4, 1.0])#, norm=LogNorm())
            else:
                hb = ax.hexbin(x, y, gridsize=gridsize, cmap=cmap, vmax=10, extent=[np.min(radii), np.max(radii), -0.2, 0.2])#, norm=LogNorm())
            ax.set_xlabel('Radius (mm)')
            if i == 0:
                ax.set_ylim(-0.4, 1.0)  # Set y-limits
            else:
                ax.set_ylim(-0.2, 0.2)
            ax.set_title(f'{strain_names[count]} Strain')
            plt.colorbar(hb, ax=ax, label=f'Number of data points')
            count += 1

    plt.tight_layout()
    plt.show()


def plot_strains_2D_transmural_xi(model):
    """
    Plot all 6 strain components transmurally.

    The function generates 6 subplots, each showing one of the strain components.
    The strains are plotted in the world coordinates, with respect to radius.

    """

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))  # 2 rows, 3 columns

    _, xi = model.fitted_mesh.embed_points(jnp.array(model.strain_points))

    strain_names = ['Circumferential (Theta)', 'Longitudinal (Z)', 'Radial (R)', 'Circ-Long','Circ-Rad', 'Long-Rad']
    strain_keys = ["E_cc", "E_ll", "E_rr", "E_cl", "E_cr", "E_lr"]  # Use dict keys

    count = 0

    for i in range(2):
        for j in range(3):
            ax = axes[i, j]
            x = xi[:, 2]  # radial direction
            y = model.strains[strain_keys[count]]  # Extract the strain component by key
            ax.grid(True, alpha=0.5)
            ax.scatter(x, y, c='blue', s=10)
            ax.set_xlabel('xi3 (radial material coordinate)')
            if i == 0:
                ax.set_ylim(-0.4, 1.0)  # Set y-limits
            else:
                ax.set_ylim(-0.2, 0.2)
            ax.set_title(f'{strain_names[count]} Strain')
            count += 1

    plt.tight_layout()
    plt.show()


def plot_strains_2D_transmural_xi_heatmap(model):
    """
    Plot all 6 strain components transmurally using a heatmap.

    The function generates 6 subplots, each showing one of the strain components.
    The strains are plotted in the world coordinates, with respect to radial material coordinate, xi3.
    The strains are visualised using a density heatmap, given large numbers of data points.

    """

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))  # 2 rows, 3 columns

    _, xi = model.fitted_mesh.embed_points(jnp.array(model.strain_points))

    strain_names = ['Circumferential (Theta)', 'Longitudinal (Z)', 'Radial (R)', 'Circ-Long','Circ-Rad', 'Long-Rad']
    strain_keys = ["E_cc", "E_ll", "E_rr", "E_cl", "E_cr", "E_lr"]  # Use dict keys

    count = 0

    for i in range(2):
        for j in range(3):
            ax = axes[i, j]
            x = xi[:, 2]  # radial material coordinate
            y = model.strains[strain_keys[count]]  # Extract the strain component by key
            gridsize = 50
            cmap = plt.get_cmap('YlOrBr')
            bgc = cmap(0.0)
            ax.set_facecolor(bgc)
            ax.grid(True, alpha=0.5)
            if i == 0:
                hb = ax.hexbin(x, y, gridsize=gridsize, cmap=cmap, vmax=10, extent=[0, 1, -0.4, 1.0])#, norm=LogNorm())
            else:
                hb = ax.hexbin(x, y, gridsize=gridsize, cmap=cmap, vmax=10, extent=[0, 1, -0.2, 0.2])#, norm=LogNorm())
            ax.set_xlabel('xi3 (radial material coordinate)')
            if i == 0:
                ax.set_ylim(-0.4, 1.0)  # Set y-limits
            else:
                ax.set_ylim(-0.2, 0.2)
            ax.set_title(f'{strain_names[count]} Strain')
            plt.colorbar(hb, ax=ax, label=f'Number of data points')
            count += 1

    plt.tight_layout()
    plt.show()


### SAVE MOVIES ###


def save_contour_mov(epi_contours, endo_contours, nFrames, config, mylogger):
    """
    Save a movie of the contours across slices and frames if enabled in config.
    """

    # Define save path and filename for animation
    save_dir = config["save_mov"]["output_dir"]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_filename = "contours_all_slices.mp4"
    save_path = Path(save_dir, save_filename)

    # Fit axis limits to data from all slices
    all_epi_list = []
    for s_key in epi_contours:
        for f_key in epi_contours[s_key]:
            all_epi_list.append(epi_contours[s_key][f_key])
    
    all_epi = np.vstack(all_epi_list) if all_epi_list else np.empty((0, 3))

    plotter = pv.Plotter(shape=(2, 2), off_screen=True)

    # Dummy initialization to set bounds
    if len(all_epi) > 0:
        dummy_cloud_epi = pv.PolyData(all_epi)
        
        # Subplot (0, 0): 3D View
        plotter.subplot(0, 0)
        plotter.add_mesh(dummy_cloud_epi, color='black', opacity=0.0)
        plotter.show_axes()
        plotter.add_axes()
        
        # Subplot (0, 1): X-Y View
        plotter.subplot(0, 1)
        plotter.add_mesh(dummy_cloud_epi, color='black', opacity=0.0)
        plotter.view_xy()
        plotter.show_axes()
        
        # Subplot (1, 0): X-Z View
        plotter.subplot(1, 0)
        plotter.add_mesh(dummy_cloud_epi, color='black', opacity=0.0)
        plotter.view_xz()
        plotter.show_axes()
        
        # Subplot (1, 1): Y-Z View
        plotter.subplot(1, 1)
        plotter.add_mesh(dummy_cloud_epi, color='black', opacity=0.0)
        plotter.view_yz()
        plotter.show_axes()

    plotter.open_movie(filename=str(save_path), framerate=6)

    for frame_idx in range(nFrames):
        frame_key = str(frame_idx + config["parameters"]["frame_of_seed"])
        
        # Combine epi and endo points from all slices for this frame
        curr_epi = []
        curr_endo = []
        for s_key in epi_contours:
            if frame_key in epi_contours[s_key]:
                curr_epi.append(epi_contours[s_key][frame_key])
        for s_key in endo_contours:
            if frame_key in endo_contours[s_key]:
                curr_endo.append(endo_contours[s_key][frame_key])

        if curr_epi and curr_endo:
            epi = np.vstack(curr_epi)
            endo = np.vstack(curr_endo)
            
            cloud_epi = pv.PolyData(epi)
            cloud_endo = pv.PolyData(endo)
            
            actors = []
            
            # Subplot 0, 0
            plotter.subplot(0, 0)
            actors.append(plotter.add_mesh(cloud_epi, render_points_as_spheres=True, point_size=6.0, color="blue", name="points_0_epi"))
            actors.append(plotter.add_mesh(cloud_endo, render_points_as_spheres=True, point_size=6.0, color="red", name="points_0_endo"))
            plotter.add_text(f'Extracted Contours - All Slices Frame {frame_key}', name='title_0', font_size=10)
            
            # Subplot 0, 1
            plotter.subplot(0, 1)
            actors.append(plotter.add_mesh(cloud_epi, render_points_as_spheres=True, point_size=6.0, color="blue", name="points_1_epi"))
            actors.append(plotter.add_mesh(cloud_endo, render_points_as_spheres=True, point_size=6.0, color="red", name="points_1_endo"))
            plotter.add_text('X-Y View', name='title_1', font_size=6)
            
            # Subplot 1, 0
            plotter.subplot(1, 0)
            actors.append(plotter.add_mesh(cloud_epi, render_points_as_spheres=True, point_size=6.0, color="blue", name="points_2_epi"))
            actors.append(plotter.add_mesh(cloud_endo, render_points_as_spheres=True, point_size=6.0, color="red", name="points_2_endo"))
            plotter.add_text('X-Z View', name='title_2', font_size=6)
            
            # Subplot 1, 1
            plotter.subplot(1, 1)
            actors.append(plotter.add_mesh(cloud_epi, render_points_as_spheres=True, point_size=6.0, color="blue", name="points_3_epi"))
            actors.append(plotter.add_mesh(cloud_endo, render_points_as_spheres=True, point_size=6.0, color="red", name="points_3_endo"))
            plotter.add_text('Y-Z View', name='title_3', font_size=6)
            
            plotter.write_frame()
            
            # Remove actors
            plotter.subplot(0, 0)
            plotter.remove_actor(actors[0])
            plotter.subplot(0, 1)
            plotter.remove_actor(actors[1])
            plotter.subplot(1, 0)
            plotter.remove_actor(actors[2])
            plotter.subplot(1, 1)
            plotter.remove_actor(actors[3])

    plotter.close()



def save_displacement_mov(locations, displacements, nFrames, config, mylogger):
    """
    Save a movie of the displacements across slices and frames if enabled in config.
    """

    # Define save path and filename for animation
    save_dir = config["save_mov"]["output_dir"]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_filename = "displacements_all_slices.mp4"
    save_path = Path(save_dir, save_filename)

    # Fit axis limits to data from all slices
    all_locs_list = []
    all_disps_list = []
    for s_key in locations:
        for f_key in locations[s_key]:
            all_locs_list.append(locations[s_key][f_key])
            all_disps_list.append(displacements[s_key][f_key])
    
    all_locs = np.vstack(all_locs_list) if all_locs_list else np.empty((0, 3))
    all_disps = np.vstack(all_disps_list) if all_disps_list else np.empty((0, 3))

    # Find max displacement magnitude for consistent color scale
    max_disp_mag = 0.0
    if len(all_disps) > 0:
        max_disp_mag = np.max(np.linalg.norm(all_disps, axis=1))

    plotter = pv.Plotter(shape=(2, 2), off_screen=True)

    # Dummy initialization to set bounds
    if len(all_locs) > 0:
        dummy_cloud = pv.PolyData(all_locs)
        
        # Subplot (0, 0): 3D View
        plotter.subplot(0, 0)
        plotter.add_mesh(dummy_cloud, color='black', opacity=0.0)
        plotter.show_axes()
        plotter.add_axes()
        
        # Subplot (0, 1): X-Y View
        plotter.subplot(0, 1)
        plotter.add_mesh(dummy_cloud, color='black', opacity=0.0)
        plotter.view_xy()
        plotter.show_axes()
        
        # Subplot (1, 0): X-Z View
        plotter.subplot(1, 0)
        plotter.add_mesh(dummy_cloud, color='black', opacity=0.0)
        plotter.view_xz()
        plotter.show_axes()
        
        # Subplot (1, 1): Y-Z View
        plotter.subplot(1, 1)
        plotter.add_mesh(dummy_cloud, color='black', opacity=0.0)
        plotter.view_yz()
        plotter.show_axes()

    plotter.open_movie(filename=str(save_path), framerate=6)

    for frame_idx in range(nFrames):
        frame_key = str(frame_idx + config["parameters"]["frame_of_seed"])
        
        # Combine locs and disps from all slices for this frame
        curr_locs = []
        curr_disps = []
        for s_key in locations:
            if frame_key in locations[s_key] and frame_key in displacements[s_key]:
                curr_locs.append(locations[s_key][frame_key])
                curr_disps.append(displacements[s_key][frame_key])
        
        if curr_locs and curr_disps:
            locs = np.vstack(curr_locs)
            disps = np.vstack(curr_disps)
            
            cloud = pv.PolyData(locs)
            cloud["vectors"] = -disps  # Pyvista arrows
            
            # Calculate scalar magnitude for coloring
            cloud["Magnitude"] = np.linalg.norm(disps, axis=1)
            
            cloud.set_active_vectors("vectors")
            arrows = cloud.glyph(orient="vectors", scale="vectors", factor=1.0)
            
            actors = []
            
            # Subplot 0, 0
            plotter.subplot(0, 0)
            actors.append(plotter.add_mesh(arrows, scalars="Magnitude", clim=[0, max_disp_mag], cmap="jet", name="arrows_0"))
            plotter.add_text(f'Extracted Displacements - All Slices Frame {frame_key}', name='title_0', font_size=10)
            
            # Subplot 0, 1
            plotter.subplot(0, 1)
            actors.append(plotter.add_mesh(arrows, scalars="Magnitude", clim=[0, max_disp_mag], cmap="jet", name="arrows_1"))
            plotter.add_text('X-Y View', name='title_1', font_size=6)
            
            # Subplot 1, 0
            plotter.subplot(1, 0)
            actors.append(plotter.add_mesh(arrows, scalars="Magnitude", clim=[0, max_disp_mag], cmap="jet", name="arrows_2"))
            plotter.add_text('X-Z View', name='title_2', font_size=6)
            
            # Subplot 1, 1
            plotter.subplot(1, 1)
            actors.append(plotter.add_mesh(arrows, scalars="Magnitude", clim=[0, max_disp_mag], cmap="jet", name="arrows_3"))
            plotter.add_text('Y-Z View', name='title_3', font_size=6)
            
            plotter.write_frame()
            
            # Remove actors
            plotter.subplot(0, 0)
            plotter.remove_actor(actors[0])
            plotter.subplot(0, 1)
            plotter.remove_actor(actors[1])
            plotter.subplot(1, 0)
            plotter.remove_actor(actors[2])
            plotter.subplot(1, 1)
            plotter.remove_actor(actors[3])

    plotter.close()