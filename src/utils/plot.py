import pyvista as pv
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp


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
    # unique_elem, inv = jnp.unique_inverse(eles)
    # out_wt = jnp.zeros((xis.shape[0], 3))
    # out_v0 = jnp.zeros((xis.shape[0], 3))
    # out_v1 = jnp.zeros((xis.shape[0], 3))
    # out_v2 = jnp.zeros((xis.shape[0], 3))
    # for ide, e in enumerate(unique_elem):
    #     mask = ide == inv
    #     try:
    #         wpts, v0, v1, v2 = get_vectors(mesh, [e], xis[mask])
    #         out_v0 = out_v0.at[mask].set(v0)
    #         out_v1 = out_v1.at[mask].set(v1)
    #         out_v2 = out_v2.at[mask].set(v2)
    #         out_wt = out_wt.at[mask].set(wpts)
    #     except Exception as e:
    #         print(e)
    #         breakpoint()

    def v_map(xi1, xi2, xi3):

            # v0 = xi1 = circumferential direction (theta)
            v0 = (xi1 / jnp.linalg.norm(xi1, axis=-1, keepdims=True))  # normalize

            # v1 = xi2 = longitudinal direction (z) - perpendicular to xi1, resticted to the xi1-xi2 plane
            xi2_orth = xi2 - jnp.sum(xi2 * xi1, axis=-1, keepdims=True) / jnp.sum(xi1 * xi1, axis=-1, keepdims=True) * xi1
            v1 = (xi2_orth / jnp.linalg.norm(xi2_orth, axis=-1, keepdims=True))  # normalize

            # v2 = xi3 = radial direction (r) - perpendicular to xi1 and xi2
            v2 = jnp.cross(v0, v1)  # cross product to get radial direction
            v2 = (v2 / jnp.linalg.norm(v2, axis=-1, keepdims=True))  # normalize

            return v0, v1, v2

    def get_clr_basis(mesh, eles, xis, ele_xi_pair=True):
            """ 
            Get the local vectors within the cylinder mesh at each embedded point.
            
            This function evaluates the embeddings of the mesh at the given elements and xi coordinates,
            and computes the vectors in world coordinates.
            It returns the world locations and the vectors v0, v1, and v2.

            This is used to define the coordinate system for which strain tensors are defined at each local coordinate.

            v0 = xi1 = circumferential direction (theta)
            v1 = xi2 = longitudinal direction (z) - perpendicular to xi1, resticted to xi1-xi2 plane
            v2 = xi3 = radial direction (r) - perpendicular to xi1 and xi2
            
            """
            if ele_xi_pair:
                world_locs = mesh.evaluate_ele_xi_pair_embeddings(eles, xis)  # world locations of the embedded points
                # define xi directions based on deriv embeddings
                xi1 = mesh.evaluate_ele_xi_pair_deriv_embeddings(eles, xis, derivs=(1, 0, 0))  # derivative in xi1 direction (C) wrt.
                xi2 = mesh.evaluate_ele_xi_pair_deriv_embeddings(eles, xis, derivs=(0, 1, 0))  # derivative in xi2 direction (L) wrt.
                xi3 = mesh.evaluate_ele_xi_pair_deriv_embeddings(eles, xis, derivs=(0, 0, 1))  # derivative in xi3 direction (R) wrt.
                v0, v1, v2 = v_map(xi1, xi2, xi3)
            else:
                world_locs = mesh.evaluate_embeddings(eles, xis)  # world locations of the embedded points
                # define xi directions based on deriv embeddings
                xi1 = mesh.evaluate_deriv_embeddings(eles, xis, derivs=(1, 0, 0))  # derivative in xi1 direction (C) wrt.
                xi2 = mesh.evaluate_deriv_embeddings(eles, xis, derivs=(0, 1, 0))  # derivative in xi2 direction (L) wrt.
                xi3 = mesh.evaluate_deriv_embeddings(eles, xis, derivs=(0, 0, 1))  # derivative in xi3 direction (R) wrt.
                v0, v1, v2 = v_map(xi1, xi2, xi3)
                
            return world_locs, v0, v1, v2

    eles, xis = model.fitted_mesh.embed_points(jnp.array(model.strain_points))

    world_locs, v0, v1, v2 = get_clr_basis(model.fitted_mesh, eles, xis)
    scene = pv.Plotter()
    model.fitted_mesh.plot(scene)
    # scene.add_arrows(out_wt, out_v0, color='r')
    # scene.add_arrows(out_wt, out_v1, color='g')
    # scene.add_arrows(out_wt, out_v2, color='b')
    scene.add_arrows(world_locs, v0, color='r')
    scene.add_arrows(world_locs, v1, color='g')
    scene.add_arrows(world_locs, v2, color='b')
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
    sn = ['Circumferential (Theta)', 'Longitudinal (Z)','Radial (R)']
    strain_keys = ["E_cc", "E_ll", "E_rr"]

    coords = model.strain_points

    scene = pv.Plotter(shape=(1,3))
    for tval in range(3):
        scene.subplot(0, tval)
        model.fitted_mesh.plot(scene, mesh_opacity=0.01)
        new_data = pv.PolyData(coords)
        new_data[f'{sn[tval]} Strain'] = model.strains[strain_keys[tval]]
        # define the color range for the strain values
        ms = np.median(model.strains[strain_keys[tval]])
        sds = np.median(np.abs(model.strains[strain_keys[tval]] - ms))
        sds = max(0.01, sds)  # sanity check on plotting
        clim = [ms - 3 * sds, ms + 3 * sds]
        if tval == 0:
            clim = [-0.3, 0.0]
        elif tval == 1:
            clim = [-0.2, 0.1]
        elif tval == 2:
            clim = [-0.3, 1.0]
        scene.add_mesh(
            new_data,
            render_points_as_spheres=True,
            point_size=20,
            clim=clim,
            cmap='viridis',
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
                ax.set_ylim(-0.3, 0.6)  # Set y-limits
            else:
                ax.set_ylim(-0.05, 0.1)
            ax.set_title(f'{strain_names[count]} Strain')
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
            ax.set_xlabel('Radius (mm)')
            if i == 0:
                ax.set_ylim(-0.4, 1.0)  # Set y-limits
            else:
                ax.set_ylim(-0.2, 0.2)
            ax.set_title(f'{strain_names[count]} Strain')
            count += 1

    plt.tight_layout()
    plt.show()
