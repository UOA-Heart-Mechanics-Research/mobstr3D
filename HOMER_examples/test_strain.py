from HOMER.io import load_mesh
from HOMER import Mesh
from pyvista import Plotter, PolyData
import numpy as np
import jax.numpy as jnp

from matplotlib import pyplot as plt


def get_vectors(mesh, xis):
    world_locs = mesh.evaluate_embeddings_in_every_element(xis)
    
    v0 = jnp.tile(np.array([1, 0, 0])[None], (world_locs.shape[0], 1))
    v1 = world_locs.copy()
    v1 = v1.at[:, 0].set(0)
    v1 = (v1 / jnp.linalg.norm(v1, axis=-1, keepdims=True))
    v2 = jnp.linalg.cross(v0, v1)
    return world_locs, v0, v1, v2

def put_in_spherical(mesh: Mesh, eles, xis, tensors):
    world_locs = mesh.evaluate_embeddings_in_every_element(xis)
    
    v0 = jnp.tile(np.array([1, 0, 0])[None], (world_locs.shape[0], 1))
    v1 = world_locs.copy()
    v1 = v1.at[:, 0].set(0)
    v1 = (v1 / jnp.linalg.norm(v1, axis=-1, keepdims=True))
    v2 = jnp.linalg.cross(v0, v1)

    fw_mat = jnp.linalg.inv(jnp.concatenate((
            v0[:, None],
            v1[:, None],
            v2[:, None]
    ), axis=1)).reshape(len(mesh.elements), -1, 3, 3)

    spherical =  fw_mat @ tensors
    return spherical

# def put_in_spherical(mesh, ele, xi, t):
#     return t

                      

init_mesh = load_mesh("../VERIFICATION/cylinder_v1_compphan_FFDonly/unfitted_mesh.json")
FFD_mesh = load_mesh("../VERIFICATION/cylinder_v1_compphan_FFDonly/fitted_mesh.json")

point_data = load_heart_points("/Users/ghar342/Documents/Repos/mobstr_3D/VERIFICATION/cylinder_v1_compphan_FFDonly/data_snr_inf/landmark_points.ipdata")
eles, xis = init_mesh.embed_points(point_data)


init_mesh = load_mesh("../VERIFICATION/cylinder_v1_compphan_FFDonly/unfitted_mesh.json")
FFD_mesh = load_mesh("../VERIFICATION/cylinder_v1_compphan_FFDonly/fitted_mesh.json")
#FFD_mesh_no_vol = load_mesh("bin/ffd_mesh_no_vol.json")

start_points = load_heart_points("../VERIFICATION/cylinder_v1_compphan_FFDonly/data_snr_inf/landmark_points.ipdata")

wmbed_e, wmbed_xis = init_mesh.embed_points(start_points, verbose=2)

# wmbed_e = np.load("bin/embeded_es.npy")
# wmbed_xis = np.load("bin/embeded_xis.npy")
# scene = Plotter()
# init_mesh.plot(scene)
# scene.add_arrows(np.array([0,0,0]), np.array([10, 0, 0]))
# scene.show()

grid = init_mesh.xi_grid(15, dim=3, boundary_points=False)


vol_strains = []
#nvol_strains = []
sub_eles = []
sub_xis = []
u, inv = np.unique_inverse(wmbed_e)
for i in range(8):
    mask = inv == i
    g = wmbed_xis[mask]

    sub_eles.append(i * np.ones(g.shape[0]))
    sub_xis.append(g)

    vol_str = FFD_mesh.strain_tensor(init_mesh, [i], g, put_in_spherical).reshape(8, -1, 3,3)[i]
    #nvol_str = FFD_mesh_no_vol.strain_tensor(init_mesh, g, put_in_spherical).reshape(8, -1, 3,3)[i]

    vol_strains.append(vol_str)
    #nvol_strains.append(nvol_str)

te = np.concatenate(sub_eles)
tx = np.concatenate(sub_xis)

vs = np.concatenate(vol_strains)
#nvs = np.concatenate(nvol_strains)


vs_world = FFD_mesh.evaluate_ele_xi_pair_embeddings(te, tx)

# plt.scatter(vs[:, 0], nvs[:, 0])
# plt.xlim([-0.5, 1.5])
# plt.xlabel(["Estimated strain w/ conserved volume"])
# plt.ylabel(["Estimated strain wo/ conserved volume"])
# plt.ylim([-0.5, 1.5])
#
# plt.show()

# raise ValueError()
# scene = Plotter()
# FFD_mesh.plot(scene, mesh_opacity=0.01)
# FFD_mesh_no_vol.plot(scene, mesh_opacity=0.1, node_colour='blue')
# scene.show()


# vs_world = FFD_mesh.evaluate_embeddings_in_every_element(grid)
# vs = FFD_mesh.strain_tensor(init_mesh, grid).reshape(-1, 3,3)
sn = ['longitudinal', 'radial', 'circumferential']
scene = Plotter(shape=(1,3))
for tval in range(3):
    scene.subplot(0,tval)
    FFD_mesh.plot(scene, mesh_opacity=0.01)
    new_data = PolyData(np.array(vs_world))
    new_data[f'strain {sn[tval]}'] = vs[:, tval,tval]
    ms = np.median(vs[:, tval, tval])
    sds = np.median(np.abs(vs[:, tval, tval] - ms))
    # lower = ms - 3 * sds
    # upper = ms + 3 * sds
    scene.add_mesh(new_data, render_points_as_spheres=True, point_size=30, clim=[-3 * sds, 3 * sds], cmap='coolwarm')

# for tval in range(3):
#     scene.subplot(1,tval)
#     FFD_mesh_no_vol.plot(scene, mesh_opacity=0.01)
#     new_data = PolyData(np.array(vs_world))
#     new_data[f'no_vol strain {sn[tval]}'] = nvs[:, tval,tval]
#     ms = np.median(nvs[:, tval, tval])
#     sds = np.median(np.abs(nvs[:, tval, tval] - ms))
#     # lower = ms - 3 * sds
#     # upper = ms + 3 * sds
#     scene.add_mesh(new_data, render_points_as_spheres=True, point_size=30, clim=[-3 * sds, 3 * sds], cmap='coolwarm')
scene.link_views()
scene.show()

raise ValueError()




strain_tensors = FFD_mesh.strain_tensor(init_mesh, grid, put_in_spherical)


scene = Plotter(shape = (1,3))

for i in range(3):
    scene.subplot(0,i)
    FFD_mesh.plot(scene, mesh_opacity=0.01)
    elem_to_eval = i
    new_data = PolyData(np.array(FFD_mesh.evaluate_embeddings_in_every_element(grid)))
    new_data[f'strain {elem_to_eval}'] = strain_tensors[:, :, elem_to_eval, elem_to_eval].flatten()
    ms = np.median(strain_tensors[:, :, elem_to_eval,elem_to_eval])
    sds = np.median(np.abs(strain_tensors[:, :, elem_to_eval,elem_to_eval] - ms))
    lower = ms - 3 * sds
    upper = ms + 3 * sds
    scene.add_mesh(new_data, render_points_as_spheres=True, point_size=10, clim=[lower, upper])
scene.link_views()
scene.show()

scene = Plotter()
FFD_mesh.plot(scene, mesh_opacity=0.01)
w, p0, p1, p2 = get_vectors(FFD_mesh, grid)
scene.add_arrows(w, p0, color='r')
scene.add_arrows(w, p1, color='g')
scene.add_arrows(w, p2, color='b')
scene.show()
