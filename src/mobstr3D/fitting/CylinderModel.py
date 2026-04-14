import datetime
import json
import sys
import numpy as np
import pyvista as pv
from pathlib import Path

import scipy
import jax.numpy as jnp
from scipy.optimize import least_squares
from copy import deepcopy
import pandas as pd

from HOMER import Mesh, MeshElement, MeshNode
from HOMER.basis_definitions import H3Basis, L1Basis, L2Basis
from HOMER.optim import jax_comp_kdtree_distance_query
from HOMER.io import dump_mesh_to_dict, parse_mesh_from_dict
from HOMER.jacobian_evaluator import jacobian



class CylinderModel:
    """
    This class holds all information about the current cylinder model being fitted.

    Attributes:

        frame               (int): The frame index for the current model.

        time                (float): Stored time for the latest generation of the current model.

        Inputs:
            inner_radius         (float): Inner radius of the cylinder.
            outer_radius         (float): Outer radius of the cylinder.
            cylinder_height      (float): Height of the cylinder.
            cylinder_bot         (float): Bottom position of the cylinder.
            translation_vector   (np.array): Vector for translation.
            num_of_elements      (np.array): Number of elements in the mesh (C,L,R).
            dist_to_data         (np.array): Distance from ends of cylinder mesh to data points.

        Data:
            endo                 (dict["Slice_*"] of Nx3 numpy arrays): Raw endocardial points separated by slice.
            epi                  (dict["Slice_*"] of Nx3 numpy arrays): Raw epicardial points separated by slice.
            loc                  (dict["Slice_*"] of Nx3 numpy arrays): Raw locations separated by slice.
            disp                 (dict["Slice_*"] of Nx3 numpy arrays): Raw displacements separated by slice.

        Fitting:

        endo_contours       (Nx3 numpy array): Endocardial contours for fitting.
        epi_contours        (Nx3 numpy array): Epicardial contours for fitting.

        displacements       (Nx3 numpy array): Calculated displacements for fitting.
        start_points        (Nx3 numpy array): Initial points for fitting.
        end_points          (Nx3 numpy array): Target points for fitting.
        fitted_points       (Nx3 numpy array): Fitted points - sometimes used for strain evaluation.

        template_mesh       (HOMER mesh): The cylinder mesh template.

        geofit_mesh         (HOMER mesh): The mesh after geometric fitting.
        geofit_RMSE         (float): Root Mean Square Error after geometric fitting.
        geofit_errors       (np.array): Residual errors after geometric fitting.

        fitted_mesh         (HOMER mesh): The mesh after FFD fitting.
        fitted_RMSE         (float): Root Mean Square Error after FFD fitting.
        fitted_errors       (np.array): Residual errors after FFD fitting.

        Strains:

        strain_points      (Nx3 numpy array): Points defined as where strains should be evaluated.
        strains            (dict): Calculated strains at the defined points.

    """

    class Inputs():
        def __init__(self, template_params, config):
            self.inner_radius = template_params["inner_radius"]
            self.outer_radius = template_params["outer_radius"]
            self.cylinder_height = template_params["cylinder_height"]
            self.cylinder_bot = template_params["cylinder_bot"]
            self.translation_vector = template_params["translation_vector"]
            self.num_of_elements = config["model_parameters"]["num_of_elements"]
            self.dist_to_data = config["model_parameters"]["dist_to_data"]

    class Data:
        def __init__(self, endo, epi, loc, disp):
            self.endo = endo
            self.epi = epi
            self.loc = loc
            self.disp = disp


    def __init__(self, frame, template_params, endo, epi, loc, disp, config):
        
        self.frame = frame

        self.time = datetime.datetime.now().isoformat()  # e.g., '1993-03-00T22:11:00.123456'
        
        self.inputs = self.Inputs(template_params, config)
        self.data = self.Data(endo, epi, loc, disp)

        self.endo_contours = jnp.vstack([arr for arr in endo.values() if arr is not None])
        self.epi_contours = jnp.vstack([arr for arr in epi.values() if arr is not None])

        self.displacements = jnp.vstack([arr for arr in disp.values() if arr is not None])
        self.start_points = jnp.vstack([arr for arr in loc.values() if arr is not None])
        self.end_points = self.start_points - self.displacements
        self.fitted_points = None

        self.template_mesh = None
        
        self.geofit_mesh = None
        self.geofit_RMSE = None
        self.geofit_errors = None

        self.fitted_mesh = None
        self.fitted_RMSE = None
        self.fitted_errors = None

        self.config = config

        self.strain_points = None
        self.strains = None


    """
    
    TEMPLATE MESH FITTING
    
    """

    def template_cylinder_H3H3L1(self ,res = (1,1,1)): 

        """
        Create a cylinder mesh template with H3, H3, and L1 basis functions.

        Define mesh in the CLR coordinates.

        w: r = sqrt(y^2 + z^2)
        v: theta = atan2(z, y)
        u: z = x

        x = long axis (height)

        """

        # Define the space in which we are going to create the mesh

        # u = long axis (height)
        space_u_bot = 0
        space_u_top = self.inputs.cylinder_height

        # v = circular axis (theta)
        space_v_bot = 0
        space_v_top = 2 * np.pi

        # Define derivative space in H3 H3 plane - u and v
        du = [0, 0, (space_u_top - space_u_bot)/res[0]]
        dv = [0, np.pi*2/res[1], 0]
        dudv = [0, 0, 0]

        # w = radial axis (r)
        space_w_bot = self.inputs.inner_radius
        space_w_top = self.inputs.outer_radius

        pgrid = np.meshgrid(
            np.linspace(space_w_bot, space_w_top, res[2] + 1),
            np.linspace(space_v_bot, space_v_top, res[1] + 1),
            np.linspace(space_u_bot, space_u_top, res[0] + 1),
            indexing='ij',

        )

        fp = np.array([p.flatten() for p in pgrid]).T

        nodes_obs = [MeshNode(loc = p, du=du, dv=dv, dudv=dudv) for p in fp]

        elements = []

        # Define steps for stepping through the nodes for node ordering - s = step
        su = 1
        sv = su * (res[0] + 1)
        sw = sv * (res[1] + 1)

        for i in range(res[0]): #du
            for j in range(res[1]): #dv
                for k in range(res[2]): #dw

                    # Define the nodes that we are going to use!
                    bl = su * i + sv * j + k * sw #base location - 0,0,0 node in element xi coords - node number will change depending on the element

                    nodes = [
                        bl,
                        bl + su,
                        bl + sv,
                        bl + su + sv,
                        bl + sw,
                        bl + su + sw,
                        bl + sv + sw,
                        bl + su + sv + sw,
                    ]
                    elements.append(MeshElement(node_indexes=nodes, basis_functions=(H3Basis, H3Basis, L1Basis)))

        self.template_mesh = Mesh(nodes=nodes_obs, elements=elements)

        # Generate the mesh from the nodes and elements
        self.template_mesh.generate_mesh()

        return self


    def refine_cylinder_all(self, refine_all_factor):
        """
        Perform refinement on the cylinder mesh in all directions.

        This function assumes the cylinder is oriented along the z-axis.
        """

        # Refine the mesh in all 3 directions by 'refine_all_factor'
        self.template_mesh.refine(refine_all_factor)

        # Generate the mesh from the nodes and elements
        self.template_mesh.generate_mesh()

        return self


    def refine_cylinder_defined(self, num_of_elements):
        """
        Perform refinement on the cylinder mesh in specified xi direction.
        Defining the node positions in the xi direction.

        'num_of_elements' = a tuple of the number of elements in each xi direction (xi_1, xi_2, xi_3).
        eg. [4, 1, 1]
        Would define node postions in xi as:
        (([0, 1], [0, 1], [0, 1/4, 2/4, 3/4, 1])) # (w, v, u) refinement

        This function assumes the cylinder is oriented along the x-axis.
        """

        # Calculate node positions in the xi direction based on 'num_of_elements'
        xi_1_positions = np.linspace(0, 1, num_of_elements[1] + 1) #w #R
        xi_2_positions = np.linspace(0, 1, num_of_elements[0] + 1) #v #T
        xi_3_positions = np.linspace(0, 1, num_of_elements[2] + 1) #u #Z

        # Refine the mesh in the specified xi direction
        self.template_mesh.refine(by_xi_refinement=(xi_1_positions, xi_2_positions, xi_3_positions))

        # Generate the mesh from the nodes and elements
        self.template_mesh.generate_mesh()

        return self


    def increase_order_cylinder_L2_xi3_template(self, num_of_elements):
        """
        Increase the order of the cylinder mesh to L2 basis functions.
        
        Fixed input: L2 in xi3
        """
        
        # Define the end number of elements in each xi direction
        xi1_elems = num_of_elements[1]
        xi2_elems = num_of_elements[0]
        xi3_elems = num_of_elements[2] # - unfinished - use to algorithmically define node reordering

        # Calculate total number of nodes in original mesh
        last_node = len(self.template_mesh.nodes) - 1

        # Copy the mesh to avoid modifying the original
        temp_mesh = deepcopy(self.template_mesh)

        # Refine the temp mesh in the xi3 direction only
        refine_xi3 = [1, 1, 2]
        # Calculate node positions in the xi direction based on 'num_of_elements'
        xi_1_positions = np.linspace(0, 1, refine_xi3[1] + 1) #w #R
        xi_2_positions = np.linspace(0, 1, refine_xi3[0] + 1) #v #T
        xi_3_positions = np.linspace(0, 1, refine_xi3[2] + 1) #u #Z
        # Refine the mesh in the specified xi direction
        temp_mesh.refine(by_xi_refinement=(xi_1_positions, xi_2_positions, xi_3_positions))
        # Generate the mesh from the nodes and elements
        temp_mesh.generate_mesh()

        # Redefine original mesh elements adding the new nodes
        new_elements = []
        counter = last_node + 1

        for elm in self.template_mesh.elements:
            add_nodes = [np.int64(counter), np.int64(counter+1), np.int64(counter+2), np.int64(counter+3)]
            counter += 2

            # Reorder the nodes in the element to include the new nodes.
            # This assumes the original element nodes are ordered as follows:
            # [0, 1, 2, 3, 4, 5, 6, 7]
            # and we want to insert the new nodes in between the existing nodes.
            
            # The new nodes will be inserted as follows:

            # for xi2 direction:
            # [0, add_nodes[0], 1, 2, add_nodes[1], 3, 4, add_nodes[2], 5, 6, add_nodes[3], 7]
            #elm.nodes = [elm.nodes[0],add_nodes[0],elm.nodes[1],elm.nodes[2],add_nodes[1],elm.nodes[3],elm.nodes[4],add_nodes[2],elm.nodes[5],elm.nodes[6],add_nodes[3],elm.nodes[7]] #order for L2 in xi2 direction

            # for xi3 direction:
            # [0, 1, 2, 3, add_nodes[0], add_nodes[1], add_nodes[2], add_nodes[3], 4, 5, 6, 7]
            elm.nodes = [elm.nodes[0], elm.nodes[1], elm.nodes[2], elm.nodes[3], add_nodes[0], add_nodes[1], add_nodes[2], add_nodes[3], elm.nodes[4], elm.nodes[5], elm.nodes[6], elm.nodes[7]] #order for L2 in xi3 direction

            new_elements.append(MeshElement(node_indexes=elm.nodes, basis_functions=(H3Basis, H3Basis, L2Basis)))

        # Rebuild the mesh with the refined nodes and new elements
        self.template_mesh = Mesh(nodes=temp_mesh.nodes, elements=new_elements)

        # Hot fix: node order in first and second element is not correct, so we need to fix it
        # This is a hardcoded value for the first element nodes, adjust as necessary
        self.template_mesh.elements[0].nodes = [0, 18, 2, 20, 36, 38, 37, 39, 1, 19, 3, 21] # This is a hardcoded value for the first element nodes, adjust as necessary - 8-1-1 H3H3L2 mesh
        # This is a hardcoded value for the second element nodes, adjust as necessary
        self.template_mesh.elements[1].nodes = [2, 20, 4, 22, 37, 39, 40, 41, 3, 21, 5, 23] # This is a hardcoded value for the first element nodes, adjust as necessary - 8-1-1 H3H3L2 mesh

        # TODO - better fix: swap node idx for node 37 and 38

        # generate the mesh from the nodes and elements
        self.template_mesh.generate_mesh()
        
        return self


    def translate_cylinder(self):
        """
        Translate the cylinder mesh by a given translation vector.

        This function assumes the cylinder is oriented along the z-axis.
        """

        for node in self.template_mesh.nodes:
            node.loc += self.inputs.translation_vector

        self.template_mesh.generate_mesh()

        return self
    

    def transform_cylinder_rtz2xyz(self):

        """
        Transform the cylinder mesh from cylindrical polar to rectangular cartesian coordinates.
        
        w: r = sqrt(y^2 + z^2)
        v: theta = atan2(z, y)
        u: z = -x

        This function assumes the cylinder is oriented along the z-axis.

        Options:
        nodes.loc = transforms node locations
        Need to also transform the derivative values:
        nodes['du'] = transforms du values
        nodes['dv'] = transforms dv values
        nodes['dudv'] = transforms dudv values
        """
    
        for node in self.template_mesh.nodes:

            # Calculate the Jacobian for the derivative transformations
            r, theta, z = node.loc
            J = np.array([
            [0, 0, 1],
            [np.cos(theta), -r*np.sin(theta), 0],
            [np.sin(theta),  r*np.cos(theta), 0]
            ])

            # Transform the node location
            node.loc = np.array([
                node.loc[2],  # x = -z
                node.loc[0] * np.cos(node.loc[1]),  # y = r * cos(theta)
                node.loc[0] * np.sin(node.loc[1])  # z = r * sin(theta)
            ])

            # Transform the derivatives
            node['du'] = J @ node['du']
            node['dv'] = J @ node['dv']
            node['dudv'] = J @ node['dudv']

        # Look at final element nodes and set to the correct values
        final_elem = self.template_mesh.elements[-1]
        final_elem.nodes = [14, 32, 0, 18, 15, 33, 1, 19] # This is a hardcoded value for the final element nodes, adjust as necessary - 8-1-1 H3H3L1 mesh
        #final_elem.nodes = [14, 50, 32, 0, 36, 18, 15, 51, 33, 1, 37, 19] # This is a hardcoded value for the final element nodes, adjust as necessary - 8-1-1 H3H3L1 mesh - adding node to xi2 direction... error
        #final_elem.nodes = [14, 32, 0, 18, 50, 51, 36, 38, 15, 33, 1, 19] # This is a hardcoded value for the final element nodes, adjust as necessary - 8-1-1 H3H3L2 mesh

        # Remove the excess nodes
        self.template_mesh._clean_pts()

        # Regenerate the mesh with connected nodes at end elements
        self.template_mesh.generate_mesh()

        return self



    """

    GEOMETRIC FITTING

    """

    def setup_geofit(self, mylogger):

        """
        Setup geometric fitting on the template mesh, both surfaces simultaneously.

        embed: defines embeddings of the contour points on the mesh surface, by point to point correspondence of the nearest points on the mesh surface.
        """
        
        inner_contours = self.endo_contours
        outer_contours = self.epi_contours

        # Get face grid for all surfaces
        face_grid_xis = self.geofit_mesh.xi_grid(dim=3, surface=True, res=200, boundary_points=False).reshape(6, -1, 3)
        # Keep only face grid with respect to the inner and outer surfaces
        # outer: xi3 = 1
        face_grid_xis_ow = face_grid_xis[-1]
        # inner: xi3 = 0
        face_grid_xis_iw = face_grid_xis[-2]

        # Evaluate the embeddings of the surface points on the mesh surface
        face_grid_locs_iw = self.geofit_mesh.evaluate_embeddings_in_every_element(face_grid_xis_iw)
        face_grid_locs_ow = self.geofit_mesh.evaluate_embeddings_in_every_element(face_grid_xis_ow)

        # Creates KD-tree eval functions that will calculate distances compatible with JAX from scipy (HOMER function)
        iw_tree = scipy.spatial.KDTree(face_grid_locs_iw)
        ow_tree = scipy.spatial.KDTree(face_grid_locs_ow)

        # Determine the embeddings (corresponding face_grid points on the mesh surface) for each contour point
        d_ow, i_ow = ow_tree.query(outer_contours, k=1, workers=-1)
        d_iw, i_iw = iw_tree.query(inner_contours, k=1, workers=-1)

        e_ow = i_ow // len(face_grid_xis_ow) # element index for each embedded point (using integer division to get the element index)
        e_iw = i_iw // len(face_grid_xis_iw)

        xi_ow = face_grid_xis_ow[i_ow % len(face_grid_xis_ow)] # xi location for each embedded point (using the modulus to get the index within the face grid)
        xi_iw = face_grid_xis_iw[i_iw % len(face_grid_xis_iw)]

        for node in self.geofit_mesh.nodes:
            # Fix the dv (long-axis) derivative to be zero
            node.fix_parameter("dv", inds=[0])
            node.fix_parameter("loc") # these nodes will be updated as a result of the 'angular fixing'
        self.geofit_mesh.generate_mesh()

        optimisable_param_bool = self.geofit_mesh.optimisable_param_bool.copy()
        template_true_params = jnp.array(self.geofit_mesh.true_param_array.copy())

        # Set up the angular fixing.
        node_loc_inds = np.array(self.geofit_mesh.associated_node_index(['loc'])).flatten().astype(int)
        template_node_locs = jnp.array(template_true_params[node_loc_inds].reshape(-1, 3))
        init_magnitude = np.linalg.norm(template_node_locs[:, 1:], axis=1) # Used for the initial parameters.
        node_unit_vectors = template_node_locs[:, 1:] / init_magnitude[:, np.newaxis]
        n_mags = len(init_magnitude)

        # Add in the additional information into the parameter array
        init_shape = self.geofit_mesh.optimisable_param_array.copy()
        init_params = np.concatenate((init_magnitude, init_shape))

        # Evaluate the Sobolev norm for smoothing
        sob = self.geofit_mesh.evaluate_sobolev()

        def geofit(params):
            """
            Construct the homer params from the input params
            """
            # Recover the two types of the params
            mag_param, shape_param = params[:n_mags], params[n_mags:]

            template_true_params = jnp.array(self.geofit_mesh.true_param_array.copy()) # Pulling the variables into the closure.
            template_node_locs = jnp.array(template_true_params[node_loc_inds].reshape(-1, 3))

            template_true_params = template_true_params.at[optimisable_param_bool].set(shape_param) # Setting the shape parameters.

            scaled_mags = node_unit_vectors * mag_param[:, None] # Setting the angular parameters.
            template_node_locs = template_node_locs.at[:, 1:].set(scaled_mags)
            template_true_params = template_true_params.at[node_loc_inds].set(template_node_locs.flatten())

            # Set the updated params as the new homer params
            homer_params = template_true_params

            # Evaluate the cost of the homer params
            inner_points = self.geofit_mesh.evaluate_embeddings_ele_xi_pair(e_iw, xi_iw, fit_params=homer_params)
            outer_points = self.geofit_mesh.evaluate_embeddings_ele_xi_pair(e_ow, xi_ow, fit_params=homer_params)

            iw_dists = (inner_points - inner_contours).flatten()
            ow_dists = (outer_points - outer_contours).flatten()

            # Grab the Sobolev distances to later apply Sobolev smoothing, if needed
            if self.config["smoothing"]["geo_smoothing"] == "sobprior":
                sobprior_weight = self.config["smoothing"]["geo_sobprior_weight"]
                sobprior_err = (sob - self.geofit_mesh.evaluate_sobolev(fit_params=homer_params)) * sobprior_weight
            else:
                sobprior_err = jnp.array([])

            return jnp.concatenate((iw_dists, ow_dists, sobprior_err))

        func, jac = jacobian(geofit, init_estimate=init_params)

        def update_from_params(mesh_instance, params):

            # Recover the two types of the params
            mag_param, shape_param = params[:n_mags], params[n_mags:]

            template_true_params = jnp.array(self.geofit_mesh.true_param_array.copy()) # Pulling the variables into the closure.
            template_node_locs = jnp.array(template_true_params[node_loc_inds].reshape(-1, 3))

            template_true_params = template_true_params.at[optimisable_param_bool].set(shape_param) # Setting the shape parameters.

            scaled_mags = node_unit_vectors * mag_param[:, None] # Setting the angular parameters.
            template_node_locs = template_node_locs.at[:, 1:].set(scaled_mags)
            template_true_params = template_true_params.at[node_loc_inds].set(template_node_locs.flatten())

            # Set the updated params as the new homer params
            homer_params = template_true_params
            mesh_instance.update_from_params(homer_params)

        return func, jac, init_params, update_from_params
    


    def setup_geofit_surf2contour(self, mylogger):

        """
        Setup geometric fitting on the template mesh, both surfaces simultaneously.

        surf2contour: finds the minimum distance contour point for all points of a grid generated on the inner and outer surfaces of the mesh.
        """
        
        inner_wall = self.endo_contours
        outer_wall = self.epi_contours

        # Creates KD-tree eval functions compatible with JAX from scipy (HOMER function)
        iw_tree = jax_comp_kdtree_distance_query(inner_wall, {"workers": -1})
        ow_tree = jax_comp_kdtree_distance_query(outer_wall, {"workers": -1})

        # Get face grid for all surfaces
        face_grid = self.geofit_mesh.xi_grid(dim=3, surface=True, res=20).reshape(6, -1, 3)
        # Keep only face grid with respect to the inner and outer surfaces
        # outer: xi3 = 1
        ow_xis = face_grid[-1]
        # inner: xi3 = 0
        iw_xis = face_grid[-2]

        for node in self.geofit_mesh.nodes:
            # Fix the dv (long-axis) derivative to be zero
            node.fix_parameter("dv", inds=[0])
            node.fix_parameter("loc") # these nodes will be updated as a result of the 'angular fixing'
        self.geofit_mesh.generate_mesh()

        optimisable_param_bool = self.geofit_mesh.optimisable_param_bool.copy()
        template_true_params = jnp.array(self.geofit_mesh.true_param_array.copy())

        # Set up the angular fixing.
        node_loc_inds = np.array(self.geofit_mesh.associated_node_index(['loc'])).flatten().astype(int)
        template_node_locs = jnp.array(template_true_params[node_loc_inds].reshape(-1, 3))
        init_magnitude = np.linalg.norm(template_node_locs[:, 1:], axis=1) # Used for the initial parameters.
        node_unit_vectors = template_node_locs[:, 1:] / init_magnitude[:, np.newaxis]
        n_mags = len(init_magnitude)

        # Add in the additional information into the parameter array
        init_shape = self.geofit_mesh.optimisable_param_array.copy()
        init_params = np.concatenate((init_magnitude, init_shape))

        # Evaluate the Sobolev norm for smoothing
        sob = self.geofit_mesh.evaluate_sobolev()

        def geofit_surf2contour(params):
            """
            Construct the homer params from the input params
            """
            # Recover the two types of the params
            mag_param, shape_param = params[:n_mags], params[n_mags:]

            template_true_params = jnp.array(self.geofit_mesh.true_param_array.copy()) # Pulling the variables into the closure.
            template_node_locs = jnp.array(template_true_params[node_loc_inds].reshape(-1, 3))

            template_true_params = template_true_params.at[optimisable_param_bool].set(shape_param) # Setting the shape parameters.

            scaled_mags = node_unit_vectors * mag_param[:, None] # Setting the angular parameters.
            template_node_locs = template_node_locs.at[:, 1:].set(scaled_mags)
            template_true_params = template_true_params.at[node_loc_inds].set(template_node_locs.flatten())

            # Set the updated params as the new homer params
            homer_params = template_true_params

            # Evaluate the cost of the homer params
            inner_points = self.geofit_mesh.evaluate_embeddings_in_every_element(iw_xis, fit_params=homer_params)
            outer_points = self.geofit_mesh.evaluate_embeddings_in_every_element(ow_xis, fit_params=homer_params)

            iw_dists = iw_tree(inner_points)
            ow_dists = ow_tree(outer_points)

            # Grab the Sobolev distances to later apply Sobolev smoothing, if needed
            if self.config["smoothing"]["geo_smoothing"] == "sobprior":
                sobprior_weight = self.config["smoothing"]["geo_sobprior_weight"]
                sobprior_err = (sob - self.geofit_mesh.evaluate_sobolev(fit_params=homer_params)) * sobprior_weight
            else:
                sobprior_err = jnp.array([])

            return jnp.concatenate((iw_dists, ow_dists, sobprior_err))

        func, jac = jacobian(geofit_surf2contour, init_estimate=init_params)

        def update_from_params_surf2contour(mesh_instance, params):

            # Recover the two types of the params
            mag_param, shape_param = params[:n_mags], params[n_mags:]

            template_true_params = jnp.array(self.geofit_mesh.true_param_array.copy()) # Pulling the variables into the closure.
            template_node_locs = jnp.array(template_true_params[node_loc_inds].reshape(-1, 3))

            template_true_params = template_true_params.at[optimisable_param_bool].set(shape_param) # Setting the shape parameters.

            scaled_mags = node_unit_vectors * mag_param[:, None] # Setting the angular parameters.
            template_node_locs = template_node_locs.at[:, 1:].set(scaled_mags)
            template_true_params = template_true_params.at[node_loc_inds].set(template_node_locs.flatten())

            # Set the updated params as the new homer params
            homer_params = template_true_params
            mesh_instance.update_from_params_surf2contour(homer_params)

        return func, jac, init_params, update_from_params_surf2contour
    


    def fit_geofit(self, mylogger):
        """
        Fit the geometric model to the data.

        This function performs the geometric fitting process by calling setup_geofit and running the optimization.
        """

        self.geofit_mesh = deepcopy(self.template_mesh)

        func, jac, init_params, update_func = self.setup_geofit(mylogger) # Change this to setup_geofit_surf2contour for surf2contour fitting

        optim = least_squares(func, init_params, jac=jac, verbose=2, max_nfev=50)
        update_func(self.geofit_mesh, optim.x)

        # Reshape the fitted errors to match the original shape
        self.geofit_errors = optim.fun.reshape(-1, 3)
        end_geofit_points = 3*(len(self.endo_contours) + len(self.epi_contours))

        # Calculate the magnitude of the fitted errors
        mag_error = np.linalg.norm(self.geofit_errors[:end_geofit_points], axis=-1)

        # Print root mean squared error
        self.geofit_RMSE = np.sqrt(np.mean(mag_error**2))
        mylogger.info(f"Geometric Fit Root Mean Squared Error: {self.geofit_RMSE:.6f}")

        # Undo fixing of parameters and regenerates mesh
        self.geofit_mesh.unfix_mesh()

        #self.geofit_mesh = deepcopy(self.template_mesh) # This is a hot fix to skip geofit mesh fitting when observing computational phantom

        return self
    

    def increase_order_cylinder_L2_xi3_geofit(self, num_of_elements):
        """
        Increase the order of the geofitted cylinder mesh to L2 basis functions.
        
        Fixed input: L2 in xi3
        For when the order increase happens after geofit.
        8-1-1 H3H3L1 to 8-1-1 H3H3L2
        
        This avoids issues where the internal node points are not adjusted correctly during the geofit process.
        """

        # Copy the mesh to avoid modifying the original
        temp_mesh = deepcopy(self.geofit_mesh)

        # Refine the temporary mesh in the xi3 direction only
        refine_xi3 = [1, 1, 2]
        # Calculate node positions in the xi direction based on 'num_of_elements'
        xi_1_positions = np.linspace(0, 1, refine_xi3[1] + 1) #w #R
        xi_2_positions = np.linspace(0, 1, refine_xi3[0] + 1) #v #T
        xi_3_positions = np.linspace(0, 1, refine_xi3[2] + 1) #u #Z
        # Refine the mesh in the specified xi direction
        temp_mesh.refine(by_xi_refinement=(xi_1_positions, xi_2_positions, xi_3_positions))
        # Generate the mesh from the nodes and elements
        temp_mesh.generate_mesh()

        # Redefine mesh using spatial hash from refined nodes
        spatial_hash = {tuple(np.trunc(node.loc*1000).tolist()):idn for idn, node in enumerate(temp_mesh.nodes)}

        # Define xi evaluation points for elements - canonical positions for L2 basis functions in xi3 direction
        xi_eval = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 0.5],
            [1, 0, 0.5],
            [0, 1, 0.5],
            [1, 1, 0.5],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ])

        new_elems = []
        for ide, elm in enumerate(self.geofit_mesh.elements):

            # Evaluate the node locations for the new element based on the xi-eval points
            points = self.geofit_mesh.evaluate_embeddings([ide], xi_eval)

            # Debugging - visualize the points and the nodes
            # s = pv.Plotter()
            # s.add_mesh(pv.PolyData(np.array(points)), color='red', point_size=5, render_points_as_spheres=True)
            # s.add_mesh(pv.PolyData(np.array([node.loc for node in temp_mesh.nodes])), color='blue', point_size=3, render_points_as_spheres=True)
            # s.show()

            # Get the node ids for the new element using the spatial hash
            node_ids = [spatial_hash[tuple(np.trunc(pt*1000).tolist())] for pt in points]

            # Create the new element with the new node ids and L2 basis function in xi3 direction
            new_elem = MeshElement(node_indexes=node_ids, basis_functions=(H3Basis, H3Basis, L2Basis))
            new_elems.append(new_elem)

        # Rebuild the mesh with the refined nodes and new elements
        self.geofit_mesh = Mesh(nodes=temp_mesh.nodes, elements=new_elems)

        # generate the mesh from the nodes and elements
        self.geofit_mesh.generate_mesh()
        
        return self



    """
    
    FFD FITTING
    
    """

    def setup_FFD(self):

        """
        Evaluates the FFD fitting problem.

        This function sets up the FFD fitting problem.
        It embeds the start points into the mesh, evaluates the Sobolev space,
        and defines the fitting function that computes the difference between the
        mesh embeddings and the end points.

        Has options for Sobolev and volume differences.
        """

        # Embed the start points into the mesh
        elem, xis = self.geofit_mesh.embed_points(jnp.vstack(self.start_points), verbose=0) #change to verbose = 3 to visualize the embedding errors
        pre_sob = self.geofit_mesh.evaluate_sobolev()
        vol_0 = self.geofit_mesh.get_volume()

        def ffd(params):

            # Evaluate the mesh embeddings
            out_data = (self.geofit_mesh.evaluate_embeddings_ele_xi_pair(elem, xis, fit_params=params) - jnp.vstack(self.end_points)).flatten()

            # Initialize smoothing terms
            sobprior_dif = jnp.array([])
            vol_dif = jnp.array([])

            # Sobolev prior smoothing (geofit mesh as the prior)
            if self.config["smoothing"]["ffd_smoothing"] == "sobprior":
                sobprior_weight = self.config["smoothing"]["ffd_sobprior_weight"]
                sobprior_dif = (self.geofit_mesh.evaluate_sobolev(fit_params=params) - pre_sob) * sobprior_weight
            # Incompressibility constraint
            elif self.config["smoothing"]["ffd_smoothing"] == "incompressible":
                inc_weight = self.config["smoothing"]["ffd_incompressible_weight"]
                vol_dif = jnp.asarray([self.geofit_mesh.get_volume(fit_params=params) - vol_0]) * inc_weight

            return jnp.concatenate((out_data, sobprior_dif, vol_dif))

        # Initialize the parameters for the fitting function
        init_params = self.geofit_mesh.optimisable_param_array

        # Define jacobian for the FFD fitting problem
        func, jac = jacobian(ffd, init_estimate=init_params)

        return func, jac, init_params, (elem, xis)


    def fit_FFD(self, mylogger):
        """
        Perform the FFD fit.

        Uses least squares optimization to fit the mesh to the target points.
        Uses the function and jacobian provided by the setup_FFD function.
        """
        # Make a copy of the mesh to avoid modifying the original
        self.fitted_mesh = deepcopy(self.geofit_mesh)

        # Setup the FFD fit
        func, jac, init_params, (elem, xis) = self.setup_FFD()

        # Perform the optimization using least squares
        optim = least_squares(func, init_params, jac=jac, verbose=0, max_nfev=50)
        # Update the mesh with the optimized parameters
        self.fitted_mesh.update_from_params(optim.x)
        # Reshape the fitted errors to match the original shape
        self.fitted_errors = optim.fun.reshape(-1, 3)
        end_fitted_points = self.end_points.shape[0]

        # Calculate the magnitude of the fitted errors
        mag_error = np.linalg.norm(self.fitted_errors[:end_fitted_points], axis=-1)

        # Print root mean squared error
        self.fitted_RMSE = np.sqrt(np.mean(mag_error**2))
        mylogger.info(f"FFD Fit Root Mean Squared Error: {self.fitted_RMSE:.6f}")

        # Print root mean squared error
        NE_fitted_RMSE = np.sqrt(np.mean(self.fitted_errors**2))
        mylogger.info(f"FFD Fit Root Mean Squared Error (non-euclidian): {NE_fitted_RMSE:.6f}")

        # Store fitted points
        self.fitted_points = self.end_points + self.fitted_errors[:end_fitted_points]

        self.fitted_mesh.generate_mesh()

        return self


    """
    
    STRAIN ANALYSIS

    """

    def get_strain_points(self, config, mylogger):

        """
        Define strain points given config.
        """

        if config["strain"]["strain_points"] == "fitted_points":
            self.strain_points = self.fitted_points
            mylogger.info(f'Using fitted points for strain calculation. (Default)')
        elif config["strain"]["strain_points"] == "external":
            #self.strain_points = external_points # must be in format [frame][slice]np.array(x,y,z)
            mylogger.warning(f'"external" strain points configuration WIP. Exiting...')
            sys.exit(1)
        else:
            mylogger.warning(f'No valid strain points configuration found. Exiting...')
            sys.exit(1)
        return
    

    def get_strains(self, config, mylogger):

        """
        Calculate strains within the modelbased on the selected strain points.

        """

        if self.strain_points is None or (hasattr(self.strain_points, "size") and self.strain_points.size == 0):
            mylogger.warning(f'No valid start points found. Exiting...')
            sys.exit(1)


        def v_map(xi1, xi2, xi3):
            """
            Define orthogonal basis vectors v0, v1, v2 based on the material coordinate system defined by xi1, xi2, xi3.

            Local wall cooordinate system defines strains in a local cardiac coordinate system.
            """

            # v0 = xi1 = circumferential direction (theta)
            v0 = (xi1 / jnp.linalg.norm(xi1, axis=-1, keepdims=True))  # normalize

            # v1 = xi2 = longitudinal direction (z) - perpendicular to xi1, resticted to the xi1-xi2 plane
            xi2_orth = xi2 - jnp.sum(xi2 * xi1, axis=-1, keepdims=True) / jnp.sum(xi1 * xi1, axis=-1, keepdims=True) * xi1
            v1 = (xi2_orth / jnp.linalg.norm(xi2_orth, axis=-1, keepdims=True))  # normalize

            # v2 = xi3 = radial direction (r) - perpendicular to xi1 and xi2
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

        def convert_to_clr(mesh, eles, xis, tensors):
            """
            Function parsed into evaluate_strain_ele_xi_pair to convert the deformationtensors from material to local wall coordinates (CLR) at each embedded point.
               
            Note: only pre-multiplication of the basis vectors is needed to convert the deformation vectors from material to local wall coordinates.
            If applying to a 3D tensor, e.g. E, post-multiplying by the transpose is required.
            
            """ 
            _, v0, v1, v2 = get_clr_basis(mesh, eles, xis, ele_xi_pair=False)            
            fw_mat = jnp.linalg.inv(jnp.concatenate((
                v0[:, :, None],
                v1[:, :, None],
                v2[:, :, None],
            ), axis=-1).astype('float64'))

            spherical =  fw_mat @ tensors #Here is where you would also apply the post-multiplication if converting a 3D tensor, e.g. E, from material to local wall coordinates.

            return spherical
        

        # Get the elements and xi coordinates for the strain points
        eles, xis = self.fitted_mesh.embed_points(jnp.array(self.strain_points))
        
        # Get the world coordinates of the strain points in fitted mesh
        vs_world_undef = self.strain_points

        # Get the world coordinates of the strain points in geofit mesh
        vs_world_def = self.start_points

        # Get the strain tensors in the local wall coordinate system at the strain points
        vs = self.fitted_mesh.evaluate_strain_ele_xi_pair(eles, xis, self.geofit_mesh, convert_to_clr)        

        # Prepare the data for export
        strain_data = {
            'index': np.arange(len(vs)),
            'X': vs_world_undef[:, 0],
            'Y': vs_world_undef[:, 1],
            'Z': vs_world_undef[:, 2],
            'x': vs_world_def[:, 0],
            'y': vs_world_def[:, 1],
            'z': vs_world_def[:, 2],
            'E_cc': vs[:, 0, 0],
            'E_ll': vs[:, 1, 1],
            'E_rr': vs[:, 2, 2],
            'E_cl': vs[:, 0, 1],
            'E_cr': vs[:, 0, 2],
            'E_lr': vs[:, 1, 2],
        }

        self.strains = strain_data
        return self.strains
    

    def export_strains_csv(self, output_dir, mylogger):
        """
        Export the calculated strains to a CSV file.

        The CSV file will contain the strain components for each strain point.
        """
        if self.strains is None:
            mylogger.warning(f'No strains calculated to export. Exiting...')
            return

        # Define the output file path
        output_file = f"{output_dir}/strains_frame_{self.frame:03d}.csv"

        # Create a DataFrame and save to CSV
        df = pd.DataFrame(self.strains)
        df.to_csv(output_file, index=False)
        mylogger.info(f'Strains exported to {output_file}')

        return
    

    """
    
    PLOTTING

    """

    def draw_inner_contours(self, scene:pv.Plotter):
        """Draw inner/endo contours as blue dots."""
        breakpoint()
        endo_pv = pv.PolyData(np.vstack(list(self.data.endo.values())))
        scene.add_mesh(endo_pv, color='blue', point_size=5, render_points_as_spheres=True, name='Endo Contour')
        scene.add_text("Endo Contours", font_size=15, color='blue', position=[0.01, 0.95], viewport=True)

    def draw_outer_contours(self, scene:pv.Plotter):
        """Draw outer/epi contours as red dots."""
        epi_pv = pv.PolyData(np.vstack(list(self.data.epi.values())))
        scene.add_mesh(epi_pv, color='red', point_size=5, render_points_as_spheres=True, name='Epi Contour')
        scene.add_text("Epi Contours", font_size=15, color='red', position=[0.01, 0.93], viewport=True)

    def draw_start_points(self, scene:pv.Plotter):
        """Draw start points as gray dots."""
        start_points_pv = pv.PolyData(np.vstack(self.start_points))
        scene.add_mesh(start_points_pv, color='lightgray', point_size=22, render_points_as_spheres=True, name='Start Points')
        scene.add_text("Start Points", font_size=15, color='gray', position=[0.01, 0.95], viewport=True)

    def draw_end_points(self, scene:pv.Plotter):
        """Draw end points as green dots."""
        end_points_pv = pv.PolyData(np.vstack(self.end_points))
        scene.add_mesh(end_points_pv, color='green', point_size=22, render_points_as_spheres=True, name='End Points')
        scene.add_text("End Points", font_size=15, color='green', position=[0.01, 0.93], viewport=True)

    def draw_displacement_arrows(self, scene:pv.Plotter):
        """Draw displacement arrows."""
        disp_magnitude = np.linalg.norm(-np.vstack(self.displacements), axis=-1)
        start_points_pv = pv.PolyData(np.vstack(self.start_points))
        start_points_pv["vectors"] = -np.vstack(self.displacements)
        start_points_pv["disp_magnitude"] = disp_magnitude
        arrows = start_points_pv.glyph(orient="vectors", scale=1.0)
        scene.add_mesh(arrows, scalars="disp_magnitude", cmap="jet", name='Displacement Arrows', show_scalar_bar=False) 
        scene.add_text("Displacements", font_size=15, color='black', position=[0.01, 0.91], viewport=True)
        scene.add_scalar_bar(title="Displacement Magnitude", n_labels=5, position_x=0.25, width=0.5, title_font_size=30)

    def draw_error_arrows(self, scene:pv.Plotter):
        """Draw error arrows between end points and fitted points."""
        fitted_error_mag = np.linalg.norm(np.vstack(self.fitted_errors), axis=-1)
        end_points_pv = pv.PolyData(np.vstack(self.end_points))
        end_points_pv["vectors"] = np.vstack(self.fitted_errors[:self.end_points.shape[0]])
        end_points_pv["error_magnitude"] = fitted_error_mag[:self.end_points.shape[0]]
        arrows = end_points_pv.glyph(orient="vectors", scale=1.0)
        scene.add_mesh(arrows, scalars="error_magnitude", cmap="jet", name='Residual Error Arrows', show_scalar_bar=False) 
        scene.add_text("Residual Errors", font_size=15, color='black', position=[0.01, 0.89], viewport=True)
        scene.add_scalar_bar(title="Error Magnitude", n_labels=5, position_x=0.25, width=0.5, title_font_size=30)

    def draw_fitted_points(self, scene:pv.Plotter):
        """Draw model predicted points as blue dots."""
        fitted_points_pv = pv.PolyData(np.vstack(self.fitted_points))
        scene.add_mesh(fitted_points_pv, color='blue', point_size=22, render_points_as_spheres=True, name='Fitted Points')
        scene.add_text("Fitted Points", font_size=15, color='blue', position=[0.01, 0.87], viewport=True)


    def plot_template_mesh(self):
        """
        Plot the mesh using pyvista.

        Cannot seem to set the font size of the axes labels in pyvista, so we will just use the default.
        """

        scene = pv.Plotter()
        self.template_mesh.plot(scene)
        scene.add_points(pv.PolyData([0.0, 0.0, 0.0]), color='magenta', point_size=15, render_points_as_spheres=True, name='Origin')
        scene.add_text("Origin", font_size=15, color='magenta', position=[0.01, 0.97], viewport=True)
        scene.add_axes()
        
        scene.camera.SetParallelProjection(True)
        scene.show()

    def plot_template_mesh_and_contours(self):
        """
        Plot the mesh and contours using pyvista.
        """

        scene = pv.Plotter()
        self.template_mesh.plot(scene)
        scene.add_points(pv.PolyData([0.0, 0.0, 0.0]), color='magenta', point_size=15, render_points_as_spheres=True, name='Origin')
        scene.add_text("Origin", font_size=15, color='magenta', position=[0.01, 0.97], viewport=True)
        scene.add_axes()
        
        # Plot contours
        self.draw_inner_contours(scene)
        self.draw_outer_contours(scene)

        scene.camera.SetParallelProjection(True)
        scene.show()

    
    def plot_geofit_mesh(self):
        """
        Plot the geofit mesh using pyvista.
        """

        scene = pv.Plotter()
        self.geofit_mesh.plot(scene)

        # Plot origin point
        scene.add_points(pv.PolyData([0.0, 0.0, 0.0]), color='magenta', point_size=15, render_points_as_spheres=True, name='Origin')
        scene.add_text("Origin", font_size=15, color='magenta', position=[0.01, 0.97], viewport=True)
        scene.add_axes()

        # Plot contours
        self.draw_inner_contours(scene)
        self.draw_outer_contours(scene)

        scene.camera.SetParallelProjection(True)
        scene.show()


    def plot_geofit_mesh_and_disps(self):
        """
        Plot the mesh and displacements using pyvista.

        Cannot seem to set the font size of the axes labels in pyvista, so we will just use the default.
        """

        scene = pv.Plotter()
        self.geofit_mesh.plot(scene)

        # Plot origin point
        scene.add_points(pv.PolyData([0.0, 0.0, 0.0]), color='magenta', point_size=15, render_points_as_spheres=True, name='Origin')
        scene.add_text("Origin", font_size=15, color='magenta', position=[0.01, 0.97], viewport=True)
        scene.add_axes()

         # Plot start points
        self.draw_start_points(scene)

        # Plot end points
        self.draw_end_points(scene)

        # Plot displacements as vector arrows using glyphs
        self.draw_displacement_arrows(scene)

        scene.camera.SetParallelProjection(True)
        scene.show()


    def plot_fitted_mesh(self):
        """
        Plot the FFD fitted mesh and end points using pyvista.

        """

        scene = pv.Plotter()
        self.fitted_mesh.plot(scene, node_colour='green', mesh_colour='green')

        # Plot origin point
        scene.add_points(pv.PolyData([0.0, 0.0, 0.0]), color='magenta', point_size=15, render_points_as_spheres=True, name='Origin')
        scene.add_text("Origin", font_size=15, color='magenta', position=[0.01, 0.97], viewport=True)
        scene.add_axes()

        # Plot end points
        #self.draw_end_points(scene)

        # Plot residual errors as vector arrows using glyphs
        self.draw_error_arrows(scene)

        # Plot fitted points
        self.draw_fitted_points(scene)

        scene.camera.SetParallelProjection(True)
        scene.show() 

    
    def plot_fitted_sidebyside(self):
        """
        Plot the geofit mesh and end points side by side with the fitted mesh using pyvista.
        """

        sn = ['Before FFD Fitting', 'After FFD Fitting']
        scene = pv.Plotter(shape=(1,2))
        for tval in range(2):
            scene.subplot(0,tval)
            if tval == 0:
                self.geofit_mesh.plot(scene)
                # Plot start points
                self.draw_start_points(scene)
                # Plot end points
                self.draw_end_points(scene)
                # Plot displacements as vector arrows using glyphs
                self.draw_displacement_arrows(scene)
            elif tval == 1:
                self.fitted_mesh.plot(scene, node_colour='green', mesh_colour='green')
                # Plot residual errors as vector arrows using glyphs
                self.draw_error_arrows(scene)
                # Plot fitted points
                # self.draw_fitted_points(scene)
            # Plot origin point
            scene.add_points(pv.PolyData([0.0, 0.0, 0.0]), color='magenta', point_size=15, render_points_as_spheres=True, name='Origin')
            scene.add_text("Origin", font_size=15, color='magenta', position=[0.01, 0.97], viewport=True)
            scene.add_axes()  

        scene.link_views()
        scene.camera.SetParallelProjection(True)
        scene.show()



    """
    
    IO

    """

    def save_model(self, outputDir):
        # Save the model as a json
        
        model_data = {
            "frame": self.frame,

            "time": self.time,

            "Inputs": {
                "inner_radius": float(self.inputs.inner_radius),
                "outer_radius": float(self.inputs.outer_radius),
                "cylinder_height": float(self.inputs.cylinder_height),
                "cylinder_bot": float(self.inputs.cylinder_bot),
                "translation_vector": [float(x) for x in self.inputs.translation_vector],
                "num_of_elements": [int(x) for x in self.inputs.num_of_elements],
                "dist_to_data": float(self.inputs.dist_to_data),
            },

            "geofit_RMSE": float(self.geofit_RMSE),
            "fitted_RMSE": float(self.fitted_RMSE),

            "Data": {
                "endo": self.data.endo,
                "epi": self.data.epi,
                "loc": self.data.loc,
                "disp": self.data.disp
            },

            "endo_contours": [l.tolist() for l in self.endo_contours],
            "epi_contours": [l.tolist() for l in self.epi_contours],

            "displacements": [l.tolist() for l in self.displacements],
            "start_points": [l.tolist() for l in self.start_points],
            "end_points": [l.tolist() for l in self.end_points],
            "fitted_points": [l.tolist() for l in self.fitted_points],

            "template_mesh": dump_mesh_to_dict(self.template_mesh),

            "geofit_mesh": dump_mesh_to_dict(self.geofit_mesh),
            "geofit_errors": [l.tolist() for l in self.geofit_errors],

            "fitted_mesh": dump_mesh_to_dict(self.fitted_mesh),
            "fitted_errors": [l.tolist() for l in self.fitted_errors],

            "strain_points": (
                [l.tolist() if l is not None else None for l in self.strain_points]
                if self.strain_points is not None and hasattr(self.strain_points, "size") and self.strain_points.size > 0
                else None
            ),
            "strains": (
                [l.tolist() if l is not None else None for l in self.strains]
                if self.strains is not None and hasattr(self.strains, "size") and self.strains.size > 0
                else None
            ),
        }

        with open(outputDir / f"model_frame_{self.frame}.json", 'w') as f:
            json.dump(model_data, f, indent=4)

        return

