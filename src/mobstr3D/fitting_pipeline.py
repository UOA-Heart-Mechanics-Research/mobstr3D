import sys
from pathlib import Path
import numpy as np
import jax.numpy as jnp

from mobstr3D.fitting.setup_fitting import load_contour, load_disp, get_template_parameters
from mobstr3D.fitting.CylinderModel import CylinderModel
from mobstr3D.utils.plot import plot_meshes_with_slider




def perform_fitting(config, mylogger):
    
    mylogger.info(f'Starting fitting...')

    dispPath = config["fitting_paths"]["input_disp_dir"]+"/displacements.json"
    contPath = config["fitting_paths"]["input_cont_dir"]+"/contours.json"

    pflagTempAll = config["plotting_flags"]["plot_template_all"]
    pflagTemp = config["plotting_flags"]["plot_template_final"]
    pflagGeo = config["plotting_flags"]["plot_geometric"]
    pflagFFD = config["plotting_flags"]["plot_ffd"]
    pflagSlider = config["plotting_flags"]["plot_slider"]

    """
    Runs the fitting pipeline.

    EXPECTED INPUTS:

    contours.json and displacements.json are expected to be in the specified format:

    contours.json {dict}
        - "endo": a dict of a dict of numpy arrays (n, 3), accessed as contours["endo"][str({frame_index})][f"Slice_{slice_index}"][x,y,z]
        - "epi": a dict of a dict of numpy arrays (n, 3), accessed as contours["epi"][str({frame_index})][f"Slice_{slice_index}"][x,y,z]

    displacements.json {dict}
        - "locations": a dict of a dict of numpy arrays (n, 3), accessed as displacements["locations"][str({frame_index})][f"Slice_{slice_index}"][x,y,z]
        - "displacements": a dict of a dict of numpy arrays (n, 3), accessed as displacements["displacements"][str({frame_index})][f"Slice_{slice_index}"][x,y,z]
        Note: each displacement vector should have an associated location coordinate.

    """

    # Load and check contours and displacements
    endo, epi = load_contour(contPath, mylogger)
    loc, disp = load_disp(dispPath, mylogger)

    # Predefine models
    models = {}

    # Determine frames to process based on config
    if config["parameters"]["frames_to_fit"] == "all":
        frames = range(config["parameters"]["frame_of_seed"], len(endo) + config["parameters"]["frame_of_seed"])
    elif config["parameters"]["frames_to_fit"] == "single":
        frames = [config["parameters"]["frame_of_interest"]]
    else:
        mylogger.error(f'Invalid frame selection: {config["parameters"]["frames_to_fit"]}')
        sys.exit(1)

    # Loop through frames
    for frame_idx in frames:

        mylogger.info(f'Fitting frame {frame_idx}...')

        endo_f = endo[str(frame_idx)] # shape [f"Slice_{slice_index}"][x,y,z]
        epi_f = epi[str(frame_idx)] # shape [f"Slice_{slice_index}"][x,y,z]
        loc_f = loc[str(frame_idx)] # shape [f"Slice_{slice_index}"][x,y,z]
        disp_f = disp[str(frame_idx)] # shape [f"Slice_{slice_index}"][x,y,z]

        ## Step 1: Define template mesh
        template_parameters = get_template_parameters(endo_f, epi_f, config)

        ## Step 2: Create model
        model = CylinderModel(frame_idx, template_parameters, endo_f, epi_f, loc_f, disp_f, config)

        ## Step 3: Generate template mesh
        
        # Generate the cylinder mesh template
        model.template_cylinder_H3H3L1()
        mylogger.info(f'Template: generated cylinder mesh template.')
        if pflagTempAll:
            model.plot_template_mesh()

        # Refine the mesh
        model.refine_cylinder_defined(config["model_parameters"]["num_of_elements"])
        mylogger.info(f'Template: refined cylinder mesh.')
        if pflagTempAll:
            model.plot_template_mesh()

        # # Increase order to L2 basis function in xi3 direction
        # model.increase_order_cylinder_L2_xi3_template(config["model_parameters"]["num_of_elements"])
        # mylogger.info(f'Template: increased order in xi3 to L2 basis function.')
        # if pflagTempAll:
        #     model.plot_template_mesh()

        # Translate the mesh by a given translation vector
        model.translate_cylinder()
        formatted_vector = [f"{val:.4f}" for val in model.inputs.translation_vector]
        mylogger.info(f'Template: translated cylinder mesh by ({", ".join(formatted_vector)}).')
        if pflagTempAll:
            model.plot_template_mesh()

        # Transform the mesh from cylindrical polar to rectangular cartesian coordinates
        model.transform_cylinder_rtz2xyz()
        mylogger.info(f'Template: transformed cylinder mesh from cylindrical polar to rectangular cartesian coordinates.')
        if pflagTempAll or pflagTemp:
            #model.plot_template_mesh()
            model.plot_template_mesh_and_contours()

        ## Step 4: Geometric fitting to contours
        model.fit_geofit(mylogger)
        mylogger.info(f'Geometric: performed geometric fitting to contours.')

        # Increase order to L2 basis function in xi3 direction after geofit
        model.increase_order_cylinder_L2_xi3_geofit(config["model_parameters"]["num_of_elements"])
        mylogger.info(f'Geometric: increased order in xi3 to L2 basis function.')
        if pflagGeo:
            model.plot_geofit_mesh()

        ## Step 5: FFD fitting
        model.fit_FFD(mylogger)
        mylogger.info(f'FFD: performed FFD fitting.')
        if pflagFFD:
            model.plot_geofit_mesh_and_disps()
            #model.plot_fitted_mesh()
            model.plot_fitted_sidebyside()

        ## Step 6: Store model
        models[str(frame_idx)] = model

    # Save models
    outputDir = Path(config["fitting_paths"]["output_dir"])
    outputDir.mkdir(parents=True, exist_ok=True)

    for frame, model in models.items():
        model.save_model(outputDir)

    # Plot FFD fitted models side by side as a slider
    if pflagSlider and config["parameters"]["frames_to_fit"] == "all":
        plot_meshes_with_slider(models)

    return