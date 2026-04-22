import sys
from pathlib import Path

from mobstr3D.utils.io import import_models
from mobstr3D.utils.plot import plot_meshes_with_slider, plot_local_coordinate_axes_strain_points, plot_strains_2D_transmural_heatmap, plot_strains_2D_transmural_xi_heatmap, plot_strains_colourmaps, plot_strains_2D_transmural, plot_strains_2D_transmural_xi


def perform_strain(config, mylogger):

    """
    Runs the strain pipeline.

    """

    mylogger.info('Starting strain pipeline...')

    inputDir = Path(config["fitting_paths"]["output_dir"])
    outputDir = Path(config["strain"]["output_dir"])
    outputDir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load in models
    models = import_models(inputDir, mylogger)

    # Flag: Plot FFD fitted models side by side as a slider
    if config["plotting_flags"]["plot_slider"] and config["parameters"]["frames_to_fit"] == "all" and not config["modules"]["fitting"]:
        plot_meshes_with_slider(models)

    # Select frames to analyse
    if config["parameters"]["frames_to_fit"] == "all":
        frames = range(config["parameters"]["frame_of_seed"], len(models) + config["parameters"]["frame_of_seed"])
    elif config["parameters"]["frames_to_fit"] == "single":
        frames = [config["parameters"]["frame_of_interest"]]
    else:
        mylogger.error(f'Invalid frame selection: {config["parameters"]["frames_to_fit"]}')
        sys.exit(1)

    for frame_idx in frames:

        model = models[str(frame_idx)]

        mylogger.info(f'Processing frame {model.frame}...')

        # Step 2: Define strain points
        model.get_strain_points(config, mylogger)

        # Step 3: Perform analysis on each model
        model.get_strains(config, mylogger)

        # Step 4: Export strain results - rexport to model files to JSON
        model.save_model(inputDir)

        # Optional: Export strain results to CSV
        model.export_strains_csv(outputDir, mylogger)

        # Debug: Visualise local wall coordinate vectors at strain points
        if config["strain"]["plot_wall_vectors"]:
            plot_local_coordinate_axes_strain_points(model)

        # Optional: Visualise strains
        if config["strain"]["plot_strains_all"]:
            # Visualise strain components as colour maps on the fitted mesh
            plot_strains_colourmaps(model)

            if config["strain"]["transmural_parameter"] == "radius":
                # Plot transmural strain profiles using a computed radius
                plot_strains_2D_transmural(model)
            elif config["strain"]["transmural_parameter"] == "xi3":
                # Plot transmural strain profiles using xi3 as the radial material coordinate
                plot_strains_2D_transmural_xi(model)
            elif config["strain"]["transmural_parameter"] == "normalised_radius":
                # Plot transmural strain profiles using a computed radius normalized to 0-1 betweneen endo and epi
                # plot_strains_2D_transmural_normalised(model)
                mylogger.info('Normalised radius transmural parameter selected for evaluating strains. This is currently a WIP - change in config - skipping transmural plots.')
            else:
                mylogger.info('Invalid transmural parameter selected for evaluating strains - check config - skipping transmural plots.')

        if config["strain"]["plot_strains_all_heatmap"]:
            if config["strain"]["transmural_parameter"] == "radius":
                # Plot transmural strain profiles using a computed radius
                plot_strains_2D_transmural_heatmap(model)
            elif config["strain"]["transmural_parameter"] == "xi3":
                # Plot transmural strain profiles using xi3 as the radial material coordinate
                plot_strains_2D_transmural_xi_heatmap(model)
            elif config["strain"]["transmural_parameter"] == "normalised_radius":
                # Plot transmural strain profiles using a computed radius normalized to 0-1 betweneen endo and epi
                # plot_strains_2D_transmural_normalised(model)
                mylogger.info('Normalised radius transmural parameter selected for evaluating strains. This is currently a WIP - change in config - skipping transmural plots.')
            else:
                mylogger.info('Invalid transmural parameter selected for evaluating strains - check config - skipping transmural plots.')

    # TODO # Optional: Perform strain-time analysis
    # if config["parameters"]["frames_to_fit"] == "all" and :
    #     model.perform_strain_time_analysis()

    mylogger.success('Strain pipeline complete.')

    return