import sys
from pathlib import Path

from src.utils.io import import_models
from src.utils.plot import plot_meshes_with_slider, plot_local_coordinate_axes_strain_points, plot_strains_colourmaps, plot_strains_2D_transmural, plot_strains_2D_transmural_xi


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

        # Debug: Visualize coordinate vectors at strain points
        # plot_local_coordinate_axes_strain_points(model)
        if config["strain"]["plot_strains_all"]:
            # Debug: Visualize strain components as colour maps on the fitted mesh
            plot_strains_colourmaps(model)
            # Debug: Plot transmural strain profiles
            plot_strains_2D_transmural_xi(model)



    
    # # Optional: Perform strain-time analysis
    # if config["parameters"]["frames_to_fit"] == "all" and :
    #     model.perform_strain_time_analysis()

    mylogger.success('Strain pipeline complete.')

    return