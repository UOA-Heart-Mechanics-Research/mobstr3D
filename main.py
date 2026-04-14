import sys
from pathlib import Path
import tomli
import argparse
import shutil
from loguru import logger

# Import mobstr3D main modules
from mobstr3D.preprocessing_pipeline import perform_preprocessing
from mobstr3D.fitting_pipeline import perform_fitting
from mobstr3D.strain_pipeline import perform_strain


def main():
    parser = argparse.ArgumentParser(description="run mobstr3D modules based on config")
    parser.add_argument('-config', '--config_file', type=str, default='config.toml', help='Path to config file')
    args = parser.parse_args()

    # Load config
    assert Path(args.config_file).exists(), \
        f'Error loading config: {args.config_file}!'
    with open(args.config_file, mode="rb") as filepath:
        logger.info(f'Loading config file: {args.config_file}')
        config = tomli.load(filepath)

    # Check config schema
    match config:
        case {

            "modules": {"preprocessing": bool(), "fitting": bool(), "strain_extraction": bool()},

            "parameters": {"frames_to_fit": str(), "frame_of_interest": int()},

            #PREPROCESSING MODULE
            "preprocessing_input": {"preprocessing_input": str()},
            "preprocessing_paths": {"dicom_dir": str(), "input_dir": str(), "output_dir": str()},

            #FITTING MODULE
            "fitting_paths": {"dicom_dir": str(), "input_disp_dir": str(), "input_cont_dir": str(), "output_dir": str()},
            "plotting_flags": {"plot_template_all": bool(), "plot_template_final": bool(), "plot_geometric": bool(), "plot_ffd": bool(), "plot_slider": bool()},
            "model_parameters": {"num_of_elements": list(), "dist_to_data": float()},
            "smoothing": {"geo_smoothing": str(), "geo_sobprior_weight": float(), "ffd_smoothing": str(), "ffd_sob_weight": float(), "ffd_sobprior_weight": float(), "ffd_incompressible_weight": float()},

            #STRAIN MODULE
            "strain": {"strain_points": str(), "output_dir": str(), "csv": bool()},

            #LOGGING
            "logging": {"show_detailed_logging": bool(), "generate_log_file": bool(), "output_dir": str()},
        }:
            pass
        case _:
            logger.error(f'Config file inputs are incorrect. Please check for allowed values or missing fields.')
            sys.exit(1)


    # Run modules based on config
    modules = config.get('modules', {})
    logger.info(f'Running modules: preprocessing={modules["preprocessing"]}, fitting={modules["fitting"]}, strain_extraction={modules["strain_extraction"]}')

    # Run preprocessing if enabled
    if modules.get('preprocessing', False):
        logger.info('Starting preprocessing module...')
        run_preprocessing(config,logger)
        # if not config["logging"]["show_detailed_logging"]:
        #     logger.add(sys.stderr)
        logger.success("Preprocessing complete.")
        # Save copy of config to output folder
        output_folder = Path(config["preprocessing_paths"]["output_dir"])
        output_folder.mkdir(parents=True, exist_ok=True)
        shutil.copy(args.config_file, output_folder)

    # Run fitting if enabled
    if modules.get('fitting', False):
        logger.info('Starting fitting module...')
        run_fitting(config,logger)
        # if not config["logging"]["show_detailed_logging"]:
        #     logger.add(sys.stderr) 
        logger.success("Fitting complete.")
        # Save copy of config to output folder
        output_folder = Path(config["fitting_paths"]["output_dir"])
        output_folder.mkdir(parents=True, exist_ok=True)
        shutil.copy(args.config_file, output_folder)

    # Run strain extraction if enabled
    if modules.get('strain_extraction', False):
        logger.info('Starting strain extraction module...') #WIP
        run_strain(config,logger)  #WIP

    return



def run_preprocessing(config,mylogger):
    try:
        perform_preprocessing(config,mylogger)
    except KeyboardInterrupt:
        mylogger.warning("Preprocessing interrupted by user.")
        sys.exit(0)
    return

def run_fitting(config,mylogger):
    try:
        perform_fitting(config,mylogger)
    except KeyboardInterrupt:
        mylogger.warning("Fitting interrupted by user.")
        sys.exit(0)
    return

def run_strain(config,mylogger):
    try:
        perform_strain(config,mylogger)
    except KeyboardInterrupt:
        mylogger.warning("Analysis interrupted by user.")
        sys.exit(0)
    return


if __name__ == "__main__":
    main()