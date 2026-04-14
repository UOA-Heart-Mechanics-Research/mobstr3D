import json
from pathlib import Path
import sys
import numpy as np

from src.fitting.CylinderModel import CylinderModel
from HOMER.io import dump_mesh_to_dict, parse_mesh_from_dict


def import_models(inputDir, mylogger):

    model_files = list(Path(inputDir).glob("model_frame_*.json"))

    if not model_files:
        mylogger.error(f'No model files found in "{inputDir}".')
        sys.exit(1)
    
    # Sort files according to frame in .json
    sorted_files = sorted(model_files, key=lambda f: get_frame(json.load(open(f, "r")), mylogger), reverse=False)

    models = {}
    for midx, model_file in enumerate(sorted_files):
        if model_file.stat().st_size == 0:
            mylogger.error(f'File {model_file} is empty. Ending...')
            sys.exit(1)
        with open(model_file, "r") as file:
            model = load_model(file)
            frame = model.frame
            models[str(frame)] = model
    return models


def get_frame(data, mylogger):
    try:
        frame = data["frame"]
        return frame
    
    except KeyError as e:
        mylogger.error(f'Missing key in MATLAB data: {e}')
        sys.exit(1)



def load_model(inputFile):

    # Load model from json
    model_data = []
    model_data = json.load(inputFile)

    # Reconstruct inputs and data as needed
    frame = model_data["frame"]
    template_params = {
        "inner_radius": model_data["Inputs"]["inner_radius"],
        "outer_radius": model_data["Inputs"]["outer_radius"],
        "cylinder_height": model_data["Inputs"]["cylinder_height"],
        "cylinder_bot": model_data["Inputs"]["cylinder_bot"],
        "translation_vector": np.array(model_data["Inputs"]["translation_vector"], dtype=float)
    }
    config = {"model_parameters": {"num_of_elements": model_data["Inputs"]["num_of_elements"], "dist_to_data": model_data["Inputs"]["dist_to_data"]}}

    endo = model_data["Data"]["endo"]
    epi  = model_data["Data"]["epi"]
    loc  = model_data["Data"]["loc"]
    disp = model_data["Data"]["disp"]

    # Create a new CylinderModel instance
    model = CylinderModel(frame, template_params, endo, epi, loc, disp, config)

    # Define params
    model.frame = model_data["frame"]

    model.time = model_data["time"]

    model.Inputs.inner_radius = model_data["Inputs"]["inner_radius"]
    model.Inputs.outer_radius = model_data["Inputs"]["outer_radius"]
    model.Inputs.cylinder_height = model_data["Inputs"]["cylinder_height"]
    model.Inputs.cylinder_bot = model_data["Inputs"]["cylinder_bot"]
    model.Inputs.translation_vector = np.array(model_data["Inputs"]["translation_vector"], dtype=float)
    model.Inputs.num_of_elements = np.array(model_data["Inputs"]["num_of_elements"], dtype=int)
    model.Inputs.dist_to_data = np.array(model_data["Inputs"]["dist_to_data"], dtype=float)

    model.Data.endo = model_data["Data"]["endo"]
    model.Data.epi  = model_data["Data"]["epi"]
    model.Data.loc  = model_data["Data"]["loc"]
    model.Data.disp = model_data["Data"]["disp"]

    model.endo_contours = np.array(model_data["endo_contours"])
    model.epi_contours = np.array(model_data["epi_contours"])

    model.displacements = np.array(model_data["displacements"])
    model.start_points = np.array(model_data["start_points"])
    model.end_points = np.array(model_data["end_points"])
    model.fitted_points = np.array(model_data["fitted_points"])

    model.template_mesh = parse_mesh_from_dict(model_data["template_mesh"])
    model.geofit_mesh = parse_mesh_from_dict(model_data["geofit_mesh"])
    model.geofit_errors = np.array(model_data["geofit_errors"])
    model.fitted_mesh = parse_mesh_from_dict(model_data["fitted_mesh"])
    model.fitted_errors = np.array(model_data["fitted_errors"])
    model.geofit_RMSE = model_data["geofit_RMSE"]
    model.fitted_RMSE = model_data["fitted_RMSE"]

    model.strain_points = model_data["strain_points"] if model_data["strain_points"] else None
    model.strains = model_data["strains"] if model_data["strains"] else None

    return model