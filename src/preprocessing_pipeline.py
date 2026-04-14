import sys

# Import mobstr3D preprocessing modules
from src.preprocessing.input_denseanalysis import perform_denseanalysis_preprocessing
from src.preprocessing.input_DICOM import perform_DICOM_preprocessing


def perform_preprocessing(config,mylogger):

    """
    Runs the preprocessing pipeline given the configured input
    """

    # Define preprocessing input type
    preprocessing_input = config["preprocessing_input"]["preprocessing_input"].lower()
    mylogger.info(f'Preprocessing input type: {preprocessing_input}')

    # Run preprocessing based on input type
    match preprocessing_input:
        case "dicom":
            perform_DICOM_preprocessing(config,mylogger)
            mylogger.success("DICOM preprocessing complete.")

        case "denseanalysis":
            perform_denseanalysis_preprocessing(config,mylogger)
            mylogger.success("DENSEanalysis preprocessing complete.")

        case "bivme":
            mylogger.info('biv-me preprocessing is WIP - skipped.')
            #perform_bivme_preprocessing(config,mylogger) #WIP
            #mylogger.success("biv-me preprocessing complete.") #WIP
        case _:
            mylogger.error(f'Preprocessing input type "{preprocessing_input}" not recognized. Please check config file.')
            sys.exit(1)

    return