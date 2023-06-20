###################################################################################################
###################################################################################################
### Definition of globals
###################################################################################################
###################################################################################################
GLB_DEFINE_PATH_PROJECT = False
PATH_PROJECT = ""
READ_FILE_MODE = "r"
PATH_DATASET = ""
PATH_DIR_LOGS = "logs"
PATH_DIR_MODELS = ""
INDEX_COLUMNS_DATASET = ""
LIST_NAME_COLUMNS_DATASET = ""
GLB_RETURN_ATTENTION_MASK = ""
GLB_ADD_SPECIAL_TOKENS = True
GLB_MAX_LENGTH_SENTENCE = 512
GLB_PADDING_TO_MAX_LENGTH = True
GLB_CROSS_VALIDATION = ""
GLB_SAVE_MODEL = ""
GLB_STORE_STATISTICS_MODEL = ""
GLB_TEST_MODEL = ""
GLB_SIZE_SPLITS_DATASET = 1
COL_OF_INTEREST = ""
CLASSIFICATION_TASK = ""
COL_OF_REFERENCE = ""
GLB_RUN_IN_GPU = True
LOGGER = None
GLB_MODEL_NAME = ""
GLB_LEARNING_RATE = 2e-5
GLB_WEIGHT_DECAY = 0
GLB_EPSILON_OPTIMIZER = 1e-8
        
###################################################################################################
###################################################################################################
### Imports
###################################################################################################
###################################################################################################
# Required packages
import yaml
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from os.path import join
import sys
import datetime as dt
import collections
import logging
from sklearn.exceptions import UndefinedMetricWarning
import warnings
# Custom code
try:
    import classification_model_utilities as mlclassif_utilities
except:
    import src.classification_model_utilities as mlclassif_utilities
try:
    import general_utilities as gral_utilities
except:
    import src.general_utilities as gral_utilities

###################################################################################################
###################################################################################################
### Logging
###################################################################################################
###################################################################################################
"""
Function:       configure_logger()
Description:    Configure logger of the project
Return:         None
"""
def configure_logger(levelStdout=logging.DEBUG, levelFile=logging.DEBUG):
    global LOGGER
    
    LOGGER = logging.getLogger(__name__)
    LOGGER.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    
    """
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(levelStdout)
    stdout_handler.setFormatter(formatter)
    """
    
    file_handler = logging.FileHandler(join(PATH_PROJECT, PATH_DIR_LOGS, gral_utilities.get_datetime_format() + '_activeLearning.log'))
    
    file_handler.setLevel(levelFile)
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    #LOGGER.addHandler(stdout_handler)
    
def infoLog(message):
    if LOGGER != None:
        LOGGER.info(message)
    else: 
        print(f"INFO  {message}")

def debugLog(message):
    if LOGGER != None:
        LOGGER.debug(message)
    else: 
        print(f"DEBUG {message}")
    
def errorLog(message):
    if LOGGER != None:
        LOGGER.error(message)
    else: 
        print(f"ERROR {message}")
    
def warnLog(message):
    if LOGGER != None:
        LOGGER.warn(message)
    else: 
        print(f"WARN  {message}")

###################################################################################################
###################################################################################################
### Control for warnings
###################################################################################################
###################################################################################################
def warn(*args, **kwargs):
    pass
warnings.warn = warn

###################################################################################################
###################################################################################################
### Functions for set-up
###################################################################################################
###################################################################################################

"""
Function:       get_projects_directory_path()
Description:    Defines the path of the project's directory
Return:         None (path is stored in a global variable)
"""
def get_projects_directory_path():
    if GLB_DEFINE_PATH_PROJECT:
        PATH_PROJECT = "/content/drive/MyDrive/Colab Notebooks/AutomatedTraumaDetectionInGCT"
    else:
        PATH_PROJECT = os.getcwd()
        
"""
Function:       read_config_file(config_file_path)
Description:    Reads the configuration file (yaml file)
Return:         None (configuration is stored in global variables)
"""
def read_config_file(config_file_path):
    if config_file_path == None:
        infoLog("Path of the configuration file is not defined. It's considered a default value ['config.yml']")
        config_file_path = "config.yml"
    
    cfg = None
    
    with open(join(PATH_PROJECT, config_file_path), READ_FILE_MODE) as ymlfile:
        cfg = yaml.safe_load(ymlfile)
        debugLog(cfg)
    
        global PATH_DATASET
        PATH_DATASET = join( PATH_PROJECT, cfg["general_set_up"]["input_dir_name"], cfg["general_set_up"]["dataset_dir_name"], cfg["general_set_up"]["dataset_filename"] )
        
        global PATH_DIR_LOGS
        PATH_DIR_LOGS = join( PATH_PROJECT, cfg["general_set_up"]["logs_dir_name"] )
        
        global PATH_DIR_MODELS
        PATH_DIR_MODELS = join( PATH_PROJECT, cfg["general_set_up"]["models_dir_name"] )
        
        global INDEX_COLUMNS_DATASET
        INDEX_COLUMNS_DATASET = cfg["dataset"]["index_columns_dataset"]
        
        global LIST_NAME_COLUMNS_DATASET
        LIST_NAME_COLUMNS_DATASET = cfg["dataset"]["list_columns_names"]
        
        global GLB_RETURN_ATTENTION_MASK
        GLB_RETURN_ATTENTION_MASK = cfg["training_model"]["return_attention_mask"]
        
        global GLB_CROSS_VALIDATION
        GLB_CROSS_VALIDATION = cfg["training_model"]["cross_validation"]
        
        global GLB_SAVE_MODEL
        GLB_SAVE_MODEL = cfg["training_model"]["save_model"]
        
        global GLB_STORE_STATISTICS_MODEL
        GLB_STORE_STATISTICS_MODEL = cfg["training_model"]["store_statistics"]
        
        global GLB_TEST_MODEL
        GLB_TEST_MODEL = cfg["training_model"]["test_model"]
        
        # Globals for the model
        global EPOCHS
        EPOCHS = cfg["training_model"]["epochs"]
        
        global EMBEDDING_SIZE
        EMBEDDING_SIZE = cfg["training_model"]["embedding_size"]
        
        global BATCH_SIZE
        BATCH_SIZE = cfg["training_model"]["batch_size"]
        
        global GLB_ADD_SPECIAL_TOKENS
        GLB_ADD_SPECIAL_TOKENS = cfg["training_model"]["add_special_tokes"]
        
        global GLB_MAX_LENGTH_SENTENCE
        GLB_MAX_LENGTH_SENTENCE = cfg["training_model"]["max_length"]
        
        global GLB_PADDING_TO_MAX_LENGTH
        GLB_PADDING_TO_MAX_LENGTH = cfg["training_model"]["pad_to_max_length"]
        
        global GLB_RUN_IN_GPU
        GLB_RUN_IN_GPU = cfg["training_model"]["run_in_gpu"]
        
        # Active training
        global GLB_SIZE_SPLITS_DATASET
        GLB_SIZE_SPLITS_DATASET = cfg["active_training"]["size_splits_dataset"]
        
        global CLASSIFICATION_TASK
        global COL_OF_INTEREST
        global COL_OF_REFERENCE
        CLASSIFICATION_TASK = cfg["active_training"]["classification_task"]
        if CLASSIFICATION_TASK == "binary":
            COL_OF_INTEREST = cfg["dataset"]["col_of_interest_binary_classif"]
            COL_OF_REFERENCE = cfg["dataset"]["col_of_reference_binary_classif"]
        elif CLASSIFICATION_TASK == "multi":
            COL_OF_INTEREST = cfg["dataset"]["col_of_interest_multi_label_classif"]
            COL_OF_REFERENCE = cfg["dataset"]["col_of_reference_multi_label_classif"]

        global GLB_MODEL_NAME
        GLB_MODEL_NAME = cfg["training_model"]["model_name"]
        
        global GLB_LEARNING_RATE
        GLB_LEARNING_RATE = float(cfg["training_model"]["learning_rate"])
        
        global GLB_WEIGHT_DECAY
        GLB_WEIGHT_DECAY = float(cfg["training_model"]["weight_decay"])
        
        global GLB_EPSILON_OPTIMIZER
        GLB_EPSILON_OPTIMIZER = float(cfg["training_model"]["epsilon_optimizer"])
        
    return cfg

"""
Function:       read_input_arguments()
Description:    Reads the arguments that user gives when execute this file
Return:         List - of input arguments
"""  
def read_input_arguments():
    return sys.argv

###################################################################################################
###################################################################################################
### Main function
###################################################################################################
###################################################################################################

def main(input_par_model_name=None):
    global LOGGER
    global GLB_MODEL_NAME
    LOGGER = None
    
    configure_logger()
    mlclassif_utilities.setLogger(LOGGER)
    
    list_statistics = list()
    
    # Get project's path
    debugLog("Reading directory path")
    get_projects_directory_path()
    # Read configuration file
    cfg = read_config_file("config.yml")
    
    # Get input arguments
    #input_arguments = read_input_arguments()
    #print(f"The number of input arguments is [{len(input_arguments)}] whose content is {input_arguments}")
    
    df_dataset = mlclassif_utilities.import_dataset_from_excel(PATH_DATASET, INDEX_COLUMNS_DATASET, LIST_NAME_COLUMNS_DATASET)
    debugLog(f"Col of interest {COL_OF_INTEREST}")
    classes_dataset = [int(elem) if gral_utilities.isfloat(elem) else elem for elem in mlclassif_utilities.get_unique_values_from_dataset(df_dataset, COL_OF_INTEREST)]
    debugLog(f"classes_dataset {classes_dataset}")
    
    # Split the dataset for the Active Training purposes
    dict_of_segments = mlclassif_utilities.give_me_segments_of_df_per_class(df_dataset, GLB_SIZE_SPLITS_DATASET, COL_OF_INTEREST, COL_OF_REFERENCE)
    odlst = collections.OrderedDict(sorted(dict_of_segments.items()))
    list_statistics = list()
    
    if input_par_model_name != None:
        GLB_MODEL_NAME = input_par_model_name
        infoLog("********************** ATENTION **********************")
        infoLog(f"Classification is being executed in mode for INPUT_PARAMETERS. Model name => {GLB_MODEL_NAME}")
    GLB_ID_MODEL = mlclassif_utilities.get_id_model(cfg, GLB_MODEL_NAME)
    if GLB_ID_MODEL == None or GLB_ID_MODEL == "":
        infoLog("ID of the model was not found. Execution is finished.")
        return
    
    for index, df_at_index in odlst.items():
        if index == 0:
            df = dict_of_segments[index]
        else:
            df = pd.concat([df, dict_of_segments[index]])
        
        # Read config file again - in case it is desired to store a model or change something at execution time
        cfg = read_config_file("config.yml")
        if input_par_model_name != None:
            GLB_MODEL_NAME = input_par_model_name
            infoLog("********************** ATENTION **********************")
            infoLog(f"Classification is being executed in mode for INPUT_PARAMETERS. Model name => {GLB_MODEL_NAME}")
        GLB_ID_MODEL = mlclassif_utilities.get_id_model(cfg, GLB_MODEL_NAME)
        if GLB_ID_MODEL == None or GLB_ID_MODEL == "":
            infoLog("ID of the model was not found. Execution is finished.")
            return
        
        model, statistics, test_corpus = mlclassif_utilities.exec_train(
                                        df, 
                                        COL_OF_INTEREST, 
                                        COL_OF_REFERENCE, 
                                        GLB_ID_MODEL, 
                                        GLB_MODEL_NAME, 
                                        batch_size = cfg["training_model"]["batch_size"],
                                        epochs = cfg["training_model"]["epochs"],
                                        max_length_sentence = cfg["training_model"]["max_length"],
                                        cross_validation = cfg["training_model"]["cross_validation"],
                                        store_statistics_model = cfg["training_model"]["store_statistics"],
                                        learning_rate = GLB_LEARNING_RATE,
                                        epsilon_optimizer = GLB_EPSILON_OPTIMIZER,
                                        weight_decay = GLB_WEIGHT_DECAY
                                    )
        debugLog("="*100)
        debugLog("*"*80)
        debugLog(f"Iteration: {index+1}/{GLB_SIZE_SPLITS_DATASET}")
        infoLog(statistics)
        debugLog("*"*80)
        debugLog("="*100)
        
        
        testing_stats = None
        if cfg["training_model"]["test_model"]:
            infoLog("To execute testing of the model on test_data")
            device = mlclassif_utilities.get_gpu_device_if_exists()
            
            test_dataset = mlclassif_utilities.create_tensor_dataset(test_corpus[1], test_corpus[2], test_corpus[0])
            test_dataloader = mlclassif_utilities.create_dataloader(test_dataset, cfg["training_model"]["batch_size"])
            
            testing_stats = mlclassif_utilities.test_model(model, device, test_dataloader)
            
            aux_json = {"training_and_val": statistics, "test": testing_stats}
            list_statistics.append(aux_json)
        else:
            list_statistics.append(statistics)
        
        
    infoLog(f"From active learning we present the following statistics: {statistics}")
    if GLB_STORE_STATISTICS_MODEL:
        json_object = {"active_learning": list_statistics}
        mlclassif_utilities.save_json_file_statistics_model(json_object, PATH_DIR_LOGS,
            pattern=f'{GLB_MODEL_NAME}-active_learning-{CLASSIFICATION_TASK}-epochs{cfg["training_model"]["epochs"]}-batchSize{cfg["training_model"]["batch_size"]}')
        
    if GLB_SAVE_MODEL:
        mlclassif_utilities.save_model(model, f"model_active_lrn_{CLASSIFICATION_TASK}_{GLB_MODEL_NAME}_Epochs-{EPOCHS}", PATH_DIR_MODELS)
        
if __name__ == "__main__":
    main()