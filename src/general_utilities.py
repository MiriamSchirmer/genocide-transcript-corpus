import datetime as dt
import logging
import os
from os.path import join
import yaml
import json

##==========================================================================================================
"""
Function:       isfloat()
Description:    Get bool that states whether the sent parameter is float or not
Return:         Boolean - True if isfloat(num) else False
"""
def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

##==========================================================================================================
"""
Function:       get_datetime_format()
Description:    Get the date and time with the format for storing files
Return:         String - Date and time
"""
def get_datetime_format():
    return dt.datetime.now().strftime("%Y%m%d%H%M%S")

##==========================================================================================================
"""
Function:       configure_logger()
Description:    Configure logger of the project
Return:         None
"""
def configure_logger(levelStdout=logging.DEBUG, levelFile=logging.DEBUG, path_project=".", path_dir_logs="logs/", _datetime="YYYYMMDD", pattern="binaryClassif"):
    global LOGGER
    
    LOGGER = logging.getLogger(__name__)
    LOGGER.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    
    """
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(levelStdout)
    stdout_handler.setFormatter(formatter)
    """
    
    file_handler = logging.FileHandler(join(path_project, path_dir_logs, _datetime + '_' + pattern + '.log'))
    
    file_handler.setLevel(levelFile)
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    #LOGGER.addHandler(stdout_handler)

    return LOGGER

##==========================================================================================================
"""
Function:       read_config_file()
Description:    Read configuration file
Parameters:     config_file_path - path to the config file
Return:         Either:
                - cfg   - the configuration JSON content
                - None  - In case, there was a problem while opening/reading the file
"""
def read_config_file(config_file_path):
    if config_file_path == None:
        config_file_path = "config.yml"
    
    with open(join(config_file_path), "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
        return cfg

    return None

##==========================================================================================================
"""
Function:       read_json_file(path_json_file)
Description:    Configure logger of the project
Parameters:     config_file_path - path to the config file
Return:         Either:
                - cfg   - the configuration JSON content
                - None  - In case, there was a problem while opening/reading the file
"""
def read_json_file(path_json_file):
    json_object = None
    try:
        json_object = json.load(open(path_json_file))

        if(type(json_object) == str):
            json_object = json.loads(json_object)

    except Exception as err:
        print("ERROR - Error while reading the JSON file")
        print(err)
        json_object = None
    return json_object
        
##==========================================================================================================
"""
Function:       print_structure_json(json_object)
Description:    Go through the structure of the JSON object and print key and datatype
Parameters:     json_object - Refers to a JSON object
Return:         None
"""
def print_structure_json(json_object, level=0, identifier="", limit_depth=1000):
    #print(json_object, level, type(json_object))
    if level == 0:
        if type(json_object) != dict:
            print("Error - Object received is not a Dict Object", type(json_object))
            return None

    if limit_depth <= level:
        return None

    sep = "\t"*level
    if len(identifier) == 0:
        print(f'{sep}{type(json_object)}')
    else:
        print(f'{sep}{identifier} - {type(json_object)}')
    
    if type(json_object) == dict:            
        for key in json_object.keys():
            if type(json_object[key]) == dict or type(json_object[key]) == list:
                print_structure_json(json_object[key], level=level+1, identifier=key, limit_depth=limit_depth)
            else:
                if limit_depth <= (level+1):
                    continue

                aux_sep = "\t"*(level+1)
                print(f'{aux_sep}{key} {type(json_object[key])}')
    elif type(json_object) == list:
        """
        sep = "\t"*level
        if len(identifier) == 0:
            print(f'{sep}{type(json_object)}')
        else:
            print(f'{sep}{identifier} - {type(json_object)}')
        """
        for index_elem, elem in enumerate(json_object):
            if type(elem) == dict or type(elem) == list:
                print_structure_json(elem, level=level+1, identifier=f'{index_elem}', limit_depth=limit_depth)
            else:
                aux_sep = "\t"*(level+1)
                print(f'{aux_sep}{type(elem)}')
    else:
        aux_sep = "\t"*(level+1)
        print(f'{aux_sep}{identifier} - {type(json_object)}')
