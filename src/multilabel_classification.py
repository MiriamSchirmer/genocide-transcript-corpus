
####### Imports
import yaml
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from os.path import join

import classification_model_utilities as mlclassif_utilities

####### Globals
GLB_DEFINE_PATH_PROJECT = False
PATH_PROJECT = ""

if GLB_DEFINE_PATH_PROJECT:
    PATH_PROJECT = ""
else:
    PATH_PROJECT = os.getcwd()

with open(join(PATH_PROJECT, "config.yml"), "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)

print(cfg)

PATH_DATASET = join( PATH_PROJECT, cfg["general_set_up"]["input_dir_name"], cfg["general_set_up"]["dataset_dir_name"], cfg["general_set_up"]["dataset_filename"] )
PATH_DIR_LOGS = join( PATH_PROJECT, cfg["general_set_up"]["logs_dir_name"] )
PATH_DIR_MODELS = join( PATH_PROJECT, cfg["general_set_up"]["models_dir_name"] )
INDEX_COLUMNS_DATASET = cfg["dataset"]["index_columns_dataset"]
LIST_NAME_COLUMNS_DATASET = cfg["dataset"]["list_columns_names"]
GLB_RETURN_ATTENTION_MASK = cfg["training_model"]["return_attention_mask"]
GLB_CROSS_VALIDATION = cfg["training_model"]["cross_validation"]
GLB_SAVE_MODEL = cfg["training_model"]["save_model"]
GLB_STORE_STATISTICS_MODEL = cfg["training_model"]["store_statistics"]
GLB_TEST_MODEL = cfg["training_model"]["test_model"]

# Globals for the model
EPOCHS = cfg["training_model"]["epochs"]
EMBEDDING_SIZE = cfg["training_model"]["embedding_size"]
BATCH_SIZE = cfg["training_model"]["batch_size"]

############################################################################################################################
############################################################################################################################
############################################################################################################################
############################################################################################################################
########
######## BEGIN Implementation
########
############################################################################################################################
############################################################################################################################

df_dataset = mlclassif_utilities.import_dataset_from_excel(PATH_DATASET, INDEX_COLUMNS_DATASET, LIST_NAME_COLUMNS_DATASET)
print(df_dataset.head())

classes_dataset = mlclassif_utilities.get_unique_values_from_dataset(df_dataset, "role")
print(f"Num of different roles in the dataset is {len(classes_dataset)} which are:")
for index, elem in enumerate(classes_dataset):
    print("\t", index+1, "-", elem)
    
courts_dataset = mlclassif_utilities.get_unique_values_from_dataset(df_dataset, "court")
print(f"Num of different courts in the dataset is {len(courts_dataset)} which are:")
for index, elem in enumerate(courts_dataset):
    print("\t", index+1, "-", elem)
    
####################################################################################
##### IMPORTANT: Number of classes for the classification task
NUM_CLASSES = len(classes_dataset) # This global corresponds to this specific task
####################################################################################

device = mlclassif_utilities.get_gpu_device_if_exists()

print(f"\n\n==> Selected device is '{device}' <==")


#If no parameters are sent, default values are considered. 
#    IDModel:      Bert
#    Model namel:  bert-base-uncased
#    Do uncase:    True

tokenizer = mlclassif_utilities.get_tokenizer() 

list_all_spans = list(df_dataset["span"])
list_all_classes = list(df_dataset["role"])

mlclassif_utilities.get_max_length_of_a_sentence_among_all_sentences(tokenizer, list_all_spans, False)

# If _return_attention_mask, a tuple of two lists is given (tensor_of_inputs, tensor_of_attention_masks)
all_spans_tokenized = mlclassif_utilities.get_all_spans_tokenized(
    mlclassif_utilities.GLB_BERT_MODEL_ID, 
    tokenizer,
    list_all_spans,
    _add_special_tokens = cfg["training_model"]["add_special_tokes"], 
    _max_length = cfg["training_model"]["max_length"],
    _pad_to_max_length = cfg["training_model"]["pad_to_max_length"],
    _return_attention_mask = GLB_RETURN_ATTENTION_MASK, 
    type_tensors = mlclassif_utilities.GLB_PYTORCH_TENSOR_TYPE
)

input_ids = None
attention_masks = None

if GLB_RETURN_ATTENTION_MASK:
    input_ids = mlclassif_utilities.convert_list_into_pytorch_tensor(all_spans_tokenized[0])
    attention_masks = mlclassif_utilities.convert_list_into_pytorch_tensor(all_spans_tokenized[1])
else:
    input_ids = mlclassif_utilities.convert_list_into_pytorch_tensor(all_spans_tokenized)

numeric_classes = mlclassif_utilities.convert_list_span_classes_into_numeric_values(classes_dataset, list_all_classes)
numeric_classes = mlclassif_utilities.convert_list_labels_into_pytorch_tensor(numeric_classes)

### Split dataset
if not GLB_CROSS_VALIDATION:
    train_labels_corpus, train_input_ids, train_attention_masks, val_labels_corpus, val_input_ids, val_attention_masks, test_labels_corpus, test_input_ids, test_attention_masks = mlclassif_utilities.split_dataset_train_val_test(numeric_classes, input_ids, attention_masks)
else:
    ### k-Fold
    train_val_corpus_cross_validation, test_corpus_cross_validation = mlclassif_utilities.split_dataset_train_val_test_k_fold(numeric_classes, input_ids, attention_masks, 0.1)

model = mlclassif_utilities.create_model(
    mlclassif_utilities.GLB_BERT_MODEL_ID,
    mlclassif_utilities.GLB_BERT_BASE_UNCASED_MODEL_NAME,
    NUM_CLASSES,
    cfg["training_model"]["run_in_gpu"] #RunInGPU
)


optimizer = mlclassif_utilities.get_optimizer(model)
scheduler = mlclassif_utilities.get_scheduler(optimizer)

if not GLB_CROSS_VALIDATION:
  train_dataset = mlclassif_utilities.create_tensor_dataset(train_input_ids, train_attention_masks, train_labels_corpus)
  val_dataset = mlclassif_utilities.create_tensor_dataset(val_input_ids, val_attention_masks, val_labels_corpus)
  test_dataset = mlclassif_utilities.create_tensor_dataset(test_input_ids, test_attention_masks, test_labels_corpus)

  train_dataloader = mlclassif_utilities.create_dataloader(train_dataset, BATCH_SIZE)
  val_dataloader = mlclassif_utilities.create_dataloader(val_dataset, BATCH_SIZE)
  test_dataloader = mlclassif_utilities.create_dataloader(test_dataset, BATCH_SIZE)

  model, statistics_model = mlclassif_utilities.train_and_validate(model, device, EPOCHS, optimizer, scheduler, train_dataloader, val_dataloader, numeric_classes.tolist())

else:
  for index_cross_val in range(len(train_val_corpus_cross_validation)):
    train_dataset = mlclassif_utilities.create_tensor_dataset(train_val_corpus_cross_validation[index_cross_val][1], train_val_corpus_cross_validation[index_cross_val][2], train_val_corpus_cross_validation[index_cross_val][0])
    val_dataset = mlclassif_utilities.create_tensor_dataset(train_val_corpus_cross_validation[index_cross_val][4], train_val_corpus_cross_validation[index_cross_val][5], train_val_corpus_cross_validation[index_cross_val][3])

    train_dataloader = mlclassif_utilities.create_dataloader(train_dataset, BATCH_SIZE)
    val_dataloader = mlclassif_utilities.create_dataloader(val_dataset, BATCH_SIZE)

    print('='*50)
    print(f"Cross-Validation Split {(index_cross_val+1)}/{len(train_val_corpus_cross_validation)}")
    print('='*50)
    model, statistics_model = mlclassif_utilities.train_and_validate(model, device, EPOCHS, optimizer, scheduler, train_dataloader, val_dataloader, numeric_classes.tolist())

if GLB_STORE_STATISTICS_MODEL:
    mlclassif_utilities.save_json_file_statistics_model(statistics_model, PATH_DIR_LOGS)

if GLB_TEST_MODEL:
    mlclassif_utilities.test_model(model, device, test_dataloader, numeric_classes.tolist())

if GLB_SAVE_MODEL:
    mlclassif_utilities.save_model(model, "model_bert_" + NUM_CLASSES + "_classes", PATH_DIR_MODELS)
