###############################################################
## IMPORTS

# numpy
import numpy as np
# pandas
import pandas as pd
# matplotlib
import matplotlib.pyplot as plt
# torch
import torch
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SubsetRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn
#Transformers
from transformers import BertForSequenceClassification, AdamW, BertTokenizer #, BertConfig
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup
# sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import plot_confusion_matrix as plot_confusion_matrix_sklearn
# general
import datetime
from datetime import date
import random
import time
import os
from os.path import join
import json
import traceback
#Custom
try:
    import general_utilities as gral_utilities
except:
    import src.general_utilities as gral_utilities


###############################################################
## GLOBALS
GLB_BERT_MODEL_ID_DEFAULT = "bert"
GLB_BERT_BASE_UNCASED_MODEL_NAME_DEFAULT = "bert-base-uncased"#"nlpaueb/legal-bert-small-uncased"
GLB_PYTORCH_TENSOR_TYPE = "pt"
GLB_DEVICE_CPU = "cpu"
GLB_FILE_READ_MODE = "r"
GLB_FILE_WRITE_MODE = "w"
GLB_LIST_MODES_4_SELECTING_DATASET = ["random", "first", "last"]
GLB_MIN_NUM_SAMPLES_PER_CLASS_4_TRAINING_AND_TESTING = 3
LOGGER = None

def setLogger(logger):
    global LOGGER
    LOGGER = logger

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

###############################################################
## UTILITIES GENERAL IMPLEMENTATION

##==========================================================================================================
"""
Function: import_dataset_from_excel
"""
def import_dataset_from_excel(path_dataset, header_index=None, columns_names_list=None):
    if header_index == None and columns_names_list == None:
        return pd.read_excel(path_dataset)
    return pd.read_excel(path_dataset, header=header_index, names=columns_names_list)

##==========================================================================================================
"""
Function: export_dataset_to_excel
"""
def export_dataset_to_excel(df, path_to_store_df=".", name_file="", _header=None, _index=True, add_timestamp_to_filename=True):
    if name_file == "":
        name_file = gral_utilities.get_datetime_format() + "_Dataset.xlsx"
    elif add_timestamp_to_filename:
        name_file = gral_utilities.get_datetime_format() + "_" + name_file

    if _header != None or len(_header) == 0:
        df.to_excel(join(path_to_store_df, name_file), index=_index)
    else:
        df.to_excel(join(path_to_store_df, name_file), header=_header, index=_index)
    infoLog("Export executed successfully")

##==========================================================================================================
"""
Function: get_gpu_device_if_exists
"""
def get_gpu_device_if_exists():
    # If there's a GPU available...
    if torch.cuda.is_available():    

        # Tell PyTorch to use the GPU.    
        device = torch.device("cuda")

        infoLog('There are %d GPU(s) available.\n\nThese are the available devices:' % torch.cuda.device_count())
        for index in range(torch.cuda.device_count()):
            infoLog(f'\t {index+1} - {torch.cuda.get_device_name(index)}')

    # If not...
    else:
        infoLog('No GPU available, using the CPU instead.')
        device = torch.device(GLB_DEVICE_CPU)
    
    return device
##==========================================================================================================
"""
Function: get_unique_values_from_dataset
"""
def get_unique_values_from_dataset(dataframe, column_name):
    return list(dataframe[column_name].unique())
##==========================================================================================================
"""
Function: get_distribution_classes_from_dataset
"""
def get_distribution_classes_from_dataset(dataframe, groupby_list_columns, chosen_column):
    return dataframe.groupby(groupby_list_columns).count()[chosen_column].reset_index()
##==========================================================================================================
"""
Function: get_max_length_of_a_sentence_among_all_sentences
"""
def get_max_length_of_a_sentence_among_all_sentences(tokenizer, list_all_sentences, add_special_tokens=True):
    # ==> Get the max length of a sentence
    max_len = 0
    list_length_sentences = list()

    # For every sentence...
    for sentence in list_all_sentences:

        # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
        input_ids = tokenizer.encode(sentence, add_special_tokens=add_special_tokens)

        # Update the maximum sentence length.
        list_length_sentences.append(len(input_ids))
        #max_len = max(max_len, len(input_ids))
        
    index_max = list_length_sentences.index(max(list_length_sentences))
    infoLog(f'Max sentence length: {list_length_sentences[index_max]} found at index {index_max}. Sentence is:\n\n\n{list_all_sentences[index_max]}\n\n\n')
    return list_length_sentences[index_max]
##==========================================================================================================
"""
Function: get_tokenizer given a model
"""
def get_tokenizer(model_id, model_name = GLB_BERT_BASE_UNCASED_MODEL_NAME_DEFAULT, lower_case=True):
    tokenizer = None
    try:
        debugLog(f'Loading tokenizer... [ID:{model_id}];[Name:{model_name}]')
        if(model_id == GLB_BERT_MODEL_ID_DEFAULT):
            tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=lower_case)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=lower_case)
        infoLog(f'{model_id} tokenizer was loaded successfully ({model_name})\n\tdo_lower_case={lower_case}')


    except Exception as excmsg:
        errorLog(f"An error happens in get_tokenizer(...) {traceback.format_exc()}.\nLet's proceed to load BERT tokenizer (default one)")

        debugLog('Loading BERT tokenizer...')
        tokenizer = BertTokenizer.from_pretrained(GLB_BERT_BASE_UNCASED_MODEL_NAME_DEFAULT, do_lower_case=True)
        infoLog(f'Tokenizer was loaded successfully ({GLB_BERT_BASE_UNCASED_MODEL_NAME_DEFAULT})\n\tdo_lower_case=True')

    return tokenizer
##==========================================================================================================
"""
Function: get_all_spans_tokenized
Note: 
 - If tokenizer=get_tokenizer(), it's created another instance of the tokenizer
 - If tokenizer=get_tokenizer, it's not created another instance unless what is sent is not an instance of some model's tokenizer
"""
def get_all_spans_tokenized(tokenizer=get_tokenizer, all_spans=[], _add_special_tokens=True, _max_length=512, _pad_to_max_length = True, _return_attention_mask=True, type_tensors=GLB_PYTORCH_TENSOR_TYPE):
    input_ids = []
    attention_masks = []

    # For every sentence...
    for span in all_spans:
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
                            span,                      # Sentence to encode.
                            add_special_tokens = _add_special_tokens, # Add '[CLS]' and '[SEP]'
                            max_length = _max_length,          # Pad & truncate all sentences.
                            pad_to_max_length = _pad_to_max_length,  #is deprecated
                            return_attention_mask = _return_attention_mask,   # Construct attn. masks.
                            return_tensors = type_tensors,     # Return pytorch tensors.
                    )
        
        # Add the encoded sentence to the list.    
            # Add the encoded sentence to the list.    
        # Add the encoded sentence to the list.    
        input_ids.append(encoded_dict['input_ids'])
        
        if _return_attention_mask:
            # And its attention mask (simply differentiates padding from non-padding).
            attention_masks.append(encoded_dict['attention_mask'])


    if _return_attention_mask:
        return input_ids, attention_masks
    else:
        return input_ids
##==========================================================================================================
"""
Function: convert_list_into_pytorch_tensor 
"""
def convert_list_into_pytorch_tensor(input_list):
    if type(input_list) != list:
        warnLog(f"Warning! - The input parameter does not correspond to the expected type value. Found {type(input_list)}")
        return None
    
    return torch.cat(input_list, dim=0)
##==========================================================================================================
"""
Function: convert_list_span_classes_into_numeric_values
"""
def convert_list_span_classes_into_numeric_values(list_classes, list_spans):
    dict_classes = {}
    for index, elem in enumerate(list_classes):
        dict_classes[elem] = index

    return [dict_classes[it] for it in list_spans]
##==========================================================================================================
"""
Function: convert_list_labels_into_pytorch_tensor 
"""
def convert_list_labels_into_pytorch_tensor(input_list):
    if type(input_list) != list:
        warnLog(f"Warning! - The input parameter does not correspond to the expected type value. Found {type(input_list)}")
        return None
    
    return torch.tensor(input_list)
##==========================================================================================================
"""
Function: create_dataset 
"""
def create_tensor_dataset(input_ids, attention_masks, labels):
    return TensorDataset(input_ids, attention_masks, labels)
##==========================================================================================================
"""
Function: split_dataset_train_val_test 
"""
def split_dataset_train_val_test(labels, input_ids, attention_masks, test_size_percentage=0.05, val_size_percentage=0.1, debug=True):
    train_valid_indices, test_indices = train_test_split(
        np.arange(len(labels)), 
        test_size=test_size_percentage, 
        shuffle=True, 
        stratify=labels
    )

    # TRAINING AND VALIDATION CORPUS (labels, input_ids, attention_masks)
    train_valid_labels = labels[train_valid_indices]
    train_valid_input_ids = input_ids[train_valid_indices]
    train_valid_attention_masks = attention_masks[train_valid_indices]

    # TEST: (labels, input_ids, attention_masks)
    test_labels_corpus = labels[test_indices] 
    test_input_ids = input_ids[test_indices]
    test_attention_masks = attention_masks[test_indices]

    train_indices, valid_indices = train_test_split(
        np.arange(len(train_valid_labels)), 
        test_size=val_size_percentage, 
        shuffle=True, 
        stratify=train_valid_labels
    )

    # TRAIN: (labels, input_ids, attention_masks)
    train_labels_corpus = train_valid_labels[train_indices] 
    train_input_ids = train_valid_input_ids[train_indices]
    train_attention_masks = train_valid_attention_masks[train_indices]

    # VALIDATION: (labels, input_ids, attention_masks)
    val_labels_corpus = train_valid_labels[valid_indices] 
    val_input_ids = train_valid_input_ids[valid_indices]
    val_attention_masks = train_valid_attention_masks[valid_indices]

    if debug == True:
        infoLog(f'''\nCORPUS TRAINING AND VALIDATION: 
            \n\t Length labels {len(train_valid_labels)}
            \n\t Length input_ids {len(train_valid_input_ids)}
            \n\t Length attention_masks {len(train_valid_attention_masks)}
            \n'''
            )

        infoLog(f'''\n\tCORPUS TRAINING:  
            \n\t\t Length labels {len(train_labels_corpus)}
            \n\t\t Length input_ids {len(train_input_ids)}
            \n\t\t Length attention_masks {len(train_attention_masks)}'''
            )

        infoLog(f'''\n\tCORPUS VALIDATION: 
            \n\t\t Length labels {len(val_labels_corpus)}
            \n\t\t Length input_ids {len(val_input_ids)}
            \n\t\t Length attention_masks {len(val_attention_masks)}'''
            )

        infoLog("")

        infoLog(f'''\nCORPUS TEST: 
            \n\t Length labels {len(test_labels_corpus)}
            \n\t Length input_ids {len(test_input_ids)}
            \n\t Length attention_masks {len(test_input_ids)}
            \n'''
        )

    return train_labels_corpus, train_input_ids, train_attention_masks, val_labels_corpus, val_input_ids, val_attention_masks, test_labels_corpus, test_input_ids, test_attention_masks
##==========================================================================================================
"""
Function: split_dataset_train_val_test_k_fold
"""
def split_dataset_train_val_test_k_fold(labels, input_ids, attention_masks, percentage_test=0.1, number_splits=5, rdm_state=42, _shuffle=True):    
    train_valid_indices, test_indices = train_test_split(
        np.arange(len(labels)), 
        test_size=percentage_test, 
        shuffle=_shuffle, 
        stratify=labels
    )
    
    # TRAINING AND VALIDATION CORPUS (labels, input_ids, attention_masks)
    train_valid_labels = labels[train_valid_indices]
    train_valid_input_ids = input_ids[train_valid_indices]
    train_valid_attention_masks = attention_masks[train_valid_indices]

    # TEST: (labels, input_ids, attention_masks)
    test_labels_corpus = labels[test_indices] 
    test_input_ids = input_ids[test_indices]
    test_attention_masks = attention_masks[test_indices]
    
    #=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
    
    cv_folds = list()
    kf = KFold(n_splits=number_splits, shuffle=_shuffle, random_state=rdm_state)
    
    for index, (train_index, val_index) in enumerate(kf.split(train_valid_labels)):
        x_train_input_ids,    x_val_input_ids    = train_valid_input_ids[train_index], train_valid_input_ids[val_index]
        y_train_labels,       y_val_labels       = train_valid_labels[train_index], train_valid_labels[val_index]
        z_train_attntn_masks, z_val_attntn_masks = train_valid_attention_masks[train_index], train_valid_attention_masks[val_index]
        
        debugLog(f'{index}) Len of train_index={len(train_index)} VS len of val_index={len(val_index)}')
        
        cv_folds.append([y_train_labels, x_train_input_ids, z_train_attntn_masks, y_val_labels, x_val_input_ids, z_val_attntn_masks])
        
    return cv_folds, [test_labels_corpus, test_input_ids, test_attention_masks]
        
    
##==========================================================================================================
"""
Function: create_dataloader
"""
def create_dataloader(dataset, batch_size, sampler=RandomSampler):#RandomSampler
    return DataLoader(
            dataset,  # The samples.
            sampler = sampler(dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )
##==========================================================================================================
"""
Function: create_model
"""
def create_model(model_id, model_name, _num_classes, runInGpu, _output_attentions=False, _output_hidden_states=False):
    model = None

    if model_id == GLB_BERT_MODEL_ID_DEFAULT:
        # Load BertForSequenceClassification, the pretrained BERT model with a single 
        # linear classification layer on top. 
        model = BertForSequenceClassification.from_pretrained(
            model_name, # Use the 12-layer BERT model, with an uncased vocab.
            num_labels = _num_classes, # The number of output labels   
            output_attentions = _output_attentions, # Whether the model returns attentions weights.
            output_hidden_states = _output_hidden_states, # Whether the model returns all hidden-states.
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, # Use the 12-layer BERT model, with an uncased vocab.
            num_labels = _num_classes, # The number of output labels   
            output_attentions = _output_attentions, # Whether the model returns attentions weights.
            output_hidden_states = _output_hidden_states, # Whether the model returns all hidden-states.
        )


    if runInGpu:
        # Tell pytorch to run this model on the GPU.
        model.cuda()

    return model
        
##==========================================================================================================
"""
Function: get_optimizer
"""
def get_optimizer(model, learning_rate = 2e-5, epsilon=1e-8, _weight_decay=0):
    learning_rate = float(learning_rate)
    optimizer = AdamW(model.parameters(),
                  lr = learning_rate,#2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = epsilon, # args.adam_epsilon  - default is 1e-8.
                  weight_decay=_weight_decay#0.01
                )
    debugLog("Optimizer")
    debugLog(optimizer)
    return optimizer
##==========================================================================================================
"""
Function: get_scheduler
"""
def get_scheduler(optimizer, value=600, _min_lr=1e-5, len_train_dataloader=None, epochs=None):
    
    scheduler = CosineAnnealingLR(
        optimizer, 
        value, 
        eta_min = _min_lr
    )
    """
    total_steps = len_train_dataloader * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)
    """
    
    return scheduler
##==========================================================================================================
"""
Function: flat_accuracy
Description: Function to calculate the accuracy of our predictions vs labels
"""
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)
##==========================================================================================================
"""
Function: format_time
"""
def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))
##==========================================================================================================
"""
Function: train_and_validate
"""
def train_and_validate(model, device, num_epochs, optimizer, scheduler, train_dataloader, validation_dataloader, classes):
    tr_metrics = []
    va_metrics = []
    tmp_print_flag = True

    loss_func = nn.CrossEntropyLoss( label_smoothing = 0.1 )

    # This training code is based on the `run_glue.py` script here:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

    # Set the seed value all over the place to make this reproducible.
    seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # We'll store a number of quantities such as training and validation loss, 
    # validation accuracy, and timings.
    training_stats = []

    # Measure the total training time for the whole run.
    total_t0 = time.time()

    # For each epoch...
    for epoch_i in range(0, num_epochs):
        train_loss = 0.0
        train_preds = []
        train_targets = []
        
        # ========================================
        #               Training
        # ========================================
        
        # Perform one full pass over the training set.

        debugLog("")
        infoLog('======== Epoch {:} / {:} ========'.format(epoch_i + 1, num_epochs))
        infoLog('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0

        io_total_train_acc = 0
        io_total_valid_acc = 0
        
        train_precision_array = list()
        train_recall_array    = list()
        train_f1_array        = list()
        val_precision_array   = list()
        val_recall_array      = list()
        val_f1_array          = list()

        # Put the model into training mode. Don't be mislead--the call to 
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

            # Progress update every 100 batches.
            if step % 100 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                
                # Report progress.
                debugLog('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            # Unpack this training batch from our dataloader. 
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the 
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention masks
            #   [2]: labels 
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because 
            # accumulating the gradients is "convenient while training RNNs". 
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()        

            # Perform a forward pass (evaluate the model on this training batch).
            # In PyTorch, calling `model` will in turn call the model's `forward` 
            # function and pass down the arguments. The `forward` function is 
            # documented here: 
            # https://huggingface.co/transformers/model_doc/bert.html#bertforsequenceclassification
            # The results are returned in a results object, documented here:
            # https://huggingface.co/transformers/main_classes/output.html#transformers.modeling_outputs.SequenceClassifierOutput
            # Specifically, we'll get the loss (because we provided labels) and the
            # "logits"--the model outputs prior to activation.
            result = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask, 
                        labels=b_labels,
                        return_dict=True)
            """
            if tmp_print_flag:
            tmp_print_flag = False
            print(f'result.keys() = {result.keys()}')
            """

            loss = result.loss
            logits = result.logits
            
            
            loss = loss_func(logits, b_labels)

            """
            print(f'loss {loss}')
            print(f'logits {logits}')
            """
            train_preds.extend(logits.argmax(dim=1).cpu().numpy())
            train_targets.extend(batch[2].numpy())

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value 
            # from the tensor.
            total_train_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

            train_acc = accuracy_score(train_targets, train_preds)
            io_total_train_acc += train_acc
            
            train_precision_array.append([precision_score(train_targets, train_preds, average="macro"), precision_score(train_targets, train_preds, average="micro")])
            train_recall_array.append([recall_score(train_targets, train_preds, average="macro"), recall_score(train_targets, train_preds, average="micro")])
            train_f1_array.append([f1_score(train_targets, train_preds, average="macro"), f1_score(train_targets, train_preds, average="micro")])

        io_avg_train_acc = io_total_train_acc / len(train_dataloader)
        train_precision_array = np.asarray(train_precision_array)
        train_recall_array    = np.asarray(train_recall_array)
        train_f1_array        = np.asarray(train_f1_array)
        
        infoLog(
            f'''Epoch {epoch_i+1} : \n\
            Train_acc : {io_avg_train_acc}\n\
            Train_precision (macro, micro): {(np.sum(train_precision_array, axis=0)/len(train_dataloader))}\n\
            Train_recall  (macro, micro): {(np.sum(train_recall_array, axis=0)/len(train_dataloader))}\n\
            Train_F1 : {(np.sum(train_f1_array, axis=0)/len(train_dataloader))}'''
        )

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)            
        
        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        debugLog("")
        infoLog("  Average training loss: {0:.2f}".format(avg_train_loss))
        infoLog("  Training epoch took: {:}".format(training_time))
        
        show_classification_report(train_targets, train_preds, f"Classification report. TRAINING at epoch {epoch_i+1}")
        #show_confusion_matrix(train_targets, train_preds, classes, f"Confusion matrix of training at epoch {epoch_i+1}")
            
        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        debugLog("")
        infoLog("Running Validation...")

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        valid_preds = []
        valid_targets = []

        # Tracking variables 
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            
            # Unpack this training batch from our dataloader. 
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using 
            # the `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids 
            #   [1]: attention masks
            #   [2]: labels 
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            
            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():        

                # Forward pass, calculate logit predictions.
                # token_type_ids is the same as the "segment ids", which 
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                result = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask,
                            labels=b_labels,
                            return_dict=True)

            # Get the loss and "logits" output by the model. The "logits" are the 
            # output values prior to applying an activation function like the 
            # softmax.
            loss = result.loss
            logits = result.logits

            valid_preds.extend(logits.argmax(dim=1).cpu().numpy())
            valid_targets.extend(batch[2].numpy())

            # Accumulate the validation loss.
            total_eval_loss += loss.item()

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to(GLB_DEVICE_CPU).numpy()

            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.
            total_eval_accuracy += flat_accuracy(logits, label_ids)
            
            valid_acc = accuracy_score(valid_targets, valid_preds)
            io_total_valid_acc += valid_acc
            
            val_precision_array.append([precision_score(valid_targets, valid_preds, average="macro"), precision_score(valid_targets, valid_preds, average="micro")])
            val_recall_array.append([recall_score(valid_targets, valid_preds, average="macro"), recall_score(valid_targets, valid_preds, average="micro")])
            val_f1_array.append([f1_score(valid_targets, valid_preds, average="macro"), f1_score(valid_targets, valid_preds, average="micro")])

        io_avg_valid_acc = io_total_valid_acc / len(validation_dataloader)
        infoLog(
                f'''Epoch {epoch_i+1} : \n\
                Valid_acc : {io_avg_valid_acc}\n\
                Valid_precision (macro, micro): {(np.sum(val_precision_array, axis=0)/len(validation_dataloader))}\n\
                Valid_recall (macro, micro): {(np.sum(val_recall_array, axis=0)/len(validation_dataloader))}\n\
                Valid_F1 (macro, micro): {(np.sum(val_f1_array, axis=0)/len(validation_dataloader))}'''
            )

        # Report the final accuracy for this validation run.
        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        infoLog("  Accuracy: {0:.2f}".format(avg_val_accuracy))

        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(validation_dataloader)
        
        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)
        
        infoLog("  Validation Loss: {0:.2f}".format(avg_val_loss))
        infoLog("  Validation took: {:}".format(validation_time))
        
        show_classification_report(valid_targets, valid_preds, f"Classification report. VALIDATION at epoch {epoch_i+1}")

        # Record all statistics from this epoch.
        training_stats.append(
            {
                "epoch": epoch_i + 1,
                "Training Loss": avg_train_loss,
                "Training Accur.": io_avg_train_acc,
                "Training Precision (macro)": (np.sum(train_precision_array, axis=0)/len(train_dataloader))[0], 
                "Training Precision (micro)": (np.sum(train_precision_array, axis=0)/len(train_dataloader))[1], 
                "Training Recall (macro)": (np.sum(train_recall_array, axis=0)/len(train_dataloader))[0],
                "Training Recall (micro)": (np.sum(train_recall_array, axis=0)/len(train_dataloader))[1],
                "Training F1 (macro)": (np.sum(train_f1_array, axis=0)/len(train_dataloader))[0],
                "Training F1 (micro)": (np.sum(train_f1_array, axis=0)/len(train_dataloader))[1],
                "Valid. Loss": avg_val_loss,
                "Valid. Accur.": avg_val_accuracy,
                "Valid. Precision (macro)": (np.sum(val_precision_array, axis=0)/len(validation_dataloader))[0], 
                "Valid. Precision (micro)": (np.sum(val_precision_array, axis=0)/len(validation_dataloader))[1], 
                "Valid. Recall (macro)": (np.sum(val_recall_array, axis=0)/len(validation_dataloader))[0],
                "Valid. Recall (micro)": (np.sum(val_recall_array, axis=0)/len(validation_dataloader))[1],
                "Valid. F1 (macro)": (np.sum(val_f1_array, axis=0)/len(validation_dataloader))[0],
                "Valid. F1 (micro)": (np.sum(val_f1_array, axis=0)/len(validation_dataloader))[1],
                "Training Time": training_time,
                "Validation Time": validation_time
            }
        )

    debugLog("")
    infoLog("Training complete!")

    infoLog("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

    return model, training_stats
##==========================================================================================================
"""
Function: test_model
"""
def test_model(model, device, test_dataloader):
    # ========================================
    #               Test
    # ========================================
    # After the completion of each test epoch, measure our performance on
    # our test set.
    
    test_precision_array   = list()
    test_recall_array      = list()
    test_f1_array          = list()

    debugLog("Running Testing...")

    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    test_preds = []
    test_targets = []

    # Tracking variables 
    total_test_accuracy = 0
    total_test_loss = 0
    nb_test_steps = 0

    io_total_test_acc = 0
    
    testing_stats = {}

    # Evaluate data for one epoch
    for batch in test_dataloader:

        # Unpack this training batch from our dataloader. 
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using 
        # the `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids 
        #   [1]: attention masks
        #   [2]: labels 
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():        

            # Forward pass, calculate logit predictions.
            # token_type_ids is the same as the "segment ids", which 
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            result = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask,
                            labels=b_labels,
                            return_dict=True)

        # Get the loss and "logits" output by the model. The "logits" are the 
        # output values prior to applying an activation function like the 
        # softmax.
        loss = result.loss
        logits = result.logits

        test_preds.extend(logits.argmax(dim=1).cpu().numpy())
        test_targets.extend(batch[2].numpy())

        # Accumulate the test loss.
        total_test_loss += loss.item()

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences, and
        # accumulate it over all batches.
        total_test_accuracy += flat_accuracy(logits, label_ids)

        test_acc = accuracy_score(test_targets, test_preds)
        test_precision_array.append([precision_score(test_targets, test_preds, average="macro"), precision_score(test_targets, test_preds, average="micro")])
        test_recall_array.append([recall_score(test_targets, test_preds, average="macro"), recall_score(test_targets, test_preds, average="micro")])
        test_f1_array.append([f1_score(test_targets, test_preds, average="macro"), f1_score(test_targets, test_preds, average="micro")])

        io_total_test_acc += test_acc

        """
        print(
                f'Test_acc : {test_acc}\n\
                Test_F1 : {test_f1}\n\
                Test_precision : {test_precision}\n\
                Test_recall : {test_recall}\n\n\n'
              )
        """
        
    avg_test_acc = io_total_test_acc / len(test_dataloader)
    infoLog(
        f'''\n\
        Test_acc : {avg_test_acc}\n\
        Test_precision (macro, micro): {(np.sum(test_precision_array, axis=0)/len(test_dataloader))}\n\
        Test_recall (macro, micro): {(np.sum(test_recall_array, axis=0)/len(test_dataloader))}\n\
        Test_F1 (macro, micro): {(np.sum(test_f1_array, axis=0)/len(test_dataloader))}'''
    )
    
    # Report the final accuracy for this test run.
    avg_test_accuracy = total_test_accuracy / len(test_dataloader)
    infoLog("  Accuracy: {0:.2f}".format(avg_test_accuracy))
    
    testing_stats["Test Accur."] = avg_test_acc
    testing_stats["Test Precision (macro)"], testing_stats["Test Precision (micro)"] = (np.sum(test_precision_array, axis=0)/len(test_dataloader))
    testing_stats["Test Recall (macro)"], testing_stats["Test Recall (micro)"] = (np.sum(test_recall_array, axis=0)/len(test_dataloader))
    testing_stats["Test F1 (macro)"], testing_stats["Test F1 (micro)"] = (np.sum(test_f1_array, axis=0)/len(test_dataloader))

    # Calculate the average loss over all of the batches.
    avg_test_loss = total_test_loss / len(test_dataloader)

    # Measure how long the test run took.
    test_time = format_time(time.time() - t0)

    infoLog("  Test Loss: {0:.2f}".format(avg_test_loss))
    infoLog("  Test took: {:}".format(test_time))
               
    show_classification_report(test_targets, test_preds, "Classification report. TEST")
    
    return testing_stats

##==========================================================================================================
"""
Function: save_model
"""
def save_model(model, model_name, path):
    try:
        print("Path", path)
        print("model_name", model_name)
        modelname_path = datetime.datetime.now().strftime("%Y%m%d%H%M%S") + "_" + model_name + "_dict.pt"
        modelname_path = ((modelname_path.replace(" ", "")).replace("/", "-")).replace("\\", "-")
        modelname_path = join(path, modelname_path)
        torch.save(model.state_dict(), modelname_path)
        modelname_path = datetime.datetime.now().strftime("%Y%m%d%H%M%S") + "_" + model_name + "_full.pt"
        modelname_path = ((modelname_path.replace(" ", "")).replace("/", "-")).replace("\\", "-")
        modelname_path = join(path, modelname_path)
        torch.save(model, modelname_path)
    except Exception as err:
        print("ERROR", err)
        torch.save(model.state_dict(), datetime.datetime.now().strftime("%Y%m%d%H%M%S") + "_model_dict.pt")
        torch.save(model, datetime.datetime.now().strftime("%Y%m%d%H%M%S") + "_model_full.pt")
    
##==========================================================================================================
"""
Function: save_json_file_statistics_model
"""
def save_json_file_statistics_model(statistics_model, path_directory, pattern=None):
    filename = ""
    if pattern == None:
        filename = datetime.datetime.now().strftime("%Y%m%d%H%M%S") + "_statistics_model.json"
    else:
        pattern = pattern.replace("/", "-")
        filename = datetime.datetime.now().strftime("%Y%m%d%H%M%S") + "_" + pattern +  "_statistics_model.json"
    
    json_file = open(join(path_directory, filename), GLB_FILE_WRITE_MODE)
    json_file.write(json.dumps(statistics_model, indent=4))
    json_file.close()
    
    return join(path_directory, filename)
##==========================================================================================================
"""
Function: show_classification_report
"""
def show_classification_report(ground_truth, prediction, title):
    infoLog(f'{title} \n {classification_report(ground_truth, prediction)}')
##==========================================================================================================
"""
Function: show_confusion_matrix
"""
def show_confusion_matrix(ground_truth, prediction, _classes, _title):
    plot_confusion_matrix(ground_truth, prediction, classes=_classes, title=_title)
    plt.show()
##==========================================================================================================
"""
Function: plot_confusion_matrix
""" 
def plot_confusion_matrix(y_true, y_pred, classes,
                          title=None,
                          cmap=plt.cm.Blues):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax
##==========================================================================================================
"""
Function: get_df_statistics_set_up
Parameters: 
    @input json_statistics - 
    @input json_setup - 
    @output df_statistics - Dataframe without any None value
""" 
def get_df_statistics_set_up(json_statistics, json_setup, _method='ffill', _index_column_name="index"):  
    df_statistics = None
    for index, (json_obj, set_up) in enumerate(zip(json_statistics, json_setup)):
        df_aux_stats = pd.DataFrame(json_obj)
        df_aux_stats[_index_column_name] = index

        df_aux_setup = pd.DataFrame(set_up)
        df_aux_setup.reset_index(drop=True)

        df_aux_stats = pd.concat([df_aux_stats, df_aux_setup], axis=1)
        df_aux_stats.fillna(method=_method, inplace=True)

        if index == 0:
            df_statistics = df_aux_stats
        else:
            df_statistics = pd.concat([df_statistics, df_aux_stats])
    df_statistics.reset_index(drop=True)
    df_statistics.set_index(_index_column_name, inplace=True)
    return df_statistics.reset_index()

##==========================================================================================================
"""
Function: draw_statistics_spec_epoch
Parameters: 
    @input df - 
    @input columns_of_interest - 
    @input epoch - 
    @input _index - 
    @input size_x - 
    @input size_y - 
    @input _dpi - 
    @input _loc - { "best" | "right" | "center" | 
                    "upper right" | "upper left" | 
                    "lower left" | "lower right" | 
                    "center left" | "center right" | 
                    "lower center" | "upper center" }
    @input withLabelsInPlot - 
    @input _title - 
    @input showPlot - 
    @input showScatter - 
    @output 
""" 
def draw_statistics_spec_epoch(df, columns_of_interest, epoch=None, _index=None, size_x=15, size_y=10, _dpi=80, _loc="best",
                               withLabelsInPlot=False, _title="", showPlot=True, showScatter=True):
  fig = plt.figure(figsize=(size_x, size_y), dpi=_dpi)
  for col in columns_of_interest:
    if epoch != None:
      df_aux = df[df["epoch"] == epoch][col]
    elif _index != None:
      df_aux = df[df["index"] == _index][col]
    else:
      print("ERROR - no epoch or index given")
      return None
    
    if showPlot == False and showScatter == False:
      plt.plot(np.arange(len(df_aux)), df_aux)
      plt.scatter(np.arange(len(df_aux)), df_aux)
    elif showPlot == False:
      plt.scatter(np.arange(len(df_aux)), df_aux)
    elif showScatter == False:
      plt.plot(np.arange(len(df_aux)), df_aux)
    else:
      plt.plot(np.arange(len(df_aux)), df_aux)

    if withLabelsInPlot == True:
      for x, y in zip(np.arange(len(df_aux)), list(df_aux)):
        plt.text(x, y, str(round(y, 4)))
  plt.title(_title)
  plt.legend(columns_of_interest, loc=_loc)

##==========================================================================================================
"""
Function: exec_train
Description: Execution of training process for classification tasks
Parameters: 
"""
def exec_train(df_dataset, column_of_interest, col_of_reference, bert_model_id, bert_model_name,
            batch_size=8,
            epochs=3,
            add_special_tokens=True, 
            max_length_sentence=512, 
            padding_to_max_length=True, 
            return_attention_mask=True, 
            tensor_type='pt', 
            cross_validation=False, 
            run_in_gpu=True, 
            model=None,
            optimizer=None, 
            scheduler=None,
            store_statistics_model=True,
            path_dir_logs="logs/",
            path_project="./",
            path_models="models/",
            store_model=False,
            learning_rate = 2e-5,
            epsilon_optimizer = 1e-8,
            weight_decay = 0
        ):
    test_corpus = None
    
    # Get classes of the dataset
    classes_dataset = get_unique_values_from_dataset(df_dataset, column_of_interest)
    num_classes = len(classes_dataset)

    debugLog(f"Num of different classes in the dataset is {num_classes} which are:")
    for index, elem in enumerate(classes_dataset):
        debugLog(f"\t {index+1} - {elem}")
        
    warnLog(df_dataset["trauma"].value_counts())

    # Define device to be used
    device = get_gpu_device_if_exists()
    debugLog(f"\n\n==> Selected device is '{device}' <==")
    
    #If no parameters are sent, default values are considered. 
    #    IDModel:      Bert
    #    Model namel:  bert-base-uncased
    #    Do uncase:    True

    # Get tokenizer
    tokenizer = get_tokenizer(bert_model_id, bert_model_name) 

    # Get lists of spans and classes
    list_all_spans = list(df_dataset[col_of_reference])
    list_all_classes = [int(elem) if gral_utilities.isfloat(elem) else elem for elem in df_dataset[column_of_interest]]
    
    # Get & print the sentence and length of the largest one
    get_max_length_of_a_sentence_among_all_sentences(tokenizer, list_all_spans, False)

    # If _return_attention_mask, a tuple of two lists is given (tensor_of_inputs, tensor_of_attention_masks)
    all_spans_tokenized = get_all_spans_tokenized(
        tokenizer,
        list_all_spans,
        _add_special_tokens = add_special_tokens, 
        _max_length = max_length_sentence,
        _pad_to_max_length = padding_to_max_length,
        _return_attention_mask = return_attention_mask, 
        type_tensors = tensor_type
    )

    input_ids = None
    attention_masks = None
    
    if return_attention_mask:
        input_ids = convert_list_into_pytorch_tensor(all_spans_tokenized[0])
        attention_masks = convert_list_into_pytorch_tensor(all_spans_tokenized[1])
    else:
        input_ids = convert_list_into_pytorch_tensor(all_spans_tokenized)
    
    numeric_classes = convert_list_span_classes_into_numeric_values(classes_dataset, list_all_classes)
    numeric_classes = convert_list_labels_into_pytorch_tensor(numeric_classes)
    
    ### Split dataset
    if not cross_validation:
        train_labels_corpus, train_input_ids, train_attention_masks, val_labels_corpus, val_input_ids, val_attention_masks, test_labels_corpus, test_input_ids, test_attention_masks = split_dataset_train_val_test(numeric_classes, input_ids, attention_masks)
        test_corpus = [test_labels_corpus, test_input_ids, test_attention_masks]
    else:
        ### k-Fold
        train_val_corpus_cross_validation, test_corpus = split_dataset_train_val_test_k_fold(numeric_classes, input_ids, attention_masks, 0.1)
        
    ### Create model
    if model == None:
        model = create_model(
            bert_model_id,
            bert_model_name,
            num_classes,
            run_in_gpu 
        )
    
    # Define optimizer
    if optimizer == None:
        optimizer = get_optimizer(model, learning_rate=learning_rate, epsilon=epsilon_optimizer, _weight_decay=weight_decay)
    
    # Define scheduler
    if scheduler == None:
        scheduler = get_scheduler(optimizer)
    

    if not cross_validation:
        train_dataset = create_tensor_dataset(train_input_ids, train_attention_masks, train_labels_corpus)
        val_dataset = create_tensor_dataset(val_input_ids, val_attention_masks, val_labels_corpus)
        test_dataset = create_tensor_dataset(test_input_ids, test_attention_masks, test_labels_corpus)
        
        train_dataloader = create_dataloader(train_dataset, batch_size)
        val_dataloader = create_dataloader(val_dataset, batch_size)
        test_dataloader = create_dataloader(test_dataset, batch_size)
        
        """
        if scheduler == None:
            scheduler = get_scheduler(optimizer, len_train_dataloader=len(train_dataloader), epochs=epochs)
        """
        
        model, statistics_model = train_and_validate(model, device, epochs, optimizer, scheduler, train_dataloader, val_dataloader, numeric_classes.tolist())
        
        #save_model(model, "model_wo-cv", join(path_project, path_models))
    
    else:
        list_statistics = list()
        for index_cross_val in range(len(train_val_corpus_cross_validation)):
            train_dataset = create_tensor_dataset(train_val_corpus_cross_validation[index_cross_val][1], train_val_corpus_cross_validation[index_cross_val][2], train_val_corpus_cross_validation[index_cross_val][0])
            val_dataset = create_tensor_dataset(train_val_corpus_cross_validation[index_cross_val][4], train_val_corpus_cross_validation[index_cross_val][5], train_val_corpus_cross_validation[index_cross_val][3])
    
            train_dataloader = create_dataloader(train_dataset, batch_size)
            val_dataloader = create_dataloader(val_dataset, batch_size)
            
            """
            if scheduler == None:
                scheduler = get_scheduler(optimizer, len_train_dataloader=len(train_dataloader), epochs=epochs)
            """
    
            debugLog('='*50)
            debugLog(f"Cross-Validation Split {(index_cross_val+1)}/{len(train_val_corpus_cross_validation)}")
            debugLog('='*50)
            model, statistics_model = train_and_validate(model, device, epochs, optimizer, scheduler, train_dataloader, val_dataloader, numeric_classes.tolist())
            list_statistics.append(statistics_model)
        
        if store_statistics_model:
            save_json_file_statistics_model({"cross-validation": list_statistics}, path_dir_logs, pattern=f"numClasses[{num_classes}]cross-val[{len(train_val_corpus_cross_validation)}]-{bert_model_name}")

        statistics_model = list_statistics
        
    return model, statistics_model, test_corpus

##==========================================================================================================
"""
Function: give_me_segments_of_df_per_class
Description: Execution of training process for classification tasks
Parameters: 
"""
def give_me_segments_of_df_per_class(df, number_of_splits, column_of_interest, column_of_reference):
    dict_of_segments = {}
    invalidSplit = False
    number_of_classes = df[column_of_interest].nunique()
    list_of_classes = df[column_of_interest].unique()
    
    counts = df[column_of_interest].value_counts()
    normalized = round(df[column_of_interest].value_counts(normalize=True), 4)
    percentages = normalized*100
    
    df_stats_dataset = pd.DataFrame({'counts': counts, 'normalized': normalized, 'percentages': percentages}).reset_index()
    
    # Validation
    for i, row in df_stats_dataset.iterrows():
        if row["counts"] < number_of_splits:
            errorLog(f"ERROR - Dataset[{row['index']}] cannot be split into the given number of splits")
            invalidSplit = True
    
    if invalidSplit:
        return None
    else:
        # Get sizes of segments and put them into a list
        list_of_size_segments = (df_stats_dataset["counts"]-(df_stats_dataset["counts"]%number_of_splits)) / number_of_splits
        
        """
        print("*"*100)
        print(df_stats_dataset)
        print("*"*100)
        """
        
        # Initialize dict_of_segments
        for i_range in range(0, number_of_splits):
            dict_of_segments[i_range] = pd.DataFrame()
        
        # Add segments to a list of segments
        for index_class, (size, type_id) in enumerate(zip(list_of_size_segments, df_stats_dataset["index"])):
            size = int(size)
            #print(index_class, "#"*100, size)
            for i_range in range(0, number_of_splits):
                #print(i_range, "*"*50, index_class, type_id, "Segment", i_range, "[", i_range*size, ":", i_range*size+size, "]")
                if index_class == 0:
                    dict_of_segments[i_range] = df[df[column_of_interest] == type_id][i_range*size:i_range*size+size]
                else:
                    if (i_range+1) == number_of_splits:
                        dict_of_segments[i_range] = pd.concat([dict_of_segments[i_range], df[df[column_of_interest] == type_id][i_range*size:]])
                    else:
                        dict_of_segments[i_range] = pd.concat([dict_of_segments[i_range], df[df[column_of_interest] == type_id][i_range*size:i_range*size+size]])
    
    return dict_of_segments

##==========================================================================================================
"""
Function: save_plot
Description: Function that helps to store a plot of matplotlib
Parameters: 
    - plot
    - pattern   - Defines a pattern
"""
def save_plot(plot, pattern="", path="", format=".png"):
    if len(pattern) == 0:
        plot.savefig(join(path, f'{gral_utilities.get_datetime_format()}-plot{format}'))
    else:
        plot.savefig(join(path, f'{gral_utilities.get_datetime_format()}-{pattern}{format}'))

##==========================================================================================================
"""
Function: draw_statistics_of_models
Description: Creates a plot with the models' statistics
Parameters: 
    - df
    - columns_of_interest
    - epoch
    - index
    - size_x
    - size_y
    - _dpi
    - _loc
    - withLabelsInPlot
    - showPlot
    - showScatter
"""
def draw_statistics_of_models(df, columns_of_interest, epoch=None, _index=None, size_x=15, size_y=10, _dpi=80, _loc="best",
                               withLabelsInPlot=False, _title="", showPlot=True, showScatter=True):
  fig = plt.figure(figsize=(size_x, size_y), dpi=_dpi)

  for i_index, model_name in enumerate(df["index"].unique()):
    for i_column, column_name in enumerate(columns_of_interest):
      df_aux = df[df["index"] == model_name][column_name]
    
      if showPlot == False and showScatter == False:
        plt.plot(np.arange(len(df_aux)), df_aux)
        plt.scatter(np.arange(len(df_aux)), df_aux)
      elif showPlot == False:
        plt.scatter(np.arange(len(df_aux)), df_aux)
      elif showScatter == False:
        plt.plot(np.arange(len(df_aux)), df_aux)
      else:
        plt.plot(np.arange(len(df_aux)), df_aux)

      if withLabelsInPlot == True:
        for x, y in zip(np.arange(len(df_aux)), list(df_aux)):
          plt.text(x, y, str(round(y, 4)))
  plt.title(_title)
  plt.legend(df["index"].unique(), loc=_loc)
  plt.xticks(np.arange(len(df_aux)), labels=np.arange(len(df_aux)))

  return plt
  
##==========================================================================================================
"""
Function: get_df_statistics_model
Description: Execute to get the dataframe of the model's statistics
Parameters:
    - json_models_statistics
    - model_names
    - _method
    - _index_column_name
"""
def get_df_statistics_model(json_models_statistics, model_names, _method='ffill', _index_column_name="index"):  
    df_statistics = None
    for index, (json_obj, name) in enumerate(zip(json_models_statistics, model_names)):
        df_aux_stats = pd.DataFrame(json_obj)
        df_aux_stats[_index_column_name] = name
        df_aux_stats.reset_index(drop=True)
        """
        df_aux_setup = pd.DataFrame(set_up)
        df_aux_setup.reset_index(drop=True)

        df_aux_stats = pd.concat([df_aux_stats, df_aux_setup], axis=1)
        df_aux_stats.fillna(method=_method, inplace=True)
        """

        if index == 0:
            df_statistics = df_aux_stats
        else:
            df_statistics = pd.concat([df_statistics, df_aux_stats])
    df_statistics.reset_index(drop=True)
    df_statistics.set_index(_index_column_name, inplace=True)
    return df_statistics.reset_index()
    
##==========================================================================================================
"""
Function: get_id_model
Description: Returns the ID of the corresponding model (specified by name_model)
Parameters: 
    - cfg           - configuration in JSON format
    - name_model    - Name of the model (usually defined by the transformers [HuggingFace models])
Returns:
    Either:
        - ID of the model (s.a. "bert" or "other")
        - Empty string that means it is not defined
"""
def get_id_model(cfg, name_model):
    for id_list_of_models in cfg["models"].keys():
        if name_model in cfg["models"][id_list_of_models]:
            return id_list_of_models
    return ""
    
##==========================================================================================================
"""
Function: get_df_with_specific_size
Description: Retrieve dataset with a specific size once all constraints are fulfilled. Otherwise, return None
        in order to stop execution.
Parameters:
    - df                - Dataframe with dataset
    - mode_selection    - Define mode for selecting elements { "random", "first", "last" }
"""
def get_df_with_specific_size(df, mode, col_of_interest, col_of_reference, desired_size):
    infoLog(f"Parameters at get_df_with_specific_size are [{mode}, {col_of_interest}, {col_of_reference}, {desired_size}]")
    if mode not in GLB_LIST_MODES_4_SELECTING_DATASET:
        infoLog(f'Mode provided by the user [{mode}] is not valid.')
        return None
    
    if not isinstance(desired_size, int):
        infoLog(f'Desired size provided by the user [{desired_size}] is not valid')
        return None
    elif desired_size > len(df):
        infoLog(f'Desired size [{desired_size}] is bigger than the current size of the dataset [{len(df)}]')
        return None
    
    list_classes = get_unique_values_from_dataset(df, col_of_interest)
    num_classes = len(list_classes)
    
    text_warning = ""
    """
    for id_class in list_classes:
        num_elems_per_class_in_dataset = len()
        if num_elems_per_class_in_dataset < GLB_MIN_NUM_SAMPLES_PER_CLASS_4_TRAINING_AND_TESTING:
            text_warning = f'{text_warning}\nClass [{id_class}] does not fulfill the min number of samples for training [{GLB_MIN_NUM_SAMPLES_PER_CLASS_4_TRAINING_AND_TESTING}]'
    """
    
    if len(text_warning) > 0:
        infoLog(text_warning)
        return None
        
    if desired_size < (num_classes*GLB_MIN_NUM_SAMPLES_PER_CLASS_4_TRAINING_AND_TESTING):
        infoLog(f"The length of the dataset is small in comparison with the desired_size in order to split the dataset into |Training|Val|Test")
        return None
        
    list_distribution = list(round(df.groupby(col_of_interest).count()/len(df), 2)[col_of_reference])
    infoLog(f"Distribution of classes {list_classes} is {list_distribution}")
    
    list_new_num_elems_per_class = [int(np.floor(dist_x_class * desired_size)) for dist_x_class in list_distribution]
    infoLog(f"New number of elements per class {list_classes} is {list_new_num_elems_per_class}")
    
    df_final = None
    for index_glb, (num_list, id_class) in enumerate(zip(list_new_num_elems_per_class, list_classes)):
        df_aux = df[df[col_of_interest]==id_class].reset_index(drop=True)
        
        list_classes_src = list(df_aux[col_of_interest])
        index_values = random.sample(list(enumerate(list_classes_src)), num_list)
        list_indices = [index for (index, val) in index_values]
        
        if index_glb == 0:
            df_final = df_aux.loc[df_aux.index[list_indices]]
        else:
            df_final = pd.concat([df_final, df_aux.loc[df_aux.index[list_indices]]], axis=0)

    df_final = df_final.reset_index(drop=True)
    
    infoLog(f"Size of final dataset is {len(df_final)} (may differ from -1 due to factor control)")
    infoLog(f"Number of elements per class\n{df_final[col_of_interest].value_counts()}")
    list_distribution = list(round(df_final.groupby(col_of_interest).count()/len(df_final), 2)[col_of_reference])
    infoLog(f"Final distribution of the new dataset {list_distribution}")
        
    return df_final
    
# DEBUG BEGIN
"""
factor = 20
proportion_1 = 13
proportion_0 = 87
debugLog("df_dataset.shape")
debugLog(df_dataset.shape)
df_dataset_w_trauma = df_dataset[df_dataset["trauma"]==1][0:(proportion_1*factor)]
df_dataset_wo_trauma = df_dataset[df_dataset["trauma"]==0][0:(proportion_0*factor)]
df_dataset = pd.concat([df_dataset_wo_trauma, df_dataset_w_trauma], axis=0)
debugLog(df_dataset["trauma"].value_counts())
debugLog("df_dataset")
debugLog(df_dataset.shape)
"""
# DEBUG END

##==========================================================================================================
"""
Function: get_df_statistics_model_with_active_training
Description: Execute the extraction of a model's information from a JSON object
Parameters:
    - df_statistics of a model
"""
def get_df_statistics_model_with_active_training(json_actLrng, json_lengths, model_name, _method='ffill', _index_column_name="model"):
    df_statistics = None

    LCL_TRAIN_AND_VAL = "training_and_val"
    LCL_TEST = "test"
    LCL_SPLIT_INDEX = "id_split"

    df_train_and_val = None
    df_test = None

    for index, elem_al in enumerate(json_actLrng):
        df_aux_tr_val_stats = pd.DataFrame.from_records(elem_al[LCL_TRAIN_AND_VAL])
        df_aux_tr_val_stats[_index_column_name] = model_name
        df_aux_tr_val_stats[LCL_SPLIT_INDEX] = index+1
        df_aux_tr_val_stats.reset_index(drop=True)
        #print(df_aux_tr_val_stats)

        json_test = elem_al[LCL_TEST]
        df_aux_test_stats = pd.DataFrame.from_records([json_test])
        df_aux_test_stats[_index_column_name] = model_name
        df_aux_test_stats[LCL_SPLIT_INDEX] = index+1
        df_aux_test_stats.reset_index(drop=True)

        if index == 0:
            df_train_and_val = df_aux_tr_val_stats
            df_test = df_aux_test_stats
        else:
            df_train_and_val = pd.concat([df_train_and_val, df_aux_tr_val_stats])
            df_test = pd.concat([df_test, df_aux_test_stats])

    df_train_and_val.set_index([_index_column_name, LCL_SPLIT_INDEX], inplace=True)
    df_test.set_index([_index_column_name, LCL_SPLIT_INDEX], inplace=True)

    df_train_and_val = df_train_and_val.reset_index()
    df_test = df_test.reset_index()

    df_statistics = df_train_and_val.merge(
        df_test,
        on=[_index_column_name, LCL_SPLIT_INDEX],
        how="left"
    )

    df_lengths = pd.DataFrame.from_records(json_lengths)
    df_statistics = df_statistics.merge(
        df_lengths,
        on=LCL_SPLIT_INDEX,
        how="left"
    )

    return df_statistics

##==========================================================================================================
"""
Function: draw_statistics_of_models_ac
Description: Creates a plot with the models' statistics
Parameters: 
    - df
    - columns_of_interest
    - epoch
    - index
    - size_x
    - size_y
    - _dpi
    - _loc
    - withLabelsInPlot
    - showPlot
    - showScatter
    - bestMetricSelection
    - dfHasCrossValidation
    - y_lim_min
    - y_lim_max
    - y_lim_interval
"""
def draw_statistics_of_models_ac(df, 
        column_of_interest, 
        size_x=15, size_y=4, _dpi=80, _loc="best",
        withLabelsInPlot=False, _title="", 
        showPlot=True, showScatter=True,
        bestMetricSelection=False,
        dfHasCrossValidation=False,
        y_lim_min = 0.5,
        y_lim_max = 1.01,
        y_lim_interval = 0.1):

    fig = plt.figure(figsize=(size_x, size_y), dpi=_dpi)
    
    if bestMetricSelection:
        if "Valid" in column_of_interest:
            df = get_smart_selection_df(df, "Validation")
        elif "Test" in column_of_interest:
            df = get_smart_selection_df(df, "Test")
        elif "Training" in column_of_interest:
            df = get_smart_selection_df(df, "Training")
            
    else:
        if dfHasCrossValidation:
            max_epoch = df["epoch"].max()
            max_cv = df["id_cross_val_x"].max()
            df = df[(df["epoch"]==max_epoch) & (df["id_cross_val_x"]==max_cv)]
        else:
            max_epoch = df["epoch"].max()
            df = df[df["epoch"]==max_epoch]

    df_x_axis = None
    #list_colors = ["steelblue", "peru"]
    list_colors = ["black", "black"]
    for i_index, model_name in enumerate(df["model"].unique()):
        df_aux = df[df["model"] == model_name][[column_of_interest, "epoch", "id_split", "length_training", "length_validation", "length_test"]]
            
        if showPlot == False and showScatter == False:
            plt.plot(np.arange(len(df_aux["id_split"])), df_aux[column_of_interest])
            plt.scatter(np.arange(len(df_aux["id_split"])), df_aux[column_of_interest])
        elif showPlot == False:
            plt.scatter(np.arange(len(df_aux["id_split"])), df_aux[column_of_interest])
        elif showScatter == False:
            plt.plot(np.arange(len(df_aux["id_split"])), df_aux[column_of_interest])
        else:
            plt.plot(np.arange(len(df_aux["id_split"])), df_aux[column_of_interest])

        if withLabelsInPlot == True:
            for x, y in zip(np.arange(len(df_aux["id_split"])), list(df_aux[column_of_interest])):
                plt.text(x, y, str(round(y, 4)), color=list_colors[i_index])
    plt.title("\n\n" + _title)
    list_models_name = [ f"M{i}:{name}" for i, name in enumerate(df["model"].unique())]
    plt.legend(list_models_name, loc=_loc)
    plt.ylim(y_lim_min, y_lim_max, y_lim_interval)
    
    df_epochs = None
    if "Valid" in column_of_interest:
        id_column_ref = "length_validation"
    elif "Test" in column_of_interest:
        id_column_ref = "length_test"
    elif "Training" in column_of_interest:
        id_column_ref = "length_training"
    df_epochs = df[[id_column_ref, "model", "epoch"]].pivot(index=id_column_ref, columns=["model"]).reset_index()
    
    for i, model_name in enumerate(df["model"].unique()):
        if i == 0:
            df_epochs["x_axis"] = "Size:" + df_epochs[(id_column_ref, "")].astype(str) + "\nM" + str(i) + " - Ep" + ":" + df_epochs[("epoch", model_name)].astype(str)
        else:
            df_epochs["x_axis"] = df_epochs["x_axis"] + "\nM" + str(i) + " - Ep" + ":" + df_epochs[("epoch", model_name)].astype(str)
            
    plt.xticks(np.arange(len(df_aux[id_column_ref])), labels=df_epochs[("x_axis", "")])
    
    return plt

##==========================================================================================================
"""
Function: draw_statistics_of_models_ac_spec_paper
Description: Creates a plot with the models' statistics. Version specific for the paper release
Parameters: 
    - df
    - columns_of_interest
    - epoch
    - index
    - size_x
    - size_y
    - _dpi
    - _loc
    - withLabelsInPlot
    - showPlot
    - showScatter
    - bestMetricSelection
    - dfHasCrossValidation
    - y_lim_min
    - y_lim_max
    - y_lim_interval
"""
def draw_statistics_of_models_ac_spec_paper(df, 
        column_of_interest, 
        size_x=15, size_y=4, _dpi=80, _loc="best",
        withLabelsInPlot=False, _title="", 
        showPlot=True, showScatter=True,
        bestMetricSelection=False,
        dfHasCrossValidation=False,
        y_lim_min = 0.5,
        y_lim_max = 1.01,
        y_lim_interval = 0.1):

    fig = plt.figure(figsize=(size_x, size_y), dpi=_dpi)
    
    if bestMetricSelection:
        if "Valid" in column_of_interest:
            df = get_smart_selection_df(df, "Validation")
        elif "Test" in column_of_interest:
            df = get_smart_selection_df(df, "Test")
        elif "Training" in column_of_interest:
            df = get_smart_selection_df(df, "Training")
            
    else:
        if dfHasCrossValidation:
            max_epoch = df["epoch"].max()
            max_cv = df["id_cross_val_x"].max()
            df = df[(df["epoch"]==max_epoch) & (df["id_cross_val_x"]==max_cv)]
        else:
            max_epoch = df["epoch"].max()
            df = df[df["epoch"]==max_epoch]

    df_x_axis = None
    #list_colors = ["steelblue", "peru"]
    list_colors = ["black", "black"]
    for i_index, model_name in enumerate(df["model"].unique()):
        df_aux = df[df["model"] == model_name][[column_of_interest, "epoch", "id_split", "length_training", "length_validation", "length_test"]]
            
        if showPlot == False and showScatter == False:
            plt.plot(np.arange(len(df_aux["id_split"])), df_aux[column_of_interest])
            plt.scatter(np.arange(len(df_aux["id_split"])), df_aux[column_of_interest])
        elif showPlot == False:
            plt.scatter(np.arange(len(df_aux["id_split"])), df_aux[column_of_interest])
        elif showScatter == False:
            plt.plot(np.arange(len(df_aux["id_split"])), df_aux[column_of_interest])
        else:
            plt.plot(np.arange(len(df_aux["id_split"])), df_aux[column_of_interest])

        if withLabelsInPlot == True:
            y_str = None
            #for x, y in zip(np.arange(len(df_aux["id_split"])), list(df_aux[column_of_interest])):
            for index_labels, (x, y) in enumerate(zip(np.arange(len(df_aux["id_split"])), list(df_aux[column_of_interest]))):
                if "F1" in column_of_interest:
                    if i_index == 0: # BertBase
                        if index_labels in [2, 6, 7]:
                            y_str = str(round(y, 4)) + "\n"
                    elif i_index == 1: # HateBert
                        if index_labels in [0, 5]:
                            y_str = str(round(y, 4)) + "\n"
                elif "Recall" in column_of_interest:
                    if i_index == 0: # BertBase
                        if index_labels in [2, 3, 4, 7]:
                            y_str = str(round(y, 4)) + "\n"
                    elif i_index == 1: # HateBert
                        if index_labels in [6]:
                            y_str = str(round(y, 4)) + "\n"
                if y_str == None: 
                    y_str = str(round(y, 4))
                plt.text(x, y, y_str, color=list_colors[i_index])
                y_str = None
    plt.title("\n\n" + _title)
    list_models_name = [ f"M{i}:{name}" for i, name in enumerate(df["model"].unique())]
    plt.legend(list_models_name, loc=_loc)
    plt.ylim(y_lim_min, y_lim_max, y_lim_interval)
    
    df_epochs = None
    if "Valid" in column_of_interest:
        id_column_ref = "length_validation"
    elif "Test" in column_of_interest:
        id_column_ref = "length_test"
    elif "Training" in column_of_interest:
        id_column_ref = "length_training"
    df_epochs = df[[id_column_ref, "model", "epoch"]].pivot(index=id_column_ref, columns=["model"]).reset_index()
    
    for i, model_name in enumerate(df["model"].unique()):
        if i == 0:
            df_epochs["x_axis"] = "Size:" + df_epochs[(id_column_ref, "")].astype(str) + "\nM" + str(i) + " - Ep" + ":" + df_epochs[("epoch", model_name)].astype(str)
        else:
            df_epochs["x_axis"] = df_epochs["x_axis"] + "\nM" + str(i) + " - Ep" + ":" + df_epochs[("epoch", model_name)].astype(str)
            
    plt.xticks(np.arange(len(df_aux[id_column_ref])), labels=df_epochs[("x_axis", "")])
    
    return plt

##==========================================================================================================
"""
Function: get_df_statistics_model_with_active_training_and_cross_validation
Description: Execute the extraction of a model's information from a JSON object
Parameters:
    - df_statistics of a model
"""
def get_df_statistics_model_with_active_training_and_cross_validation(json_actLrng, json_lengths, model_name, _method='ffill', _index_column_name="model"):
    df_statistics = None

    LCL_TRAIN_AND_VAL = "training_and_val"
    LCL_TEST = "test"
    LCL_SPLIT_INDEX = "id_split"
    LCL_CV_INDEX = "id_cross_val"

    df_train_and_val = None
    df_test = None

    for index_al, elem_al in enumerate(json_actLrng):
        for index_cv, elem_cv in enumerate(elem_al[LCL_TRAIN_AND_VAL]):
            df_aux_tr_val_stats = pd.DataFrame.from_records(elem_cv)
            df_aux_tr_val_stats[_index_column_name] = model_name
            df_aux_tr_val_stats[LCL_SPLIT_INDEX] = index_al+1
            df_aux_tr_val_stats[LCL_CV_INDEX] = index_cv+1
            df_aux_tr_val_stats.reset_index(drop=True)

            if index_al == 0 and index_cv == 0:
                df_train_and_val = df_aux_tr_val_stats
            else:
                df_train_and_val = pd.concat([df_train_and_val, df_aux_tr_val_stats])

        json_test = elem_al[LCL_TEST]
        df_aux_test_stats = pd.DataFrame.from_records([json_test])
        df_aux_test_stats[_index_column_name] = model_name
        df_aux_test_stats[LCL_SPLIT_INDEX] = index_al+1
        df_aux_test_stats[LCL_CV_INDEX] = index_cv+1
        df_aux_test_stats.reset_index(drop=True)

        if index_al == 0:
            df_test = df_aux_test_stats
        else:
            df_test = pd.concat([df_test, df_aux_test_stats])

    
    df_train_and_val.set_index([_index_column_name, LCL_SPLIT_INDEX, LCL_CV_INDEX], inplace=True)
    df_test.set_index([_index_column_name, LCL_SPLIT_INDEX, LCL_CV_INDEX], inplace=True)

    df_train_and_val = df_train_and_val.reset_index()
    df_test = df_test.reset_index()

    df_statistics = df_train_and_val.merge(
        df_test,
        on=[_index_column_name, LCL_SPLIT_INDEX],
        how="left"
    )

    df_lengths = pd.DataFrame.from_records(json_lengths)
    df_statistics = df_statistics.merge(
        df_lengths,
        on=LCL_SPLIT_INDEX,
        how="left"
    )

    return df_statistics
    
##==========================================================================================================
"""
Function: get_smart_selection_df
Description: Retrieve information of statistics from models that were executed without CrossValidation
Parameters:
    - df        - Dataframe with the statistics of the model
    - factor    - Define whether the desired elements belong to { "Training", "Validation", "Test" }
"""
def get_smart_selection_df(df, factor):
    training_columns = [ "Training Precision (macro)", "Training Precision (micro)", "Training Recall (macro)", "Training Recall (micro)", "Training F1 (macro)", "Training F1 (micro)" ]
    validation_columns = [ 'Valid. Precision (macro)', 'Valid. Precision (micro)', 'Valid. Recall (macro)', 'Valid. Recall (micro)', 'Valid. F1 (macro)', 'Valid. F1 (micro)' ]
    test_columns = [ 'Test Precision (macro)', 'Test Precision (micro)', 'Test Recall (macro)', 'Test Recall (micro)', 'Test F1 (macro)', 'Test F1 (micro)' ]
    
    columns_priority_ranking = None
    columns_of_interest = None
    if factor == "Training":
      columns_priority_ranking = [ 'Training F1 (macro)', "Training Recall (macro)" ]
      columns_of_interest = training_columns
    elif factor == "Validation":
      columns_priority_ranking = [ 'Valid. F1 (macro)', "Valid. Recall (macro)" ]
      columns_of_interest = validation_columns
    elif factor == "Test":
      columns_priority_ranking = [ 'Valid. F1 (macro)', "Valid. Recall (macro)" ]#[ 'Test F1 (macro)', "Test Recall (macro)" ]
      columns_of_interest = test_columns
    else:
      print("ERROR - Not valid option")
      return None
    
    #df_selection = df[["model", "id_split", "epoch"] + columns_of_interest + ["length_training", "length_validation", "length_test"]]
    if factor != "Test":
        df_selection = df[["model", "id_split", "epoch"] + columns_of_interest + ["length_training", "length_validation", "length_test"]]
    else:
        df_selection = df[["model", "id_split", "epoch"] + validation_columns + test_columns + ["length_training", "length_validation", "length_test"]]
    
    df_smart_selection = None
    for index_model, model_name in enumerate(df_selection["model"].unique()):
      for i, index_split in enumerate(df_selection[df_selection["model"]==model_name]["id_split"].unique()):
        df_row = df_selection[(df_selection["model"] == model_name) & (df_selection["id_split"] == index_split)].sort_values(columns_priority_ranking, axis=0, ascending=False)[0:1]
        if i==0 and index_model==0:
          df_smart_selection = df_row
        else:
          df_smart_selection = pd.concat([df_smart_selection, df_row])
    return df_smart_selection
    