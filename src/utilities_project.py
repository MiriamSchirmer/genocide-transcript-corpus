import pandas as pd
from enum import Enum
import string
import re

import nltk
from nltk.corpus import words
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download()

ROOT_DRIVE_PATH = '/content/drive'
list_stop_words_english = stopwords.words("english")
stemmer = WordNetLemmatizer()
tokenizer = RegexpTokenizer(r'\w+')

#################################################################################
### Connect to drive account

#= Connection is established on the root path of Drive
def drive_connect():
    from google.colab import drive
    drive.mount(ROOT_DRIVE_PATH)

#= Connection is established on the specified path
def drive_connect_to_path(path):
    from google.colab import drive
    if path == "":
        drive_connect()
    else:
        drive.mount(path)

#################################################################################
### Read CSV file using Pandas
def pandas_read_csv(path_file, _delimiter):
    information = pd.Series()
    try:
        if _delimiter == None:
            information = pd.read_csv(path_file)
        else:
            information = pd.read_csv(path_file, delimiter=_delimiter)
    except Exception as exc:
        print(f"\nERROR An error occurs while reading CSV file with pandas")

    return information

#################################################################################
### Enumeration with the pre-processing steps
class PreprocessingText(Enum):
    LOWER_CASE = 1
    PUNCT_REGEX = 2
    PUNCTUATION = 3
    NUMBERS = 4
    STOP_WORDS = 5
    STEMMING = 6
    LEMMATIZATION = 7
    REMOVE_URL = 8
    
    
#################################################################################
### Preprocessing text (sentences)
#= @input a string with the initial information
#= @returns a string 

def pre_processing_sentence(sentence_tokens, list_preprocessing_steps):
    if (type(list_preprocessing_steps) != 'list'):
        list_preprocessing_steps = [PreprocessingText.LOWER_CASE, PreprocessingText.PUNCT_REGEX, PreprocessingText.PUNCTUATION, PreprocessingText.NUMBERS, PreprocessingText.STOP_WORDS, PreprocessingText.STEMMING, PreprocessingText.LEMMATIZATION, PreprocessingText.REMOVE_URL]
    
    
    # LowerCase
    if PreprocessingText.LOWER_CASE in list_preprocessing_steps:
        new_sentence = sentence_tokens.lower()

    # Remove URLs
    if PreprocessingText.REMOVE_URL in list_preprocessing_steps:
        new_sentence = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', new_sentence)
    
    # Tokenization with RegEx
    if PreprocessingText.PUNCT_REGEX in list_preprocessing_steps:
        new_sentence = ' '.join(tokenizer.tokenize(new_sentence))
    
    # Remove punctuation
    if PreprocessingText.PUNCTUATION in list_preprocessing_steps:
        new_sentence = ''.join([character for character in new_sentence if character not in string.punctuation])
    
    # Remove numbers
    if PreprocessingText.NUMBERS in list_preprocessing_steps:
        new_sentence = ''.join([ character for character in new_sentence if not character.isdigit() ])
    
    # Remove stopwords
    if PreprocessingText.STOP_WORDS in list_preprocessing_steps:
        new_sentence = [ token for token in new_sentence.split() if token not in list_stop_words_english ]
        new_sentence = ' '.join(new_sentence)
    
    # Remove lemmas
    if PreprocessingText.LEMMATIZATION in list_preprocessing_steps:
        new_sentence = [ stemmer.lemmatize(token) for token in new_sentence.split() ]
        new_sentence = ' '.join(new_sentence)
    
    return new_sentence