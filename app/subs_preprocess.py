import pandas as pd
import re
import numpy as np
import pysrt
import spacy
import en_core_web_sm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import pickle

def load():
    with open('sub_pipe.pcl', "rb") as f:
        return pickle.load(f)
pipe = load()

nlp = spacy.load("en_core_web_sm")

def sub_processing(upload_file):
    try:
        subs = pysrt.from_string(upload_file.getvalue().decode('cp1252'))
        print('Decode ANSI success')
        if subs.text == '':
            subs = pysrt.from_string(upload_file.getvalue().decode('utf-16'))
            print('Decode UTF-16 success')
        print('Read file success')
    except Exception as ex:
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message)
        return
    
    # Очистка текста
    text = re.sub('<i>|</i>', '', subs.text)
    text = re.sub('\<.*?\>', '', text)      # удаляем то что в скобках <>
    text = re.sub('\n', ' ', text)               # удаляем разделители строк
    # text = re.sub('<font.*?font>', '', text)
    text = re.sub('\(.*?\)', '', text)           # удаляем то что в скобках ()
    text = re.sub('\[.*?\]', '', text)           # удаляем то что в скобках []
    text = re.sub('[A-Z]+?:', '', text)          # удаляем слова написанные заглавными буквами с двоеточием(это имена тех кто говорит)
    text = re.sub('\.+?:', '\.', text)           # Заменяем троеточия на одну точку
    text = text.lower()
    text = re.sub('[^a-z\.\!\?]', ' ', text)     # удаляем всё что не буквы и не .?!
    text = re.sub(' +', ' ', text)               # удаляем " +"
    spacy_results = nlp(text)
    text = ' '.join([token.lemma_ for token in spacy_results])
    text = [text]
    text = pipe.transform(text).toarray()
    return text