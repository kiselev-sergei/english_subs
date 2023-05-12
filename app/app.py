import streamlit as st
import catboost
from catboost import CatBoostClassifier, Pool
import pandas as pd
import nltk
import time
import pickle
from subs_preprocess import sub_processing

model = CatBoostClassifier()
features = ['Subtitles']
def load():
    with open('./sub_model.pcl', 'rb') as mod:
        return pickle.load(mod)
model = load()

TITLE = 'English level prediction with film subtitles'
st.set_page_config(
                   page_title=TITLE,
                   page_icon='🎬',
                   initial_sidebar_state='expanded',
                 )
st.title(TITLE)
st.write('This ML-based app predicts the linguistic level of a film for English learners. The classification is based on CEFR levels (A1, A2, B1, B2, C1, C2). Upload susbtitles in .srt format to know the level.')

upload_file = st.file_uploader('Upload subtitles in .srt format', type='srt')

def make_predict(data, model):
    """
    :param data:
    :param model:
    :return:
    """
    predict_pool = Pool(data=data,
                       )
    predict = model.predict(predict_pool)
    decode = {1:'A1',
              2:'A2',
              3:'B1',
              4:'B2',
              5:'C1',
              6:'C2'
             }
    return predict

if upload_file:

    print(upload_file.name)

    df = sub_processing(upload_file)
    if df is None:
        st.write('Problem with the subs file. Try another one')
    else:
        st.header(f'This film is labeled **:[{make_predict(df, model)}]** on CEFR classification')