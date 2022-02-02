import utils_for_dashboard
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.subplots as sp
import plotly.figure_factory as ff
import plotly.graph_objects as go
from io import BytesIO, StringIO
import requests

st.set_page_config(layout = "wide")

# Import data
train=pd.read_csv('src/data-registry/train.csv')
train['dataset'] = 'train'

# Import pickle model
with open(r"src/model_v1.pkl", "rb") as input_file:
    loaded_model = pickle.load(input_file)

# Import data for prediction
pred_train = train[['text','sentiment', 'dataset']]
data=utils_for_dashboard.data_preproc_v1(train)

# Create title for all pages
st.title('Sentiment Analysis')
st.markdown('The project is to build a model that will determine the tone (neutral, positive, negative) of the tweet text.')

st.header('Predict Data')
st.markdown('Predict data using machine learning models that have been built with high accuracy')
# ---
st.write('Option 1 : **Input your text**')
with st.form(key='Enter name'):
    message = st.text_input('Enter the word or sentences for which you want to know the sentiment')
    submit_button = st.form_submit_button(label='Submit')
    if submit_button:
        # create new data to predict
        list_df_pred = {'text' : [message]}
        pred = pd.DataFrame(list_df_pred)
        pred['sentiment']=np.nan
        pred['dataset'] = 'pred'
        
        # creating a Dataframe object 
        df_pred = pd.DataFrame(list_df_pred)
        df = pd.concat([pred_train, pred],axis=0)
        df = df.reset_index().drop('index',axis=1)

        # predict data
        train, pred = utils_for_dashboard.data_preproc(df)
        bow_train, bow_pred = utils_for_dashboard.bag_of_words(train, pred)
        col = utils_for_dashboard.get_bow_columns(bow_train)
        # loaded_model = pickle.load(open('model_v1.pkl', 'rb'))
        final_pred_result=utils_for_dashboard.predict_data(pred, bow_pred, col, loaded_model)

        # show results
        st.write('Sentiment results : ', final_pred_result['sentiment_result'].values[0])
        st.markdown('Last 10 Tweet Data')
        data_contains=data[data['clean_text'].str.contains(message, na = False)][['text', 'sentiment']]
        data_contains.columns=['text', 'sentiment_result']
        st.write('Total tweets containing your word : ', data_contains.shape[0])
        st.table(data_contains.head(5))