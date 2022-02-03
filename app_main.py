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
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')

st.set_page_config(layout = "wide")

# Import data
train=pd.read_csv('src/data-registry/train.csv')
train['dataset'] = 'train'

# Import pickle model
with open(r"src/model_v1.pkl", "rb") as input_file:
    loaded_model = pickle.load(input_file)

# Import data for prediction
pred_train = train[['text','sentiment', 'dataset']]

# Preprocess
data=utils_for_dashboard.data_preproc_v1(train)
start_date=pd.to_datetime('2021-08-01')
end_date=pd.to_datetime('2021-08-08')
data["tweet_date"] = np.random.choice(pd.date_range(start_date, end_date), len(data))
list_sentiment=['all']+list(data['sentiment'].unique())

# Create title for all pages
st.title('Sentiment Analysis V3')
st.markdown('The project is to build a model that will determine the tone (neutral, positive, negative) of the tweet text.')

menu = st.sidebar.radio( "What's your go?", ('Explore', 'Predict'))

if menu == 'Explore': 
    sentiment = st.sidebar.selectbox("Sentiment ", list_sentiment)
    today = data['tweet_date'].min()
    tomorrow = data['tweet_date'].max()
    start_date = st.sidebar.date_input('Start date', today)
    start_date2=pd.to_datetime(start_date)
    end_date = st.sidebar.date_input('End date', tomorrow)
    end_date2=pd.to_datetime(end_date)
    data=data[(data['tweet_date']>=start_date2)&(data['tweet_date']<=end_date2)]

    if sentiment == 'all' :
      # st.write('------------------------------------------------------------')
      total=data.shape[0]
      st.write('Total tweets : ',total)
      st.write('------------------------------------------------------------')
      st.header("Sentiment Trend")

      # Trend
      df_trend=data.groupby(['tweet_date', 'sentiment'])['textID'].count().reset_index()
      fig = px.bar(df_trend, x="tweet_date", y="textID", color="sentiment", title="Sentiment Trend", width=800, height=400)
      fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), yaxis_title=None)
      st.plotly_chart(fig)

      st.write('------------------------------------------------------------')
      st.header("Top Words")

      col1, col2 = st.columns(2)

      df_pos=utils_for_dashboard.pre_bigram(data, 1)
      figure1 = px.bar(df_pos, x='total tweet', y='word', orientation='h')
      figure1.update_layout(title_text='Top 10 Bigrams',
                            # yaxis={'categoryorder':'total ascending'}, 
                            width=800, 
                            height=400, 
                            yaxis_title=None,
                            yaxis = dict(tickfont = dict(size=16))
                            )

      col1.plotly_chart(figure1,use_container_width = True)

      # bigram
      df_pos=utils_for_dashboard.pre_bigram(data, 2)
      figure2 = px.bar(df_pos, x='total tweet', y='word', orientation='h')
      figure2.update_layout(title_text='Top 10 Bigrams',
                            # yaxis={'categoryorder':'total ascending'}, 
                            width=800, 
                            height=400, 
                            yaxis_title=None,
                            yaxis = dict(tickfont = dict(size=16))
                            )

      col2.plotly_chart(figure2,use_container_width = True)

      st.write('------------------------------------------------------------')
      st.header("Detail Text")
      st.table(data[['tweet_date', 'textID', 'text', 'sentiment','cleansed_text']].head(10))

    else : 
      data2=data[data['sentiment']==sentiment]
      # st.write('------------------------------------------------------------')
      total=data2.shape[0]
      st.write('Total tweets : ',total)
      st.write('------------------------------------------------------------')
      st.header("Sentiment Trend")
      
      df_trend=data2.groupby(['tweet_date'])['textID'].count().reset_index().rename(columns={'tweet_date':'Date'})
      fig = px.line(df_trend, x="Date", y="textID")
      fig.update_layout(title_text='Top 10 Bigrams', yaxis_title=None)
      st.plotly_chart(fig)

      st.write('------------------------------------------------------------')
      st.header("Top Words")

      col1, col2 = st.columns(2)

      df_pos=utils_for_dashboard.pre_bigram(data2, 1)
      figure1 = px.bar(df_pos, x='total tweet', y='word', orientation='h')
      figure1.update_layout(title_text='Top 10 Bigrams',
                            # yaxis={'categoryorder':'total ascending'}, 
                            width=800, 
                            height=400, 
                            yaxis_title=None,
                            yaxis = dict(tickfont = dict(size=16))
                            )

      col1.plotly_chart(figure1,use_container_width = True)

      # bigram
      df_pos=utils_for_dashboard.pre_bigram(data2, 2)
      figure2 = px.bar(df_pos, x='total tweet', y='word', orientation='h')
      figure2.update_layout(title_text='Top 10 Bigrams',
                            # yaxis={'categoryorder':'total ascending'}, 
                            width=800, 
                            height=400, 
                            yaxis_title=None,
                            yaxis = dict(tickfont = dict(size=16))
                            )

      col2.plotly_chart(figure2,use_container_width = True)

      st.write('------------------------------------------------------------')
      st.header("Detail Text")
      st.table(data2[['tweet_date', 'textID', 'text', 'sentiment','cleansed_text']].head(10)) 
else : 
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
            data_contains=data[data['clean_text'].str.contains(message, na = False)].sort_values(by='tweet_date', ascending=False)[['text', 'sentiment']]
            data_contains.columns=['text', 'sentiment_result']
            st.write('Total tweets containing your word : ', data_contains.shape[0])
            st.table(data_contains.head(5))

            # create visualization
            chart=data_contains.groupby('sentiment_result')['text'].count().reset_index()
            fig = px.bar(chart, x="sentiment_result", y="text", title="Wide-Form Input")
            fig.update_layout(title="Sentiment Distribution", xaxis_title="Sentiment Results", yaxis_title="Total Data")
            st.plotly_chart(fig)
    # ------
    st.write('Option 2 : **Upload your csv file**')
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        pred = pd.read_csv(uploaded_file)
        pred['sentiment']=np.nan
        pred['dataset'] = 'pred'

        df = pd.concat([pred_train, pred],axis=0)
        df = df.reset_index().drop('index',axis=1)

        train, pred = utils_for_dashboard.data_preproc(df)
        bow_train, bow_pred = utils_for_dashboard.bag_of_words(train, pred)
        col = utils_for_dashboard.get_bow_columns(bow_train)
        # loaded_model = pickle.load(open('model_v1.pkl', 'rb'))
        final_pred_result=utils_for_dashboard.predict_data(pred, bow_pred, col, loaded_model)
        st.write('Sentiment results')
        st.write('Total tweets containing your word : ', final_pred_result.shape[0])
        st.table(final_pred_result.head(5))

        # create visualization
        chart=final_pred_result.groupby('sentiment_result')['text'].count().reset_index()
        fig = px.bar(chart, x="sentiment_result", y="text", title="Wide-Form Input")
        fig.update_layout(title="Sentiment Distribution", xaxis_title="Sentiment Results", yaxis_title="Total Data")
        st.plotly_chart(fig)

        # download data
        def convert_df(df):
            return df.to_csv(index=False, sep='|').encode('utf-8')
        
        csv = convert_df(final_pred_result)
        
        st.download_button( "Press to Download", csv, "file.csv", "text/csv", key='download-csv')