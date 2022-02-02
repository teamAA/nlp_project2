import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from nltk import ngrams
from nltk.tokenize import word_tokenize
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')

def tokenization(text):
    # Word Tokenization
    text = re.split('\W+', text)
    return text

def remove_stopwords(text):
    stopword = nltk.corpus.stopwords.words('english')
    text = [word for word in text if word not in stopword]
    return text

def lemmatizing(text):
    # Word Lemmatization 
    lm = nltk.WordNetLemmatizer()
    text = [lm.lemmatize(word, nltk.corpus.wordnet.VERB) for word in text]
    return text

def stemming(text):
    # Word Stemming 
    ps = nltk.PorterStemmer()
    text = [ps.stem(word) for word in text]
    return text

# Remove Whitespace
def empty_token(text):
    text = [x for x in text if len(x)>0]
    return text

def argmax_2(lst):
  lst = lst.tolist()
  return (lst.index(max(lst)))-1

def data_preproc_v1(df):

    df['text'] = df['text'].astype(str)

    df['clean_text'] = df['text'].apply(lambda x : x.lower())
    # Regex to get letter only 
    df['clean_text'] = [re.sub(r"(@[A-Za-z]+)|([^A-Za-z \t])|(\w+:\/\/\S+)", " ", df['clean_text'][x]) for x in range(0,df.shape[0])]

    df['cleansed_text'] = df['clean_text']\
        .apply(lambda x: tokenization(x.lower()))\
        .apply(lambda x: remove_stopwords(x))\
        .apply(lambda x: lemmatizing(x))\
        .apply(lambda x: stemming(x))\
        .apply(lambda x: empty_token(x)) 

    df['cleansed_text2']=df['cleansed_text'].apply(lambda x :  ' '.join([str(item) for item in x]))   

    return df

def data_preproc(df):

    df['text'] = df['text'].astype(str)
    # Regex to get letter only 
    df['text'] = [re.sub(r"(@[A-Za-z]+)|([^A-Za-z \t])|(\w+:\/\/\S+)", " ", df['text'][x]) for x in range(0,df.shape[0])]

    df['cleansed_text'] = df['text']\
        .apply(lambda x: tokenization(x.lower()))\
        .apply(lambda x: remove_stopwords(x))\
        .apply(lambda x: lemmatizing(x))\
        .apply(lambda x: stemming(x))\
        .apply(lambda x: empty_token(x))    


    train_cleaned = df[df['dataset']=='train']
    pred_cleaned = df[df['dataset']=='pred'].reset_index().drop('index',axis=1)

    return train_cleaned, pred_cleaned

def bag_of_words(train, pred):
    # The default token pattern removes tokens of a single character. That's why we don't have the "I" and "s" tokens in the output
    sentences_train = [' '.join(x) for x in train['cleansed_text']]
    sentences_pred = [' '.join(x) for x in pred['cleansed_text']]

    count_vectorizer = CountVectorizer()
    count_vectorizer.fit(sentences_train)
    bag_of_words_train = count_vectorizer.transform(sentences_train)
    bag_of_words_pred = count_vectorizer.transform(sentences_pred)

    # Show the Bag-of-Words Model as a pandas DataFrame
    feature_names = count_vectorizer.get_feature_names()
    df_bow_train = pd.DataFrame(bag_of_words_train.toarray(), columns = feature_names)
    df_bow_pred = pd.DataFrame(bag_of_words_pred.toarray(), columns = feature_names)

    return df_bow_train, df_bow_pred

def get_bow_columns(df_bow):
    col = []
    for i in list(df_bow.columns):
        if len(set(i)) >= 3:
            col.append(i)
    return col

def predict_data(pred, bow_pred, col, loaded_model):
    # predict data
    bow_pred['sentiment_result']=loaded_model.predict(bow_pred[col])

    # merge with original data
    pred_result=pd.concat([pred,bow_pred[['sentiment_result']]],1)

    # convert result
    pred_result['sentiment_result']=np.where(pred_result['sentiment_result']==0, 'neutral',
        np.where(pred_result['sentiment_result']==1, 'positive', 'negative'))
    final_pred_result=pred_result[['text', 'sentiment_result']]

    return final_pred_result

def pre_bigram(data, parameter) : 
    df_data_pos = " ".join(data['cleansed_text2'])
    token_text_pos = word_tokenize(df_data_pos)
    bigrams_pos = ngrams(token_text_pos, parameter)
    frequency_pos = Counter(bigrams_pos)
    df_pos = pd.DataFrame(frequency_pos.most_common(10))
    df_pos['word']=df_pos[0].apply(lambda x : ' '.join(x))
    df_pos['total tweet']=df_pos[1]
    df_pos=df_pos.sort_values(by='total tweet', ascending=True)
    return df_pos
