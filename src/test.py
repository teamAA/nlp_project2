import sys
import os
import dvc.api
import pickle
import train
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

import sys
import re
import utils
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def load_data():
    train = pd.read_csv('data-registry/train.csv')
    test = pd.read_csv('data-registry/test.csv')

    print(f"train shape: {train.shape}")
    train['dataset'] = 'train'
    test['dataset'] = 'test'

    df = pd.concat([train,test],axis=0)
    df = df.reset_index().drop('index',axis=1)

    return df

def data_preproc_custom(df):
    df['sentiment'] = np.where(df['sentiment']=='neutral',0,df['sentiment'])
    df['sentiment'] = np.where(df['sentiment']=='positive',1,df['sentiment'])
    df['sentiment'] = np.where(df['sentiment']=='negative',-1,df['sentiment'])

    df['text'] = df['text'].astype(str)
    # Regex to get letter only 
    df['text'] = [re.sub(r"(@[A-Za-z]+)|([^A-Za-z \t])|(\w+:\/\/\S+)", " ", df['text'][x]) for x in range(0,df.shape[0])]

    df['cleansed_text'] = df['text']\
        .apply(lambda x: utils.tokenization(x.lower()))\
        .apply(lambda x: utils.remove_stopwords(x))\
        .apply(lambda x: utils.lemmatizing(x))\
        .apply(lambda x: utils.stemming(x))\
        .apply(lambda x: utils.empty_token(x))    


    train_cleaned = df[df['dataset']=='train']
    test_cleaned = df[df['dataset']=='test'].reset_index().drop('index',axis=1)
    expected_test_cleaned = df[df['dataset']=='testing_text'].reset_index().drop('index',axis=1)

    return train_cleaned, test_cleaned, expected_test_cleaned

def bag_of_words_custom(train, test, expected_test):
    # The default token pattern removes tokens of a single character. That's why we don't have the "I" and "s" tokens in the output
    sentences_train = [' '.join(x) for x in train['cleansed_text']]
    sentences_test = [' '.join(x) for x in test['cleansed_text']]
    senteces_expected_test = [' '.join(x) for x in expected_test['cleansed_text']]

    count_vectorizer = CountVectorizer()
    count_vectorizer.fit(sentences_train)

    bag_of_words_train = count_vectorizer.transform(sentences_train)
    bag_of_words_test = count_vectorizer.transform(sentences_test)
    bag_of_words_expected_test = count_vectorizer.transform(senteces_expected_test)

    # Show the Bag-of-Words Model as a pandas DataFrame
    feature_names = count_vectorizer.get_feature_names()
    df_bow_train = pd.DataFrame(bag_of_words_train.toarray(), columns = feature_names)
    df_bow_test = pd.DataFrame(bag_of_words_test.toarray(), columns = feature_names)
    df_bow_expected_test = pd.DataFrame(bag_of_words_expected_test.toarray(), columns = feature_names)

    return df_bow_train, df_bow_test, df_bow_expected_test

def get_bow_columns(df_bow):
    col = []
    for i in list(df_bow.columns):
        if len(set(i)) >= 3:
            col.append(i)
    return col

def testing():

    #expected output
    testing_text = ['he is good','she is beautiful','they are very good','he is stupid','you are bad']
    expected_output = [1,1,1,-1,-1]

    # create new data to be tested
    testing_text = {'text' : testing_text,'sentiment':expected_output}
    testing_text = pd.DataFrame(testing_text)
    testing_text['dataset'] = 'testing_text'

    df = load_data()
    # merging with testing_text
    df = pd.concat([df,testing_text],axis=0)
    df = df.reset_index().drop('index',axis=1)
    
    train, test, expected_test = data_preproc_custom(df)
    bow_train, bow_test, bow_expected_test = bag_of_words_custom(train, test,expected_test)

    #get column name
    col = get_bow_columns(bow_train)
    
    #get model
    model = train.main(log = False)

    logregpred = model.predict_proba(bow_test[col])
    pred_logreg = []
    for i in range(0,len(logregpred)):
        pred_logreg.append(utils.argmax_2(logregpred[i]))

    # Testing for accuracy performance
    acc_score = round(accuracy_score(test['sentiment'].tolist(),pred_logreg), 5)
    print("accuracy: ", acc_score)
    if acc_score <= 0.4:
        raise ValueError("Accuracy is too low")
    elif acc_score >= 0.95:
        raise ValueError("Accuracy is too high")
    else:
        pass
    
    # testing expected output
    logregpred2 = model.predict_proba(bow_expected_test[col])
    pred_logreg2 = []
    for i in range(0,len(logregpred2)):
        pred_logreg2.append(int(utils.argmax_2(logregpred2[i])))
    
    for i in range(len(pred_logreg2)):
        if expected_output[i] != pred_logreg[i]:
            raise ValueError(f"{testing_text[i]} should be label {expected_output[i]} not {pred_logreg[i]}")
        else :
            pass

    pickle.dump(model, open("model_v1.pkl", 'wb'))
    print("Passed the test!")


if __name__ == "__main__":
    testing()
    

