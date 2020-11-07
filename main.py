import multiprocessing

import numpy as np
import pandas as pd
import seaborn as sns
import re
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import utils
import csv
from tqdm import tqdm
import multiprocessing
import nltk
from nltk.corpus import stopwords


#Set options for pandas to display everything
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
#Load the csv file containing the tweets
data = pd.read_csv("C:/Users/lilia/github/Tweet_classification_sentiment/Tweets.csv", encoding = "ISO-8859-1", engine='python')
print(data.head())
#    0. Data Analysis
#Change default plot options
plot_size = plt.rcParams["figure.figsize"]
#print(plot_size[0])
#print(plot_size[1])

plot_size[0] = 8
plot_size[1] = 6
plt.rcParams["figure.figsize"] = plot_size

#Plot various columns of data in a pie chart (plot only one at a time)
#data.airline.value_counts().plot(kind='pie', autopct='%1.0f%%')
#data.airline_sentiment.value_counts().plot(kind='pie', autopct='%1.0f%%', colors=["red", "yellow", "green"])
#data.negativereason.value_counts().plot(kind='pie', autopct='%1.0f%%')
#plt.show()

#Plot the sentiment for each individual airline
#airline_sentiment = data.groupby(['airline', 'airline_sentiment']).airline_sentiment.count().unstack()
#airline_sentiment.plot(kind='bar')
#plt.show()

#Plothe confidence level for each sentiment
#sns.barplot(x='airline_sentiment', y='airline_sentiment_confidence' , data=data)
#plt.show()


#   1.Preprocess - Data Cleaning
#Data extraction
features_tweets = data.iloc[:, 10].values #extract the 10th column (text)
labels_sentiments = data.iloc[:, 1].values #extract the 1st column (airline_sentiment)

#Cleaning
processed_features_tweets = []
for sentence in range(0, len(features_tweets)):
    # Remove special characters
    processed_tweet = re.sub(r'\W', ' ', str(features_tweets[sentence]))
    # Remove single characters
    processed_tweet= re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_tweet)
    # Remove single characters from the start
    processed_tweet = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_tweet)
    # Substituting multiple spaces with single space
    processed_tweet = re.sub(r'\s+', ' ', processed_tweet, flags=re.I)
    # Removing prefixed 'b'
    processed_tweet = re.sub(r'^b\s+', '', processed_tweet)
    # Converting to Lowercase
    processed_tweet = processed_tweet.lower()
    #Append tweet to the new list of cleaned tweets
    processed_features_tweets.append(processed_tweet)

#Converting to numerical data (TF-IDF)
vectorizer = TfidfVectorizer (max_features=2500, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
processed_features_tweets_num = vectorizer.fit_transform(processed_features_tweets).toarray()


#   2.First train on pre-processed data

#Train / Test split (80/20)
X_train_RFC, X_test_RFC, y_train_RFC, y_test_RFC = train_test_split(processed_features_tweets_num, labels_sentiments, test_size=0.2, random_state=0)
#Training - RandomForestClassifier
text_classifier_RFC = RandomForestClassifier(n_estimators=200, random_state=0)
text_classifier_RFC.fit(X_train_RFC, y_train_RFC)
#Testing - RandomForestClassifier
predictions_RFC = text_classifier_RFC.predict(X_test_RFC)
print("\nRandom Forest Classifier\n")
print(confusion_matrix(y_test_RFC,predictions_RFC))
print(classification_report(y_test_RFC,predictions_RFC))
print(accuracy_score(y_test_RFC, predictions_RFC))


#   3.Doc2Vec on your non pre processed data
def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            tokens.append(word.lower())
    return tokens
#14640 rows, 80/20 split so 11712 -> train and 2928 -> test
tweets_tagged_train=[]
tweets_tagged_test=[]
tags_index = {'negative': 0 , 'neutral': 1, 'positive': 2}
for i in range(len(features_tweets)):
    if i <11713:
        tweets_tagged_train.append(TaggedDocument(words=tokenize_text(features_tweets[i]),tags=[tags_index.get(labels_sentiments[i])]))

    else:
        tweets_tagged_test.append(TaggedDocument(words=tokenize_text(features_tweets[i]),tags=[tags_index.get(labels_sentiments[i])]))

cores = multiprocessing.cpu_count()
model_dbow = Doc2Vec(dm=1, vector_size=300, negative=5, hs=0, min_count=2, sample = 0, workers=cores, alpha=0.025, min_alpha=0.001)
model_dbow.build_vocab([x for x in tqdm(tweets_tagged_train)])
tweets_tagged_train  = utils.shuffle(tweets_tagged_train)
model_dbow.train(tweets_tagged_train,total_examples=len(tweets_tagged_train), epochs=30)
model_dbow.save('./Model.d2v')

def vector_for_learning(model, input_docs):
    sents = input_docs
    targets, feature_vectors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, feature_vectors
y_train, X_train = vector_for_learning(model_dbow, tweets_tagged_train)
y_test, X_test = vector_for_learning(model_dbow, tweets_tagged_test)

#Doc2Vec Logistic reg
logreg = LogisticRegression(n_jobs=1, C=1e5, max_iter=100)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print("\nLogistic Regression Doc2Vec\n")
print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1 score : {}'.format(f1_score(y_test, y_pred, average='weighted')))

#Doc2Vec RandomForestClassifier
text_classifier_RFC = RandomForestClassifier(n_estimators=200, random_state=0)
text_classifier_RFC.fit(X_train, y_train)
predictions_RFC = text_classifier_RFC.predict(X_test)
print("\nRandom Forest Classifier Doc2Vec\n")
print(confusion_matrix(y_test,predictions_RFC))
print(classification_report(y_test,predictions_RFC))
print(accuracy_score(y_test, predictions_RFC))


#   4.Graph the classified documents, color coded by sentiment