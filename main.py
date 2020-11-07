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
processed_features_tweets = vectorizer.fit_transform(processed_features_tweets).toarray()


#   2.First train on pre-processed data
#Train / Test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(processed_features_tweets, labels_sentiments, test_size=0.2, random_state=0)
#Training - RandomForestClassifier
text_classifier_RFC = RandomForestClassifier(n_estimators=200, random_state=0)
text_classifier_RFC.fit(X_train, y_train)
#Testing - RandomForestClassifier
predictions_RFC = text_classifier_RFC.predict(X_test)
print(confusion_matrix(y_test,predictions_RFC))
print(classification_report(y_test,predictions_RFC))
print(accuracy_score(y_test, predictions_RFC))


#   3.Train Doc2Vec on your non pre processed data

#   4.Apply Doc2Vec on each document in your data

#   5.Compute the sentiment

#   6.Use a classifier (naïve baise, random forest, …) to classify your documents by sentiment

#   7.Graph the classified documents, color coded by sentiment