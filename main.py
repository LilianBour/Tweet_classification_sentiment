import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import csv
import multiprocessing
import nltk
from nltk.corpus import stopwords

#Function to tokenize the text = separate in words
def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            tokens.append(word.lower())
    return tokens

#Function to build the vector feature for the Classifier
def vector_for_learning(model, input_docs):
    sents = input_docs
    targets, feature_vectors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, feature_vectors

#Function to write data to csv
def write_to_csv(input, output_file, delimiter='\t'):
    with open(output_file, "w") as f:
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerows(input)

#Set options for pandas to display everything
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

#Load the csv file containing the tweets
data = pd.read_csv("C:/Users/lilia/github/Tweet_classification_sentiment/Tweets.csv", encoding = "ISO-8859-1", engine='python')
print(data.head())
#Data extraction
features_tweets = data.iloc[:, 10].values #extract the 10th column (text)
labels_sentiments = data.iloc[:, 1].values #extract the 1st column (airline_sentiment)



#    1. Data Analysis
#Change default plot options
plot_size = plt.rcParams["figure.figsize"]
plot_size[0] = 8
plot_size[1] = 6
plt.rcParams["figure.figsize"] = plot_size

# Plot various columns of data in a pie chart (plot only one at a time)
#data.airline.value_counts().plot(kind='pie', autopct='%1.0f%%')
#data.airline_sentiment.value_counts().plot(kind='pie', autopct='%1.0f%%', colors=["red", "yellow", "green"])
#data.negativereason.value_counts().plot(kind='pie', autopct='%1.0f%%')
#plt.show()

# Plot the sentiment for each individual airline
#airline_sentiment = data.groupby(['airline', 'airline_sentiment']).airline_sentiment.count().unstack()
#airline_sentiment.plot(kind='bar')
#plt.show()

# Plot the confidence level for each sentiment
#sns.barplot(x='airline_sentiment', y='airline_sentiment_confidence' , data=data)
#plt.show()



#   2.Classic random forest classifier
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

#Train / Test split (80/20)
X_train_RFC, X_test_RFC, y_train_RFC, y_test_RFC = train_test_split(processed_features_tweets_num, labels_sentiments, test_size=0.2, random_state=0)

#Training - RandomForestClassifier
text_classifier_RFC = RandomForestClassifier(n_estimators=10, random_state=0)
text_classifier_RFC.fit(X_train_RFC, y_train_RFC)

#Testing and Evaluating - RandomForestClassifier
predictions_RFC = text_classifier_RFC.predict(X_test_RFC)
print("\nRandom Forest Classifier\n")
print(confusion_matrix(y_test_RFC,predictions_RFC))
print(classification_report(y_test_RFC,predictions_RFC))
print('Accuracy %s' % accuracy_score(y_test_RFC, predictions_RFC))
print('F1 score : {}'.format(f1_score(y_test_RFC, predictions_RFC, average='weighted')))



#   3.Doc2Vec on your non pre processed data
sentiments_tags = {'negative': 'negative', 'neutral': 'neutral', 'positive': 'positive'}
cat_train_raw = {} # Contains raw training data organized by category
cat_test_raw = {} # Contains raw test data organized by category
#Creating test/train (14640 rows, 80/20 split so 11712 -> train and 2928 -> test)
for i in range(len(features_tweets)):
    if i <11713:
        cat_train_raw.setdefault(sentiments_tags.get(labels_sentiments[i]), []).append(features_tweets[i])
    else:
        cat_test_raw.setdefault(sentiments_tags.get(labels_sentiments[i]), []).append(features_tweets[i])
print('\nTokenization\n')
cat_train_tag = {}  # Contains clean tagged training data organized by category. To be used for the training corpus.
cat_test_clean = {}  # Contains clean un-tagged training data organized by category.
offset = 0  #To manage IDs of tagged documents
for key, val in cat_train_raw.items():
    cat_train_tag[key] = [TaggedDocument(tokenize_text(text), [i + offset]) for i, text in enumerate(val)]
    offset += len(val)
offset = 0
for key, val in cat_test_raw.items():
    cat_test_clean[key] = [tokenize_text(text) for i, text in enumerate(val)]
    offset += len(val)
#Training data to actually train the model
train_corpus = [taggeddoc for taggeddoc_list in list(cat_train_tag.values()) for taggeddoc in taggeddoc_list]

#Creating model Doc2Vec
print('\nCreating model\n')
cores = multiprocessing.cpu_count() #to use multiples cores
model_dbow = Doc2Vec(dm=1, vector_size=50, workers=cores, alpha=0.01, min_alpha=0.001, epochs=600)
model_dbow.build_vocab(train_corpus)
model_dbow.train(train_corpus, total_examples=model_dbow.corpus_count, epochs=model_dbow.epochs)
model_dbow.save('./Model.d2v')

#Create features for model evaluation
print('\nCreating features\n')
y_train, X_train = vector_for_learning(model_dbow, train_corpus)
y_test, X_test = vector_for_learning(model_dbow, train_corpus)

print('\nModel evaluation\n')
#Doc2Vec Logistic reg
"""
logreg = LogisticRegression(n_jobs=1, C=1e5, max_iter=30)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print("\nLogistic Regression Doc2Vec\n")
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1 score : {}'.format(f1_score(y_test, y_pred, average='weighted')))

#Doc2Vec RandomForestClassifier
text_classifier_Doc2Vec = RandomForestClassifier(n_estimators=50, random_state=0)
text_classifier_Doc2Vec.fit(X_train, y_train)
predictions_RFC_doc2vec = text_classifier_Doc2Vec.predict(X_test)
print("\nRandom Forest Classifier Doc2Vec\n")
print(confusion_matrix(y_test,predictions_RFC_doc2vec))
print(classification_report(y_test,predictions_RFC_doc2vec))
print('Accuracy %s' % accuracy_score(y_test, predictions_RFC_doc2vec))
print('F1 score : {}'.format(f1_score(y_test, predictions_RFC_doc2vec, average='weighted')))
"""


#   4.Graph the classified documents, color coded by sentiment - Vizualisation
print('\nMetadata\n')
metadata = {}
inferred_vectors_test = {} # Contains, category-wise, inferred doc vecs for each document in the test set
for cat, docs in cat_test_clean.items():
    inferred_vectors_test[cat] = [model_dbow.infer_vector(doc) for doc in list(docs)]
    metadata[cat] = len(inferred_vectors_test[cat])

print('\nWriting vectors\n')
veclist_metadata = []
veclist = []
for cat in cat_train_raw.keys():
    for tag in [cat] * metadata[cat]:
        veclist_metadata.append([tag])
    for vec in inferred_vectors_test[cat]:
        veclist.append(list(vec))
write_to_csv(veclist, "doc2vec_TweetsSentiments_vectors.csv")
write_to_csv(veclist_metadata, "doc2vec_TweetsSentiments_vectors_metadata.csv")
print('\nDone\n')
#Go to website and load both csv files to vizualize data
#http://projector.tensorflow.org
