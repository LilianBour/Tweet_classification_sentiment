import multiprocessing

from gensim.models import Doc2Vec
from sklearn import datasets

categories = ["soc.religion.christian", "sci.space", "talk.politics.mideast", "rec.sport.baseball"]
cat_dict = {} # Contains raw training data organized by category
cat_dict_test = {} # Contains raw test data organized by category
for cat in categories:
    cat_dict[cat] = datasets.fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), categories=[cat]).data
    cat_dict_test[cat] = datasets.fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), categories=[cat]).data

import gensim


def tokenize(text, stopwords, max_len=20):
    return [token for token in gensim.utils.simple_preprocess(text, max_len=max_len) if token not in stopwords]


cat_dict_tagged_train = {}  # Contains clean tagged training data organized by category. To be used for the training corpus.
cat_dict_test_clean = {}  # Contains clean un-tagged training data organized by category.

offset = 0  #Used for managing IDs of tagged documents
for k, v in cat_dict.items():
    cat_dict_tagged_train[k] = [gensim.models.doc2vec.TaggedDocument(tokenize(text, [], max_len=200), [i + offset]) for i, text in enumerate(v)]
    offset += len(v)
offset = 0 #Used for managing IDs of tagged documents
for k, v in cat_dict_test.items():
    cat_dict_test_clean[k] = [tokenize(text, [], max_len=200) for i, text in enumerate(v)]
    offset += len(v)

# Eventually contains final versions of the training data to actually train the model
train_corpus = [taggeddoc for taggeddoc_list in list(cat_dict_tagged_train.values()) for taggeddoc in taggeddoc_list]

cores = multiprocessing.cpu_count()
model_dbow = Doc2Vec(dm=1, vector_size=300, negative=5, hs=0, min_count=2, sample = 0, workers=cores, alpha=0.025, min_alpha=0.001)
model_dbow.build_vocab(train_corpus)
model_dbow.train(train_corpus, total_examples=model_dbow.corpus_count, epochs=model_dbow.epochs)

metadata = {}
inferred_vectors_test = {} # Contains, category-wise, inferred doc vecs for each document in the test set
for cat, docs in cat_dict_test_clean.items():
    inferred_vectors_test[cat] = [model_dbow.infer_vector(doc) for doc in list(docs)]
    metadata[cat] = len(inferred_vectors_test[cat])

import csv

def write_to_csv(input, output_file, delimiter='\t'):
    with open(output_file, "w") as f:
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerows(input)


veclist_metadata = []
veclist = []
for cat in cat_dict.keys():
    for tag in [cat] * metadata[cat]:
        veclist_metadata.append([tag])
    for vec in inferred_vectors_test[cat]:
        veclist.append(list(vec))
write_to_csv(veclist, "doc2vec_TweetsSentiments_vectors.csv")
write_to_csv(veclist_metadata, "doc2vec_TweetsSentiments_vectors_metadata.csv")