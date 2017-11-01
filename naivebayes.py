import numpy as np

with open("data\\sentiment labelled sentences\\imdb_labelled.txt",
          "r") as text_file:
    lines = text_file.read().split("\n")

with open("data\\sentiment labelled sentences\\yelp_labelled.txt",
          "r") as text_file:
    lines += text_file.read().split("\n")

with open("data\\sentiment labelled sentences\\amazon_cells_labelled.txt",
          "r") as text_file:
    lines += text_file.read().split("\n")

lines = [
    line.split("\t") for line in lines
    if len(line.split("\t")) == 2 and line.split("\t")[1] != ""
]

train_documents = [line[0] for line in lines]
train_labels = [int(line[1]) for line in lines]

from sklearn.feature_extraction.text import CountVectorizer

count_vectorizer = CountVectorizer(binary=True)
train_documents = count_vectorizer.fit_transform(train_documents)

from sklearn.naive_bayes import BernoulliNB
classifier = BernoulliNB().fit(train_documents, train_labels)


def classify(message):

    result = "Classified: " + message  + " AS "

    out = classifier.predict(count_vectorizer.transform([message]))
    if (out[0] == 1):
        result += "Positive"
    else:
        result += "Negative"

    print(result)

classify("This is the best movie")
classify("This is the worst movie")
classify("this movie was less than satisfactory")
