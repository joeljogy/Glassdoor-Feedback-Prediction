# -*- coding: UTF-8 -*-
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.classify.util import accuracy
from nltk.classify import NaiveBayesClassifier

##
##Sentiment Analysis for prediction
reviews = pd.read_csv('./newreviewdata.csv')

##Combine the columns pros and cons into a single column named review
reviews["review"] = reviews["pros"] + ". " + reviews["cons"]
reviews.to_csv('./combinedreview_data.csv',index=False)



def format_sentence(sent):
    return({word: True for word in nltk.word_tokenize(sent)})

pos = []
neg = []
for index, row in reviews.iterrows():
    try: 
        if row['category'] == 'positive':
            try: 
                pos.append([format_sentence(row['review']), 'pos'])
            except UnicodeDecodeError:
                pass
        else:
            try:
                neg.append([format_sentence(row['review']), 'neg'])
            except UnicodeDecodeError:
                pass
    except TypeError:
        pass



# next, split labeled data into the training and test data
training = pos[:int((.8)*len(pos))] + neg[:int((.8)*len(neg))]
test = pos[int((.8)*len(pos)):] + neg[int((.8)*len(neg)):]

classifier = NaiveBayesClassifier.train(training)

print "The accuracy for prediction is: ",accuracy(classifier, test) 
n = raw_input("Enter review(pros and cons written together for prediction: ")

print classifier.classify(format_sentence(n))
b = raw_input("Enter Y to close: ")


