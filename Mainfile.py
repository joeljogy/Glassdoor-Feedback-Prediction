# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.classify.util import accuracy
from nltk.classify import NaiveBayesClassifier
import os



class ReviewResearch():
        """docstring for GeoDataParser"""
        def __init__(self, company_name= None):
                self.company_name = company_name

        def read_data(self, filename):
            ##Read the CSV file
            reviews = pd.read_csv(filename)
            
            # Viewing the data
            #print reviews.shape
            #print reviews.head()
            #print reviews.info()
            #print reviews.describe()

            return reviews


        def CFD(self,reviews, lang, outfile):
            ##Do Conditional Frequency Distribution to find most frequently used words

            stopWords = set(stopwords.words(lang))
            otherstopwords = ['.',')','(',',','-','I','A',"n't",'get','work','honeywell',"''",'``',"'s","%","much",'!',';','...']
            for t in otherstopwords:
                stopWords.add(t)

            #Finding 50 frequest positive/pros context words
            pros_list=list(reviews.pros)
            allpros_words = []
            for line in pros_list:
                pros_words = nltk.word_tokenize(str(line).lower())
                for i in pros_words:
                    allpros_words.append(i)
            pros_wordsFiltered = []
            for w in allpros_words:
                if w not in stopWords:
                    pros_wordsFiltered.append(w)


            frequencies_words = nltk.FreqDist(pros_wordsFiltered).most_common()
            poswords_most_frequent = [word[0] for word in frequencies_words]
            #print poswords_most_frequent[0:50]

            #Finding 50 frequest negative/cons context words
            cons_list=list(reviews.cons)
            allcons_words = []
            for line in cons_list:
                cons_words = nltk.word_tokenize(str(line).lower())
                for i in cons_words:
                    allcons_words.append(i)
            cons_wordsFiltered = []
            for w in allcons_words:
                if w not in stopWords:
                    cons_wordsFiltered.append(w)


            frequencies_words = nltk.FreqDist(cons_wordsFiltered).most_common()
            negwords_most_frequent = [word[0] for word in frequencies_words]
            #print negwords_most_frequent[0:50]


            #Taking the most frequently used 50 words in pros column and cons column to build positive and negative vocab resp
            positive_vocab = poswords_most_frequent[0:50]
            negative_vocab = negwords_most_frequent[0:50]

            extra_stopwords = ['good','great','less','upper','like','many','lot','every','one']
            for t in extra_stopwords:
                stopWords.add(t)

                
            pros_wordsFiltered = []
            for w in allpros_words:
                if w not in stopWords:
                    pros_wordsFiltered.append(w)


            frequencies_words = nltk.FreqDist(pros_wordsFiltered).most_common()
            poswords_most_frequent_new = [word[0] for word in frequencies_words]

            cons_wordsFiltered = []
            for w in allcons_words:
                if w not in stopWords:
                    cons_wordsFiltered.append(w)


            frequencies_words = nltk.FreqDist(cons_wordsFiltered).most_common()
            negwords_most_frequent_new = [word[0] for word in frequencies_words]



            positive_vocab_new = poswords_most_frequent_new[0:50]
            negative_vocab_new = negwords_most_frequent_new[0:50]

            pos_neg_vocab = []
            pos_neg_vocab.append(positive_vocab_new)
            pos_neg_vocab.append(negative_vocab_new)
            
            

            #Adding the column 'category' to classify if the comment is more positive or more negative
            for index, row in reviews.iterrows():
                pos_score = 0
                neg_score = 0
                pros_words = nltk.word_tokenize(str(row['pros']).lower())
                for i in pros_words:
                    if i in positive_vocab:
                        pos_score +=1
                cons_words = nltk.word_tokenize(str(row['cons']).lower())
                for j in cons_words:
                    if j in negative_vocab:
                        neg_score +=1
                if pos_score>neg_score:
                    reviews.loc[index, 'category'] = 'positive'
                elif neg_score>pos_score:
                    reviews.loc[index, 'category'] = 'negative'
                else:
                    reviews.loc[index, 'category'] = 'neutral'

            #Saves the the new csv file with category appended
            reviews.to_csv(outfile,index=False)

            return(pos_neg_vocab)

        def study_trends(self,newdata):
            ##Plot for quarterly analysis of positive vs negative feedback

            #Data to plot
            N = 10
            positive = tuple()
            negative = tuple()
            neutral = tuple()
            quarterly = pd.read_csv(newdata)
            pos = 0
            neg = 0
            neu = 0
            quarter=1
            for index, row in quarterly.iterrows():
                if row['quarter'] == quarter:
                    x = row['category']
                    if x == 'positive':
                        pos+=1
                    elif x == 'negative':
                        neg+=1
                    else:
                        neu+=1
                else:
                    quarter+=1
                    positive += (pos,)
                    negative += (neg,)
                    neutral += (neu,)
                    pos = 0
                    neg = 0
                    neu = 0
                    if row['quarter'] == quarter:
                        x = row['category']
                        if x == 'positive':
                            pos+=1
                        elif x == 'negative':
                            neg+=1
                        else:
                            neu+=1
            positive += (pos,)
            negative += (neg,)
            neutral += (neu,)

            print "Count of feedbacks based on 10 quarters:"
            print
            print "Positive count = ",positive
            print "Negative count = ",negative
            print "Neutral count = ",neutral


            #Create plot
            fig, ax = plt.subplots()
            index = np.arange(N)
            bar_width = 0.20
            opacity = 0.8
             
            rects1 = plt.bar(index, positive, bar_width,
                             alpha=opacity,
                             color='b',
                             label='Positive')
             
            rects2 = plt.bar(index + bar_width, negative, bar_width,
                             alpha=opacity,
                             color='r',
                             label='Negative')

            rects3 = plt.bar(index + bar_width + bar_width, neutral, bar_width,
                             alpha=opacity,
                             color='g',
                             label='Neutral')

            plt.xlabel('Quarter')
            plt.ylabel('Feedback/Review count')
            plt.title('Trend of comments in every quarter')
            plt.xticks(index + bar_width, ('1st', '2nd', '3rd', '4th', '5th','6th','7th','8th','9th','10th'))
            plt.legend()
             
            plt.tight_layout()
            plt.show()


        def sentiment_analysis(self,filename):
            os.startfile(filename)


        def titles_relation(self,newdata):
                titles = pd.read_csv(newdata)
                title_names = []
                stopWords = set(stopwords.words('english'))
                otherstopwords = ['.',')','(',',','-','I','A',"n't",'get','work',"''",'``',"'s","%","much",'!',';','...',"good","great"]
                for t in otherstopwords:
                    stopWords.add(t)


                for index, row in titles.iterrows():
                    if row['category'] == 'negative':
                        title_words = nltk.word_tokenize(str(row['title']).lower())
                        for i in title_words:
                            title_names.append(i)

                title_wordsFiltered = []
                for w in title_names:
                    if w not in stopWords:
                        title_wordsFiltered.append(w)


                frequencies_words = nltk.FreqDist(title_wordsFiltered).most_common()
                negwords_most_frequent = [word[0] for word in frequencies_words]
                print
                print "Top 50 words from title that match to negative comments "
                print
                print negwords_most_frequent[0:50]
                print
                j = raw_input("Enter Y to move for prediction: ")

        def top_grievances(self,pos_neg_data):
                print
                print "Top reasons for grievances are ",pos_neg_data[1]
                print

        def top_positivity(self,pos_neg_data):
                print
                print "Top reasons for positivity are ",pos_neg_data[0]
                print


if __name__ == "__main__":
        
        # class instance
        review_research = ReviewResearch()

        # reading and viewing the data
        review_data = review_research.read_data(filename='GlassdoorData.csv')

        # do conditional frequency distribution to find top 50 words 
        pos_neg_vocab_data = review_research.CFD(review_data,lang='english',outfile='./newreviewdata.csv')

        # plot positive vs negative vs neutral comments quarterly basis
        review_research.study_trends(newdata="./newreviewdata.csv")

        # display top reasons for grievances and positivity
        review_research.top_grievances(pos_neg_vocab_data)
        review_research.top_positivity(pos_neg_vocab_data)

        # titles with max complaints
        review_research.titles_relation(newdata="./newreviewdata.csv")

        # sentiment analysis for prediction
        review_research.sentiment_analysis(filename='prediction.py')


        wait=raw_input("Enter Y to quit")




