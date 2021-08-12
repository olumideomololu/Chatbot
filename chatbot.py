#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 21:06:31 2020

@author: psxoo4
"""

import csv
import nltk
from numpy import dot
from numpy.linalg import norm
from nltk.corpus import stopwords
from nltk.stem.snowball import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
nltk.download('stopwords')

# Creation of Term document matrix for Question and Answer Database
with open('COMP3074-CW1-Dataset.csv', mode='r',encoding='utf-8') as file:
    reader = csv.reader(file)
    reader.__next__()
    queryBase = {rows[1].lower(): rows[2] for rows in reader}

queryList = queryBase.keys()
queryAnswers = list(queryBase.values())

# Stemming function
p_stemmer = PorterStemmer()
analyzer = CountVectorizer().build_analyzer()

def stemmed_words(doc):
    return (p_stemmer.stem(w) for w in analyzer(doc))

# Term-Document Modelling
count_vect = CountVectorizer(stop_words=stopwords.words('english'), analyzer=stemmed_words)
queryCounts = count_vect.fit_transform(queryList)

tfidf_transformer = TfidfTransformer(use_idf=True, sublinear_tf=True).fit(queryCounts)
QueryTf = tfidf_transformer.transform(queryCounts).toarray()

# Query Processing function
def QueryProc(query):
    tmpList = [query.lower()]
    tmpCount = count_vect.transform(tmpList)
    tmptf = tfidf_transformer.transform(tmpCount).toarray()

    MaxSim = 0
    index = 0
    i = 0
    while i < len(queryList): # Identity question within database with maximum similiarity to supplied query
        cosSim = dot(QueryTf[i], tmptf[0])/(norm(QueryTf[i])*norm(tmptf[0]))
        if cosSim > MaxSim:
            MaxSim = cosSim
            index = i
        i += 1
    
    if MaxSim < 0.7:
        return "Sorry, I am not in possession of that information at the moment" # return error message if no answer with sufficient similarity has been identified 
    else:
        return queryAnswers[index] # return answer to question with highest similarity to supplied query


# creation of Term-Document Matrix for intent matching and small talk
with open('smalltalk.csv', mode='r',encoding='utf-8') as file:
    reader = csv.reader(file)
    reader.__next__()
    talkBase = {rows[0].lower(): rows[1] for rows in reader}

talkList = talkBase.keys()
talkAnswers = list(talkBase.values())

# Term-Document Modelling
talk_vect = CountVectorizer(analyzer=stemmed_words)
talkCounts = talk_vect.fit_transform(talkList)

talk_transformer = TfidfTransformer(use_idf=True, sublinear_tf=True).fit(talkCounts)
talkTf = talk_transformer.transform(talkCounts).toarray()

# Intent and Small talk Processing function
def talkProc(query):
    tmpList = [query.lower()]
    tmpCount = talk_vect.transform(tmpList)
    tmptf = talk_transformer.transform(tmpCount).toarray()

    MaxSim = 0
    index = 0
    i = 0
    while i < len(talkList): # Identity question within database with maximum similiarity to supplied query
        cosSim = dot(talkTf[i], tmptf[0])/(norm(talkTf[i])*norm(tmptf[0]))
        if cosSim > MaxSim:
            MaxSim = cosSim
            index = i
        i += 1
    
    if MaxSim < 0.6: # similairty Threshold 
        return QueryProc(query) # Search Question and answer databse for reponse to query
    else:
        return talkAnswers[index] # return appropriate response to intent or small talk

# User Interface 
print("Hi there, I am an interactive question answering chatbot.")
name = input("What's you name, please\n")
stop = False
print(f"Hello {name}, what you like to do? We can have a nice chat but I can also do other things, like changing your name or answering a question")
while not stop:
    print()
    intent = talkProc(input(f"What would you like to do now, {name}?\n"))
    if intent == "exit":
        print(f"Goodbye {name}, I hope this was fun.")
        stop = True
    elif intent == "change name":
        name = input("please tell me your preferred name.\n")
        print(f"that has been noted, {name}")
    elif intent == "ask question":
        question = input("Feel free to ask\n")
        print(QueryProc(question))
    elif intent == "have a chat":
        smalltalk = input("How are you\n")
        print(talkProc(smalltalk),end="\n")
    elif intent == "show name":
        print(f"your name is {name}")
    else:
        print(intent) # return response to small talk


# report ideas Generative  limitations of similarity, self learning feature, information sourcing via web scraping to build database + self 


