from math import log10
import numpy as np
from sklearn.svm import SVC
import pandas as pd
from collections import Counter
import json
from Sentence import Sentence

def termFrequency(term, sentence):
    return sentence.countWord(term)

def calculateA(term, documents, category):
    categoryDocuments = documents[category]
    return sum(1 for document in categoryDocuments if document.countWord(term) > 0)

def calculateB(term, documents, category):
    totalCount = 0
    for key in documents.keys():
        if key != category:
            totalCount += sum(1 for document in documents[key] if document.countWord(term) > 0)
    return totalCount

def calculateC(term, documents, category, A=None):
    categoryDocuments = documents[category]
    if A != None:
        return len(categoryDocuments) - A
    else:
        return sum(1 for document in categoryDocuments if document.countWord(term) == 0)

def calculateD(term, documents, category, B=None):
    totalCount = 0
    if B != None:
        total = 0
        for key in documents.keys():
            if key != category:
                total += len(documents[key])
        return total - B
    else:
        for key in documents.keys():
            if key != category:
                totalCount += sum(1 for document in documents[key] if document.countWord(term) == 0)
        return totalCount

def probabilistic(term, text, documents, category):
    # print("term: ", term)
    A = calculateA(term, documents, category)
    # print("A: ", A)
    B = calculateB(term, documents, category)
    # print("B: ", B)
    C = calculateC(term, documents, category, A=A)
    # print("C: ", C)
    D = calculateD(term, documents, category, B=B)
    # print("D: ", D)
    return termFrequency(term, text) * log10(1 + ((A / (B + 1)) * (A / (C + 1))))


def main():
    df = pd.read_csv('data/pre_train.csv', names=['lyric', 'class'])
    groupedSongs = {}
    for row in df.iterrows():
        if row[1][1] in groupedSongs:
            groupedSongs[row[1][1]].append(Sentence(row[1][0], row[1][1]))
        else:
            groupedSongs[row[1][1]] = [Sentence(row[1][0], row[1][1])]
    results = {}
    for key in groupedSongs.keys():
        scores = {}
        for song in groupedSongs[key]:
            for word in song.words():
                if word in scores:
                    wordWeight = probabilistic(word, song, groupedSongs, key)
                    if scores[word] < wordWeight :
                        scores[word] = wordWeight
                else:
                    scores[word] = probabilistic(word, song, groupedSongs, key)
        sortedScores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results[key] = sortedScores[:50]
    print(json.dumps(results))

if __name__ == '__main__':
  main()