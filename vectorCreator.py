from Sentence import Sentence
import json
from collections import Counter
import csv
from logTfIdF import probabilistic
import pandas as pd

def getWords(topWordsFile):
    topwords = json.load(topWordsFile)
    wordCounter = Counter()
    for key in topwords.keys():
        for word in topwords[key]:
            wordCounter.update([word[0]])
    return list(wordCounter.keys())

def main():
    with open('./newTopWords.json') as topWordsFile:
        words = getWords(topWordsFile)
        df = pd.read_csv('data/pre_train.csv', names=['lyric', 'class'])
        groupedSongs = {}
        for row in df.iterrows():
            if row[1][1] in groupedSongs:
                groupedSongs[row[1][1]].append(Sentence(row[1][0], row[1][1]))
            else:
                groupedSongs[row[1][1]] = [Sentence(row[1][0], row[1][1])]

        with open("data/tfidf_train.csv", "w") as vectorFile:
            vectorFileWriter = csv.writer(vectorFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
            vectorFileWriter.writerow(words + ['class'])
            for key in groupedSongs.keys():
                for song in groupedSongs[key]:
                    results = [probabilistic(word, song, groupedSongs, key) for word in words]
                    results += [key]
                    vectorFileWriter.writerow(results)


    pass

if __name__ == '__main__':
  main()