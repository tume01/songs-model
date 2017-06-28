from sklearn.externals import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import json

def loadEncoders():
    df = pd.read_csv('data/pre_train.csv')
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X = X.values
    X = [element[0] for element in X]

    sentimentEncoder = LabelEncoder()
    sentimentEncoder = sentimentEncoder.fit(y)

    tfidf = TfidfVectorizer()

    tfidf = tfidf.fit(X)
    return tfidf, sentimentEncoder

def main():
    tfidf, sentimentEncoder = loadEncoders()
    songsModel = joblib.load('model_linear_svm_tfidf_w.pkl')
    beatlesDataFrame = pd.read_csv('./data/pre_train_beatles.csv')

    x = beatlesDataFrame.iloc[:, 0]
    x = tfidf.transform(x)
    y = songsModel.predict_proba(x)
    sentimentLabels = sentimentEncoder.inverse_transform(songsModel.classes_)
    songs = []
    for index, result in enumerate(y):
        song = {
            'name': beatlesDataFrame['name'][index],
            'album': beatlesDataFrame['album'][index],
            'lyric': beatlesDataFrame['lyric'][index],
            'tags': dict(zip(sentimentLabels, y[index]))
        }
        songs.append(song)

    print(json.dumps(songs))
    pass

if __name__ == '__main__':
    main()