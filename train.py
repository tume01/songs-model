import numpy as np
from sklearn.svm import SVC
import pandas as pd
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import json
from collections import Counter

def getWords(topWordsFile):
    topwords = json.load(topWordsFile)
    wordCounter = Counter()
    for key in topwords.keys():
        for word in topwords[key]:
            wordCounter.update([word[0]])
    return list(wordCounter.keys())

def create_train():
  df = pd.read_csv('data/pre_train.csv')
  X = df.iloc[:, :-1]
  y = df.iloc[:, -1]
  X = X.values
  X = [element[0] for element in X]

  sentiment_encoder = LabelEncoder()
  y = sentiment_encoder.fit_transform(y)
  print(sentiment_encoder.classes_)
  with open('./newTopWords.json') as topWordsFile:
    vocabulary = getWords(topWordsFile)
  tfidf = TfidfVectorizer()

  X = tfidf.fit_transform(X)


  X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=1)

  svm_model = SVC(probability=True, kernel='linear', C=1, class_weight='balanced')

  search_parameters = {
      'degree': [2, 3, 4], 'gamma': (np.arange(0.1, 1.1, 0.1)), 'C': (np.arange(0.1, 1.1, 0.1)), 'kernel': ['linear']
  }
  # model_svm = GridSearchCV(
  #     svm_model, search_parameters, cv=5, scoring='accuracy', n_jobs=8)
  svm_model.fit(X_train, y_train)
  # print(model_svm.best_score_)
  # print(model_svm.best_params_)
  y_pred = svm_model.predict(X_test)
  print(metrics.accuracy_score(y_pred, y_test))
  print("model trained")

  # gnb = MultinomialNB()
  # gnb.fit(X_train, y_train)
  # y_pred = gnb.predict(X_test)
  # print(metrics.accuracy_score(y_pred, y_test))
  joblib.dump(svm_model, 'model_linear_svm_tfidf_w.pkl')

def main():
  create_train()

if __name__ == '__main__':
  main()