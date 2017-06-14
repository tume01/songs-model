import numpy as np
from sklearn.svm import SVC
import pandas as pd
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB

def create_train():
  df = pd.read_csv('data/pre_train_comments.csv')
  X = df.iloc[:, :-1]
  y = df.iloc[:, -1]
  X = X.values
  X = [element[0] for element in X]

  sentiment_encoder = LabelEncoder()
  y = sentiment_encoder.fit_transform(y)

  tfidf = TfidfVectorizer()

  X = tfidf.fit_transform(pd.DataFrame({'comments':X})['comments'].values.astype('U'))


  X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=1)

  svm_model = SVC()

  search_parameters = {
      'degree': [2, 3, 4], 'gamma': (np.arange(0.1, 1.1, 0.1)), 'C': (np.arange(0.1, 1.1, 0.1)), 'kernel': ['poly']
  }
  model_svm = GridSearchCV(
      svm_model, search_parameters, cv=10, scoring='accuracy', n_jobs=8)
  model_svm.fit(X_train, y_train)
  print(model_svm.best_score_)
  print(model_svm.best_params_)
  y_pred = model_svm.predict(X_test)
  print(metrics.accuracy_score(y_pred, y_test))
  print("model trained")

  # gnb = GaussianNB()
  # gnb.fit(X_train, y_train)
  # y_pred = gnb.predict(X_test)
  # print(metrics.accuracy_score(y_pred, y_test))

def main():
  create_train()

if __name__ == '__main__':
  main()