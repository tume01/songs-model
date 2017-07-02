import matplotlib.pyplot as plt
import numpy as np
import csv
import json
from sklearn.svm import SVC
import pandas as pd
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    #plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.save('salida.png')
    
def tagging():
  df = pd.read_csv('data/tfidf_train.csv')
  X = df.iloc[:, :-2]
  y = df.iloc[:, -2]
  t=df.iloc[:,-1:]
  svm_model = 
  #y_pred = svm_model.predict(X)
  lclases=svm_model.classes_   
  cm =confusion_matrix(y, y_pred, lclases)
  plot_confusion_matrix( cm, lclases )      
  
  salida=svm_model.predict_proba(X)
  total=np.concatenate( (t,salida),axis=1 )

  CL={}  
  for it in total:
    CL[int(it[0])] =   list(it[1:])   
    
  with open('./new_songs.json') as data_file:
    songs = json.load(data_file)
    nsongs=[]
    for ind,song in enumerate(songs):
        if ind in CL:
            song['tagof']= lclases[ np.argmax(CL[ind]) ] 
            song['probs']= CL[ind]
            nsongs.append(song)
    json.dump( nsongs, open('data/salida_nuevo.json','w'))        
def main():
  tagging()

if __name__ == '__main__':
  main()