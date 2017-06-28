import nltk
import csv
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import json
from collections import Counter

wordnet_lemmatizer = WordNetLemmatizer()
translator = str.maketrans('', '', string.punctuation)

with open('tagsTASI.json') as data_file:
    data = json.load(data_file)

T = {}
T['love'] = ['love','romance','romantic','love song','soulmate','relationship','relationships','marriage','crush','love triangle','sweet','young love','true love']
T['sad'] = ['lonely','depression','sad','loneliness','sadness','dark','broken','alone','emo','darkness','isolation','cry','homesick']
T['breakup'] = ['breakup','break up','break-up','heartbreak','cheating','betrayal','divorce','infidelity','disillusionment','betrayal','disappointment','guilt','heartache','jealousy','leaving','unrequited love']
T['nostalgia'] = ['melancholia','loss','melancholy','memories','regret','nostalgia','homesick','longing']
T['death'] = ['suicidal','suicide','death','war','murder','abuse','pain','evil','funeral','drugs','addiction','alcoholism','heroin','bullying','satan','domestic abuse','cancer','cocaine','weed','dying']
T['sex'] = ['sex','rape','sexy','oral sex','masturbation','sexuality','prostitution','lust','virginity','desire','voyeurism']
T['happy'] = ['happysongs','happy','happiness','beauty','comedy','inspiration','dream','empathy','life','christmas','holiday','hope','faith','friendship','beautiful','amazing','freedom','peace','family','friends','awesome','perseverance','funny']
T = data
def pre_process(text):
  tokens = [word.lower() for word in nltk.word_tokenize(text)]
  stop_words = stopwords.words("english") + [str(element) for element in string.punctuation] + ['n\'t', '\'m', '...', '``', "''", "'s", '..', "'re", "'ve", "'d", "'ll", 'oh', 'na']
  no_stop_tokens = [word for word in tokens if word not in stop_words]
  lemmas = [wordnet_lemmatizer.lemmatize(word) for word in no_stop_tokens]
  return lemmas

def count_songs():
  with open('./new_songs.json') as data_file:
    songs = json.load(data_file)
    songs_tag_counter = Counter()
    for song in songs:
      tags = [tag.lower() for tag in song['tags']]
      for sentiment in T:
        for tag in tags:
          if tag in T[sentiment]:
            songs_tag_counter.update([sentiment])

    print(songs_tag_counter.most_common(10))

def create_train():
  with open('./TheBeatles_fix.json') as data_file:
    albums = json.load(data_file)
    with open("data/pre_train_beatles.csv", "w") as pre_train_file:
      with open("data/pre_train_comments_beatles.csv", "w") as pre_train_comments_file:
        pre_train_writer = csv.writer(pre_train_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
        pre_train_comments = csv.writer(pre_train_comments_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
        pre_train_writer.writerow([
            'lyric',
            'name',
            'album'
          ])
        for album in albums:
          for song in album['songs']:
            lyric = pre_process(song['lyrics'])
            # comment = pre_process(" ".join(str(element) for element in song['comments']))
            lyric = " ".join(str(element) for element in lyric)
            # comment = " ".join(str(element) for element in comment)
            # tags = [tag.lower() for tag in song['tags']]
            pre_train_writer.writerow([
                    lyric,
                    song['name'],
                    album['album']
                    ])
            # for sentiment in T:
            #    for tag in tags:
            #     if tag in T[sentiment]:
            #       pre_train_writer.writerow([
            #         lyric,
            #         sentiment
            #         ])
            #       pre_train_comments.writerow([
            #         comment,
            #         sentiment
            #         ])

def main():
  create_train()

if __name__ == '__main__':
  main()