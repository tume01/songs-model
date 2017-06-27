from collections import Counter

class Sentence(object):
    """docstring for Sentence"""
    def __init__(self, text, category):
        super(Sentence, self).__init__()
        self.text = text
        self.category = category
        self.wordCounter = self.countWords(text)

    def countWords(self, text):
        return Counter(text.split(" "))

    def countWord(self, word):
        if word in self.wordCounter:
            return self.wordCounter[word]
        return 0

    def words(self):
        return self.wordCounter.keys()