import json

def countCategory(category):
    with open('./beatlesResults.json') as jsonFile:
        songs = json.load(jsonFile)

    return 0

def main():
    print(countCategory('happy'))
    print(countCategory('sad'))
    print(countCategory('sex'))
    print(countCategory('breakup'))
    print(countCategory('death'))
    print(countCategory('love'))
    pass

if __name__ == '__main__':
    main()