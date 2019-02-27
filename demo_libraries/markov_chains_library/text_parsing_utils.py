from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

## load and preprocess text
def load_preprocess(csvname):
    # load in text dataset - lower case all
    text = open(csvname).read().lower()
    
    # cut out first chunk of giberish text
    text = text[947:]

    # remove some obvious tag-related gibberish throughout
    characters_to_remove = ['0','1','2','3','4','5','6','7','8','9','_','[',']','}','.  .  .','\\']
    for i in characters_to_remove:
        text = text.replace(i,'')

    # some gibberish that looks like it needs to be replaced with a ' '
    text = text.replace('\n',' ')
    text = text.replace('\r',' ')
    text = text.replace('--',' ')
    text = text.replace(',,',' ')
    text = text.replace('   ',' ')
    
    return text

## parse a text into words
def parse_words(text):
    # load in function from scikit learn that 
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform([text])
    analyze = vectorizer.build_analyzer()

    # get all unique words in input corpus
    tokens = analyze(text)
    unique_words = vectorizer.get_feature_names() 
    
    # unique nums to map words too
    unique_nums = np.arange(len(unique_words))

    # this dictionary is a function mapping each unique word to a unique integer
    words_to_keys = dict((i, n) for (i,n) in zip(unique_words,unique_nums))

    # this dictionary is a function mapping each unique integer to a unique word
    keys_to_words = dict((i, n) for (i,n) in zip(unique_nums,unique_words))
    
    # convert all of our tokens (words) to keys
    keys = [words_to_keys[a] for a in tokens]
    
    return tokens,keys,words_to_keys,keys_to_words

### parse text into characters
def parse_chars(text):
    # break text into individual characters
    chars = list(text)
    
    # count the number of unique characters in the text
    unique_chars = sorted(list(set(text)))
    
   # unique number range to map characters too
    unique_nums = np.arange(0,len(unique_chars))

    # this dictionary is a function mapping each unique character to a unique integer
    chars_to_keys = dict((i, n) for (i,n) in zip(unique_chars,unique_nums))

    # this dictionary is a function mapping each unique integer to a unique character
    keys_to_chars = dict((i, n) for (i,n) in zip(unique_nums,unique_chars))
    
    # convert all of our characters to keys
    keys = [chars_to_keys[a] for a in text]
    
    return chars,keys,chars_to_keys,keys_to_chars