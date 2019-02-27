# custom imports
from . import text_parsing_utils as util

# standard imporrts
import numpy as np


class Markov:
    def __init__(self,csvname):
        # preprocess input text (remove bad characters, make all lowercase, etc.,)
        self.text = util.load_preprocess(csvname)
        
        # parse into individual words
        self.tokens,self.keys,self.words_to_keys,self.keys_to_words = util.parse_words(self.text)
        
        # generate starting index of generative model - do this here so different order models
        # can be more easily compared
        self.starter_ind = np.random.permutation(len(self.tokens))[0]
        self.starter_word = self.tokens[self.starter_ind]
        self.starter_key = self.words_to_keys[self.starter_word]
    
    # make transition probabilities based on discrete count of input text
    def make_transition_matrix(self,order):
        # get unique keys - for dimension of transition matrix 
        unique_keys = np.unique(self.keys)
        num_unique_words = len(unique_keys)
        num_words = len(self.tokens)

        # generate initial zeros order O transition matrix
        # use a dictionary - or else this for sure won't scale 
        # to any order > 1
        transition_matrix = {}

        # sweep through tokens list, update each individual distribution
        # as you go - each one a column
        for i in range(order,num_words):
            # grab current key, and previous order keys
            next_key = self.keys[i]
            prev_keys = tuple(self.keys[i-order:i])

            ## update transition matrix
            # we've seen current key already
            if prev_keys in transition_matrix.keys():
                if next_key in transition_matrix[prev_keys].keys():
                    transition_matrix[prev_keys][next_key] += 1
                else:
                    transition_matrix[prev_keys][next_key] = 1
            else:      # we haven't seen key already, so create new subdict
                transition_matrix[prev_keys] = {}
                transition_matrix[prev_keys][next_key] = 1
                
        # assign to global
        self.order = order
        self.transition_matrix = transition_matrix
        
    def generate_text(self,num_words):
        # use transition matrix to generate sentence of desired length
        # starting at randomly chosen word (all of this is done using
        # the associated keys, then re-translated into words)
        generated_words = self.tokens[self.starter_ind:self.starter_ind +self.order]
        generated_keys = [self.words_to_keys[s] for s in generated_words]

        # produce next keys / words
        for i in range(num_words):
            # get current key
            prev_keys = tuple(generated_keys[i:i+self.order])

            # use maximum index of this distribution in transition matrix
            # to get next key
            stats = self.transition_matrix[prev_keys]
            next_key = max(stats, key=lambda key: stats[key])

            # store next key
            generated_keys.append(next_key)
            
        # translate generated keys back into words and print
        for n in range(self.order,len(generated_keys)):
            key = generated_keys[n]           
            word = self.keys_to_words[key]
            generated_words.append(word)
            
        # return predictions
        sentence = ' '.join(generated_words)
        
        # seperate seed from generated component
        seed = generated_words[:self.order]
        self.seed =  ' '.join(seed)
        generated = generated_words[self.order:]
        self.generated =  ' '.join(generated)
        
        # print true text
        print ('-------- TRUE TEXT -------')
        true_text = [self.tokens[s] for s in range(self.starter_ind,self.starter_ind + self.order + num_words)]
        true_text  = ' '.join(true_text)
        print (true_text)
        print ('\n')
        
        # print seed and generated component
        print ('-------- ORDER = ' + str(self.order) + ' MODEL TEXT -------')
        print('\x1b[31m' + self.seed + '\x1b[0m' + ' ' + '\x1b[34m' + self.generated + '\x1b[0m')