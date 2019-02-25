from sklearn.feature_extraction.text import CountVectorizer
from autograd import numpy as np

class Text:
   
    ## load and preprocess text
    def load_preprocess(self,csvname):
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

        self.text = text

    ### parse text into characters
    def parse_chars(self,**kwargs):
        # break text into individual characters
        self.chars = list(self.text)

        # count the number of unique characters in the text
        self.unique_chars = sorted(list(set(self.text)))

       # unique number range to map characters too
        self.unique_keys = np.arange(0,len(self.unique_chars))
        self.num_keys = np.size(self.unique_keys)

        # this dictionary is a function mapping each unique character to a unique integer
        self.chars_to_keys = dict((i, n) for (i,n) in zip(self.unique_chars,self.unique_keys))

        # this dictionary is a function mapping each unique integer to a unique character
        self.keys_to_chars = dict((i, n) for (i,n) in zip(self.unique_keys,self.unique_chars))

        # convert all of our characters to keys
        self.keys = [self.chars_to_keys[a] for a in self.text]
        num_keep = 20000
        if 'num_keep' in kwargs:
            num_keep = kwargs['num_keep']
        self.keys = np.array(self.keys[:num_keep])[np.newaxis,:]

    # parse an input sequence
    def window_series(self,x,order):
        self.order = order 
        
        # containers for input/output pairs
        x_in = []
        x_out = []
        T = x.size

        # window data
        for t in range(T - order):
            # get input sequence
            temp_in = x[:,t:t + order]
            x_in.append(temp_in)

            # get corresponding target
            temp_out = x[:,t + order]
            x_out.append(temp_out)

        # make array and cut out redundant dimensions
        x_in = np.array(x_in)
        x_in = x_in.swapaxes(0,1)[0,:,:].T
        x_out = np.array(x_out).T
        return x_in,x_out

    # transform character-based input/output into equivalent numerical versions
    def encode_io_pairs_fixed(self,keys,order):    
        self.order = order 

        # window series
        x,y = self.window_series(keys,order)

        # dimensions of windowed data
        order,num_data = x.shape

        # loop over inputs/outputs and tranform and store in x
        x_onehot = []
        y_onehot = []
        for n in range(num_data):
            # one-hot encode input
            temp = np.zeros((order,self.num_keys))
            for o in range(order):
                temp[o,x[:,n][o]] = 1
            x_onehot.append(temp.flatten())
            
            # one-hot encode output
            temp = np.zeros((self.num_keys,1))
            temp[y[:,n][0],0] = 1
            y_onehot.append(temp)

        # one hot encode output too
        x_onehot = np.array(x_onehot).T
        y_onehot = np.array(y_onehot)[:,:,0].T
        return x_onehot,y,y_onehot

    # generate text given inputs
    def generate_text(self,model,normalizer,w,starter_ind,num_chars,order):
        # get weights to use        
        # use transition matrix to generate sentence of desired length
        # starting at randomly chosen word (all of this is done using
        # the associated keys, then re-translated into words)
        generated_keys = self.keys[:,starter_ind:starter_ind+order]
        generated_keys = generated_keys.tolist()[0]

        # run over 
        for i in range(num_chars):
            # get current key
            prev_keys = generated_keys[i:i+order]
            x_prev = np.array(prev_keys)[:,np.newaxis]

            ### one-hot encode ###
            x_onehot = []
            temp = np.zeros((order,self.num_keys))
            for o in range(order):
                temp[o,x_prev[:,0][o]] = 1
            x_onehot.append(temp.flatten())
            x_onehot = np.array(x_onehot).T
            
            # normalize input
            x_onehot = normalizer(x_onehot)

            # use maximum index of this distribution in transition matrix
            # to get next key
            a = model(x_onehot,w).T
            next_key = np.argmax(a,axis = 1).flatten()[0]

            # store next key
            generated_keys.append(next_key)

        # translate generated keys back into characters and print
        generated_chars = ''
        for key in generated_keys:
            next_char = self.keys_to_chars[key]
            generated_chars += next_char

        # return predictions
        sentence = ''.join(generated_chars)

        # seperate seed from generated component
        seed = generated_chars[:order]
        seed =  ''.join(seed)
        generated = generated_chars[order:]
        generated =  ''.join(generated)

        # print true text
        print ('-------- TRUE TEXT -------')
        true_text = [self.chars[s] for s in range(starter_ind,starter_ind + order + num_chars)]
        true_text  = ''.join(true_text)
        print (true_text)
        print ('\n')

        # print seed and generated component
        print ('-------- ORDER = ' + str(order) + ' MODEL TEXT -------')
        print('\x1b[31m' + seed + '\x1b[0m' + '' + '\x1b[34m' + generated + '\x1b[0m')