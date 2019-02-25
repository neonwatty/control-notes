from . import word_level_markov_model as util

def show_order(csvname,order,num_words):
    # get instance of markov model, load in text
    # load in and preprocess text
    model = util.Markov(csvname)
    
    # produce probabilities for order O model
    model.make_transition_matrix(order = order)
    model.generate_text(num_words)