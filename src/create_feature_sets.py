# imports
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences

# load the data
def get_data():

    # load the train data
    with open("../data/train_qa.txt","rb") as file:
        
        # save in train_data
        train_data=pickle.load(file)

    # load the test data
    with open("../data/test_qa.txt","rb") as file:

        # save in test_data
        test_data=pickle.load(file)

    print("Data loaded!")
    print(f"Length of train_data = {len(train_data)}")
    print(f"Length of test_data = {len(test_data)}")

    return train_data,test_data

# setting up vocabulary of all the words
def create_vocab(train_data,test_data):
    
    # this will hold all the vocab words
    vocab=set()

    # all of our data
    all_data=train_data+test_data

    # iterate over all the story, question, answer pairs and add them to all_data
    for story,question,answer in all_data:
        
        # add all words from story
        vocab=vocab.union(set(story))

        # add all words from question
        vocab=vocab.union(set(question))

    # add "yes" and "no" as possible answers to vocab
    vocab.add('no')
    vocab.add('yes')

    return vocab,all_data

# function to create train and test features
def vectorize_data(data, word_index, max_story_len,max_question_len):
    '''
    INPUT: 
    
    data: consisting of Stories,Queries,and Answers
    word_index: word index dictionary from tokenizer
    max_story_len: the length of the longest story (used for pad_sequences function)
    max_question_len: length of the longest question (used for pad_sequences function)


    OUTPUT:
    
    Vectorizes the stories,questions, and answers into padded sequences. We first loop for every story, query , and
    answer in the data. Then we convert the raw words to an word index value. Then we append each set to their appropriate
    output list. Then once we have converted the words to numbers, we pad the sequences so they are all of equal length.
    
    Returns this in the form of a tuple (X,Xq,Y) (padded based on max lengths)
    '''

    # X = Stories
    X=[]

    # Xq = Question
    Xq=[]

    # Y = Correct answer
    Y=[]

    # iterate over all the story, question, answer pairs and add them to all_data
    for story, question, answer in data:

        # Grab the word index for every word in story
        x=[word_index[word.lower()] for word in story]

        # Grab the word index for every word in question
        xq=[word_index[word.lower()] for word in question]

        # Grab the Answers (either Yes/No)
        # Index 0 is reserved so we're going to use + 1
        y=np.zeros(len(word_index)+1)

        # y is all 0s so is yes set it =1
        y[word_index[answer]]=1

        # Append each set to final
        X.append(x)
        Xq.append(xq)
        Y.append(y)

        # Finally, pad the sequences based on their max length so the RNN can be trained on uniformly long sequences.
        
    # RETURN TUPLE FOR UNPACKING
    return (pad_sequences(X, maxlen=max_story_len),pad_sequences(Xq, maxlen=max_question_len), np.array(Y))