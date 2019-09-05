# imports
from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, Permute, Dropout
from keras.layers import add,dot,concatenate
from keras.layers import CuDNNLSTM

# function that will create our model
def create_model(max_story_len,max_question_len,vocab_size):

    # initialise input_sequence and question input layers
    input_sequence=Input((max_story_len,))
    question=Input((max_question_len,))

    # Input gets embedded to a sequence of vectors
    # Input Encoder m
    input_encoder_m=Sequential()
    input_encoder_m.add(Embedding(input_dim=vocab_size,output_dim=64))
    input_encoder_m.add(Dropout(0.3))
    # This encoder will output:
    # (samples, story_maxlen, embedding_dim)

    # embed the input into a sequence of vectors of size query_maxlen
    # Input Encoder c
    input_encoder_c=Sequential()
    input_encoder_c.add(Embedding(input_dim=vocab_size,output_dim=max_question_len))
    input_encoder_c.add(Dropout(0.3))
    # output: (samples, story_maxlen, query_maxlen)

    # Question Encoder
    # embed the question into sequence of vectors
    question_encoder=Sequential()
    question_encoder.add(Embedding(input_dim=vocab_size,output_dim=64,input_length=max_question_len))
    question_encoder.add(Dropout(0.3))
    # output: (samples, query_maxlen, embedding_dim)

    # Encode input sequence and questions to sequences of dense vectors
    input_encoded_m=input_encoder_m(input_sequence)
    input_encoded_c=input_encoder_c(input_sequence)
    question_encoded=question_encoder(question)

    # pi = Softmax(uTmi)
    # shape: `(samples, story_maxlen, query_maxlen)`
    match=dot([input_encoded_m,question_encoded],axes=(2,2))
    match=Activation('softmax')(match)

    # o=Sum(pici)
    # add the match matrix with the second input vector sequence
    response=add([match,input_encoded_c]) # (samples, story_maxlen, query_maxlen)
    response=Permute((2,1))(response) # (samples, query_maxlen, story_maxlen)

    # ^a = Softmax(W(o + u))
    # concatenate the match matrix with the question vector sequence
    answer=concatenate([response,question_encoded])

    # Reduce with LSTM
    answer=CuDNNLSTM(32)(answer)  # (samples, 32)

    # Regularization with Dropout
    answer=Dropout(0.5)(answer)
    answer=Dense(vocab_size)(answer)  # (samples, vocab_size)

    # we output a probability distribution over the vocabulary
    answer=Activation('softmax')(answer)

    # build the final model
    model=Model([input_sequence,question],answer)
    model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

    print(model.summary())

    return model

