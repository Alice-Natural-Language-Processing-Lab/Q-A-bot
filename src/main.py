# imports
import create_feature_sets as cf
from keras.preprocessing.text import Tokenizer
import model as m
from keras.models import load_model
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

# function that saves the result curve
def save_results(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("../results.png")

# the main function
def main():
    
    # load data
    train_data,test_data=cf.get_data()

    # create vocab
    vocab,all_data=cf.create_vocab(train_data,test_data)

    # integer encode sequences of words
    tokenizer = Tokenizer(filters=[])
    tokenizer.fit_on_texts(vocab)

    # some variables to be used later
    vocab_size=len(vocab)+1 #we add an extra space to hold a 0 for Keras's pad_sequences

    # max_story_len
    max_story_len=max([len(data[0]) for data in all_data])

    # max_question_len
    max_question_len=max([len(data[1]) for data in all_data])

    word_index=tokenizer.word_index

    # create feature vectors
    inputs_train,questions_train,answers_train=cf.vectorize_data(train_data, tokenizer.word_index,max_story_len,max_question_len)
    inputs_test,questions_test,answers_test = cf.vectorize_data(test_data, tokenizer.word_index,max_story_len,max_question_len)

    # create model
    model=m.create_model(max_story_len,max_question_len,vocab_size)

    # train
    history = model.fit([inputs_train, questions_train], answers_train,batch_size=32,epochs=100,validation_data=([inputs_test, questions_test], answers_test))

    # save model
    filename="../model/chatbot_100_epochs.h5"
    model.save(filename)
    
    # summarize history for accuracy
    save_results(history)

    # predict the results
    model=load_model("../model/chatbot_100_epochs.h5")

    # compile the model
    model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

    # predicting on your own story and question
    my_story = "John left the kitchen . Sandra dropped the football in the garden ."
    my_question = "Is the football in the garden ?"

    # prepare data for prediction
    mydata = [(my_story.split(),my_question.split(),'yes')]
    
    # vectorize the data
    my_story,my_question,my_ans=cf.vectorize_data(mydata,word_index,max_story_len,max_question_len)

    # predict
    preds=model.predict(([my_story,my_question]))

    #Generate prediction from model
    val_max = np.argmax(preds[0])

    for key, val in tokenizer.word_index.items():
        if val == val_max:
            k = key
    print("Predicted answer is: ", k)
    print("Probability of certainty was: ", preds[0][val_max])

    """
    Predicted answer is:  yes
    Probability of certainty was:  0.99996233
    """

if __name__ == '__main__':
    main()
    