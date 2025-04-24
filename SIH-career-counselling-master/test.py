# libraries
import random
from tensorflow.keras.optimizers import SGD
from keras.layers import Dense, Dropout
from keras.models import load_model
from keras.models import Sequential
import numpy as np
import pickle
import json
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('wordnet')
# init file
words = []
classes = []
documents = []
ignore_words = ["?", "!"]
data_file = open(r"intents.json").read()
intents = json.loads(data_file)
print("Ok")
for intent in intents["intents"]:
    #iterating through every intent
    for pattern in intent["patterns"]:
        #Tokenizing each word using nltk
        w=nltk.word_tokenize(pattern,preserve_line=True)
        words.extend(w)
        #Appending a tuple of the word and its tag to document list
        documents.append((w,intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])
#Lemmatization:The process of grouping similar words together. Could be different tenses of the same word, synonyms wtc
lematizer=WordNetLemmatizer()
#Instantiating an object for WordNetLemmatizer class
words = [lematizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words=sorted(list(set(words)))
classes=sorted(list(set(classes)))
pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))


#initialising training data
training=[]
empty_output=[0]*len(classes)
for doc in documents:
    bag=[]
    pattern_words=doc[0]
    pattern_words=[lematizer.lemmatize(word.lower()) for word in pattern_words]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    output_row=list(empty_output)
    output_row[classes.index(doc[1])]=1
    training.append([bag,output_row])

random.shuffle(training)
print(type(training[0][0]))
training=np.array(training,dtype=object)

train_x=list(training[:,0])
train_y=list(training[:,1])


model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation="softmax"))
model.summary()

sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save("chatbot_model.h5", hist)
print("model created")



