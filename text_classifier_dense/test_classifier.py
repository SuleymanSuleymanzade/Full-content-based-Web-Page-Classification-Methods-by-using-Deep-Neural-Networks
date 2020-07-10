import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
# print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.
import itertools
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix
from tensorflow import keras
layers = keras.layers
models = keras.models

data = pd.read_csv("./Dataset/bbc-text.csv")




TRAIN_PROPORTION = 0.8
max_words = 1000 # for tokenizer

train_size = int(len(data) * TRAIN_PROPORTION)


def train_test_split(data, train_size):
    train = data[:train_size]
    test = data[train_size:]
    return train, test

train_cat, test_cat = train_test_split(data['category'], train_size)
train_text, test_text = train_test_split(data['text'], train_size)


x_train = tokenize.texts_to_matrix(train_text)
x_test = tokenize.texts_to_matrix(test_text)

# Use sklearn utility to convert label strings to numbered index
encoder = LabelEncoder()
encoder.fit(train_cat)
y_train = encoder.transform(train_cat)
y_test = encoder.transform(test_cat)



# Converts the labels to a one-hot representation
num_classes = np.max(y_train) + 1
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# This model trains very quickly and 2 epochs are already more than enough
# Training for more epochs will likely lead to overfitting on this dataset
# You can try tweaking these hyperparamaters when using this model with your own data
batch_size = 32
epochs = 2
drop_ratio = 0.5


# Build the model
model = models.Sequential()
model.add(layers.Dense(512, input_shape=(max_words,)))
model.add(layers.Activation('relu'))
# model.add(layers.Dropout(drop_ratio))
model.add(layers.Dense(num_classes))
model.add(layers.Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# model.fit trains the model
# The validation_split param tells Keras what % of our training data should be used in the validation set
# You can see the validation loss decreasing slowly when you run this
# Because val_loss is no longer decreasing we stop training to prevent overfitting
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1)

# Evaluate the accuracy of our trained model
score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])



# Here's how to generate a prediction on individual examples

def get_examples(start, end):
	text_labels = encoder.classes_
	for i in range(start, end):
		print(x_test[i])
		print(len(x_test[i]))
		prediction = model.predict(np.array([x_test[i]]))  
		predicted_label = text_labels[np.argmax(prediction)]
		print(test_text.iloc[i][:50], "...")
		print('Actual label:' + test_cat.iloc[i])
		print("Predicted label: " + predicted_label + "\n")

#get_examples(1,2)

def predict_text(my_text):
	text_labels = encoder.classes_
	text_example = [my_text]
	new_tokenize = keras.preprocessing.text.Tokenizer(num_words=max_words, char_level=False)
	new_tokenize.fit_on_texts(train_text)
	vector_text = new_tokenize.texts_to_matrix(text_example)[0]
	prediction = model.predict(np.array([vector_text]))
	predicted_label = text_labels[np.argmax(prediction)]
	return predicted_label

my_text = "what's up man how are you doing? Are we gonna play football"
ans = predict_text(my_text)
print(ans)


'''
text_labels = encoder.classes_
text_example = ["hello mam wha't up how are you doind told me michael jackson"]
new_tokenize = keras.preprocessing.text.Tokenizer(num_words=max_words, char_level=False)
new_tokenize.fit_on_texts(train_text)
vector_text = new_tokenize.texts_to_matrix(text_example)[0]
prediction = model.predict(np.array([vector_text]))
predicted_label = text_labels[np.argmax(prediction)]
print(predicted_label)
'''

