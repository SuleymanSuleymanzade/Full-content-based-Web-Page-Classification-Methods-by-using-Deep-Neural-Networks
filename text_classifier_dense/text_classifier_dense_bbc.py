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
#import seaborn as sns
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # desable warning



#https://scikit-learn.org/0.18/auto_examples/model_selection/plot_confusion_matrix.html    for confusion matrix

data_path = os.path.join(os.path.dirname(__file__), 'Dataset')
data_path = os.path.join(data_path, 'bbc-text.csv')
data = pd.read_csv(data_path)


TRAIN_PROPORTION = 0.8
train_size = int(len(data) * TRAIN_PROPORTION)


class TextClassifierDenseBBC:
	def __train_test_split(self, data, train_size):
		train = data[:train_size]
		test = data[train_size:]
		return train, test

	def __init__(self, max_words = 1000):
		self.max_words = max_words
		train_cat, test_cat = self.__train_test_split(data['category'], train_size)
		train_text, test_text = self.__train_test_split(data['text'], train_size)

		self.tokenize = keras.preprocessing.text.Tokenizer(num_words=self.max_words, char_level=False)
		self.tokenize.fit_on_texts(train_text) # fit tokenizer to our training text data

		self.x_train = self.tokenize.texts_to_matrix(train_text)
		self.x_test = self.tokenize.texts_to_matrix(test_text)

		self.encoder = LabelEncoder()
		self.encoder.fit(train_cat)
		self.y_train = self.encoder.transform(train_cat)
		self.y_test = self.encoder.transform(test_cat)

		self.num_classes = np.max(self.y_train) + 1
		self.y_train = keras.utils.to_categorical(self.y_train, self.num_classes)
		self.y_test = keras.utils.to_categorical(self.y_test, self.num_classes)

	def build_model(self, drop_ratio = 0.5):

		self.model = models.Sequential()
		self.model.add(layers.Dense(512, input_shape=(self.max_words,)))
		self.model.add(layers.Activation('relu'))
		# model.add(layers.Dropout(drop_ratio))
		self.model.add(layers.Dense(self.num_classes))
		self.model.add(layers.Activation('softmax'))

		self.model.compile(loss='categorical_crossentropy',
			optimizer='adam',
			metrics=['accuracy'])

	def train_model(self, batch_size = 32, epochs = 2, show_status = False):
		self.history = self.model.fit(self.x_train, self.y_train,
			batch_size=batch_size,
			epochs=epochs,
			verbose=1,
			validation_split=0.1)
		# Evaluate the accuracy of our trained model
		self.score = self.model.evaluate(self.x_test, self.y_test,
			batch_size=batch_size, verbose=1)
		
		if show_status:
			print('Test loss:', self.score[0])
			print('Test accuracy:', self.score[1])

	def predict_from_digital_data(self, digital_data):
		text_labels = self.encoder.classes_
		predicted_labels = text_labels[np.argmax(digital_data)]
		return predicted_labels

	def predict_text(self, my_text):
		text_labels = self.encoder.classes_
		text_example = [my_text]
		#new_tokenize = keras.preprocessing.text.Tokenizer(num_words=max_words, char_level=False)
		#new_tokenize.fit_on_texts(train_text)
		vector_text = self.tokenize.texts_to_matrix(text_example)[0]
		prediction = self.model.predict(np.array([vector_text]))
		predicted_label = text_labels[np.argmax(prediction)]
		return (predicted_label, prediction)

	def __get_predicted_and_estimated_data(self):
		''' Data for Confusion matrix '''
		y_softmax = self.model.predict(self.x_test)
		self.y_test_1d = []
		self.y_pred_1d = []

		for i in range(len(self.y_test)):
		    probs = self.y_test[i]
		    index_arr = np.nonzero(probs)
		    one_hot_index = index_arr[0].item(0)
		    self.y_test_1d.append(one_hot_index)

		for i in range(0, len(y_softmax)):
		    probs = y_softmax[i]
		    predicted_index = np.argmax(probs)
		    self.y_pred_1d.append(predicted_index)

	def __confusion_matrix_util(self, cm, classes, title='Confusion matrix',cmap=plt.cm.Blues):
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		plt.imshow(cm, interpolation='nearest', cmap=cmap)
		plt.title(title, fontsize=30)
		plt.colorbar()
		tick_marks = np.arange(len(classes))
		plt.xticks(tick_marks, classes, rotation=45, fontsize=22)
		plt.yticks(tick_marks, classes, fontsize=22)
		
		fmt = '.2f'
		thresh = cm.max() / 2.
		for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
			plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
				color="white" if cm[i, j] > thresh else "black")
		plt.ylabel('True label', fontsize=25)
		plt.xlabel('Predicted label', fontsize=25)

	def plot_confusion_matrix(self):
		self.__get_predicted_and_estimated_data() # init self.y_pred_id and self.y_test_id
		cnf_matrix = confusion_matrix(self.y_test_1d, self.y_pred_1d)
		plt.figure(figsize=(24,24))
		text_labels = list(self.encoder.classes_)
		self.__confusion_matrix_util(cnf_matrix, classes=text_labels, title="Confusion matrix")
		#sns.set(font_scale=3.0)
		plt.show()


'''
text_classifier = TextClassifierDenseBBC()
text_classifier.build_model()
text_classifier.train_model(show_status = True)
text_classifier.plot_confusion_matrix()

my_text = "president Macron met with Obama then they meed prime minister of Britain"
my_text2 = "two guys play football"
res = text_classifier.predict_text(my_text)
print('-------------------------------')
print(res[0])
print(res[1])
res = text_classifier.predict_text(my_text2)
print('-------------------------------')
print(res[0])
print(res[1])
'''