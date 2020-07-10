import keras 
import numpy as np
from keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, Dropout, Embedding, LSTM, Flatten
from keras.models import Model
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
from keras.utils.vis_utils import plot_model
from keras.layers.merge import concatenate
from sklearn.metrics import confusion_matrix
import pydot_ng as pydot
from keras.models import load_model

data = pd.read_csv('datasets/bbc-text.csv')
data['target'] = data['category'].astype('category').cat.codes
data['num_words'] = data['text'].apply(lambda x: len(x.split()))

bins = [0, 50, 75, np.inf]

data['bins'] = pd.cut(data['num_words'], bins=[0, 100, 300, 500, 800, np.inf],
	labels = ['0-100', '100-300', '300-500', '500-800', '>800'])

word_distribution = data.groupby('bins')\
	.size()\
	.reset_index()\
	.rename(columns = {0: 'counts'})

#print(word_distribution.head())
'''
sns.barplot(x='bins', y='counts', data=word_distribution).\
	set_title("Word distribution in bbc-text dataset")
'''
#print(word_distribution.head())
#print(data['target'])

num_class = len(np.unique(data['category'].values))
y = data['target'].values
#print(y)
MAX_LENGTH = 500
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['text'].values)
post_seq = tokenizer.texts_to_sequences(data['text'].values)
post_seq_padded = pad_sequences(post_seq, maxlen = MAX_LENGTH)

X_train, X_test, y_train, y_test = train_test_split(post_seq_padded, y, test_size = 0.05)


model = load_model('model.h5')



predicted = model.predict([X_test, X_test, X_test])
predicted = np.argmax(predicted, axis=1)

#cm = confusion_matrix(y_test, predicted)

#plt.figure(figsize = (5.8, 4))
#plt.title('Neural Network \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, predicted)))
#sns.heatmap(cm, annot=True)
#plt.show()

print(accuracy_score(y_test, predicted))
print(predicted)