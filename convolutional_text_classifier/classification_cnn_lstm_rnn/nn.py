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
from keras.utils.vis_utils import plot_model
from keras.layers.merge import concatenate
from sklearn.metrics import confusion_matrix


from keras import backend as K
K.tensorflow_backend._get_available_gpus()


plt.style.use('ggplot')
data = pd.read_csv('datasets/bbc-text.csv')

#print(data.head())
# bbc news dataset [category, text]
#print(data['category'].value_counts())
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

MAX_LENGTH = 1000
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['text'].values)
post_seq = tokenizer.texts_to_sequences(data['text'].values)
post_seq_padded = pad_sequences(post_seq, maxlen = MAX_LENGTH)



X_train, X_test, y_train, y_test = train_test_split(post_seq_padded, y, test_size = 0.05)

vocab_size = len(tokenizer.word_index) + 1


inputs1 = Input(shape=(MAX_LENGTH, ))
embedding_layer1 = Embedding(vocab_size, 128, input_length = MAX_LENGTH)(inputs1)
x1 = Flatten()(embedding_layer1)
x1 = Dense(32, activation = 'relu')(x1)


inputs2 = Input(shape=(MAX_LENGTH, ))
embedding_layer2 = Embedding(vocab_size, 128, input_length = MAX_LENGTH)(inputs2)
x2 = Flatten()(embedding_layer2)
x2 = Dense(32, activation = 'relu')(x2)


inputs3 = Input(shape=(MAX_LENGTH, ))
embedding_layer3 = Embedding(vocab_size, 128, input_length = MAX_LENGTH)(inputs3)
x3 = Flatten()(embedding_layer3)
x3 = Dense(32, activation = 'relu')(x3)


merged = concatenate([x1, x2, x3])

common_dense = Dense(10, activation = 'relu')(merged)
predictions = Dense(num_class, activation = 'softmax')(common_dense)

model = Model(inputs= [inputs1, inputs2, inputs3], outputs=predictions)
model.compile(optimizer = 'adam',
	loss = 'categorical_crossentropy',
	metrics = ['acc'])



model.summary()
plot_model(model,show_shapes=True,to_file = 'generated_models/bbc_model_structure.png')


file_path = "weights-simple.hdf5"
checkpointer = ModelCheckpoint(file_path, monitor='val_acc',
	verbose = 1, save_best_only = True, mode = 'max')
history = model.fit([X_train, X_train, X_train], batch_size = 64, y = to_categorical(y_train),
	verbose =  1, validation_split = 0.25,
	shuffle = True, epochs = 5, callbacks = [checkpointer])


df = pd.DataFrame({'epochs':history.epoch, 'accuracy': history.history['acc'], 'validation_accuracy': history.history['val_acc']})
g = sns.pointplot(x="epochs", y="accuracy", data=df, fit_reg=False)
g = sns.pointplot(x="epochs", y="validation_accuracy", data=df, fit_reg=False, color='green')



predicted = model.predict([X_test, X_test, X_test])
predicted = np.argmax(predicted, axis=1)


cm = confusion_matrix(y_test, predicted)


plt.figure(figsize = (5.8, 4))
plt.title('Neural Network \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, predicted)))
sns.heatmap(cm, annot=True)
plt.show()

print(accuracy_score(y_test, predicted))

print(predicted)




