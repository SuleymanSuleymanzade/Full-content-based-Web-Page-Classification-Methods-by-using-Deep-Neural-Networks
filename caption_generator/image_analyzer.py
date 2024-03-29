from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.applications.xception import Xception
from keras.models import load_model
from pickle import load
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os 
#import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # desable warning
TOKENIZER = os.path.join( os.path.dirname(__file__), "tokenizer.p")
MODEL = os.path.join(os.path.dirname(__file__), "models/model_9.h5")


class ImageAnalyzer():
    
    def __init__(self):
        pass 

    def extract_features(self, filename, model):
        try:
            image = Image.open(filename)
        except:
            print("ERROR: Couldn't open image! Make sure the image path and extension is correct")
        image = image.resize((299,299))
        image = np.array(image)
        # for images that has 4 channels, we convert them into 3 channels
        if image.shape[2] == 4: 
            image = image[..., :3]
        image = np.expand_dims(image, axis=0)
        image = image/127.5
        image = image - 1.0
        feature = model.predict(image)
        return feature

    def word_for_id(self, integer, tokenizer):
        for word, index in tokenizer.word_index.items():
            if index == integer:
                return word
        return None

    def generate_desc(self, model, tokenizer, photo, max_length):
        in_text = 'start'
        for i in range(max_length):
            sequence = tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence], maxlen=max_length)
            pred = model.predict([photo,sequence], verbose=0)
            pred = np.argmax(pred)
            word = self.word_for_id(pred, tokenizer)
            if word is None:
                break
            in_text += ' ' + word
            if word == 'end':
                break
        return in_text


    def get_caption(self, image_path, max_length = 32, status = False):
        tokenizer = load(open(TOKENIZER, "rb"))
        model = load_model(MODEL)
        xception_model = Xception(include_top = False, pooling = "avg")
        photo = self.extract_features(image_path, xception_model)
        img = Image.open(image_path)

        description = self.generate_desc(model, tokenizer, photo, max_length)
        description = " ".join(description.strip().split()[1:-1])
        if status:
            print(description)
            plt.imshow(img)
        return description


