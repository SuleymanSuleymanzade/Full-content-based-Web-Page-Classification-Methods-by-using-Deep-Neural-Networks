from web_crawler import WebCrawler, DataMiner
from caption_generator import ImageAnalyzer
from text_classifier_dense import *
import numpy as np
import pandas as pd
from combiner import Combiner
import os
import csv
#wc = WebCrawler("www.cnn.com")
#wc.get_text_data("https://news.day.az/showbiz/1231769.html")
'''
ia = ImageAnalyzer()
image_cation = ia.get_caption("./caption_generator/examples/download.jpg")
print(image_cation)

dense_classifier_bbc = TextClassifierDenseBBC()
dense_classifier_bbc.build_model()
dense_classifier_bbc.train_model(show_status = True)

ans  = dense_classifier_bbc.predict_text(image_cation)

print(ans)
'''

'''
t_global = 1.113

t1_class = np.array([[0.136998  , 0.40185034, 0.14847934, 0.19308965, 0.11958268]])
t1_weights = 4

t2_class = np.array([[0.234998  , 0.10185034, 0.84847934, 0.29308965, 0.15958268]])
t2_weights =  10

t3_class = np.array([[0.136998  , 0.40185034, 0.14847934, 0.19308965, 0.11958268]])
t3_weights =  5


p_global = 1.112
p1_class = np.array([[0.136998  , 0.40185034, 0.14847934, 0.19308965, 0.11958268]])
p1_weights = 5

p2_class = np.array([[0.136998  , 0.40185034, 0.14847934, 0.19308965, 0.11958268]])
p2_weights = 2

p3_class = np.array([[0.136998  , 0.40185034, 0.14847934, 0.19308965, 0.11958268]])
p3_weights = 3

cm = Combiner()
cm.set_global_image_weight(0.49)
cm.set_global_text_weight(0.51)
cm.push_to_text_stack(t1_class, t1_weights)
cm.push_to_text_stack(t2_class, t2_weights)
cm.push_to_text_stack(t3_class, t3_weights)
cm.push_to_image_stack(p1_class, p1_weights)
cm.push_to_image_stack(p2_class, p2_weights)
ans = cm.get_summary(report = True)

'''


'''
web_crawler = WebCrawler()
image_analyzer = ImageAnalyzer()
cur_file = os.getcwd()
dm = DataMiner(cur_file, web_crawler, 'http://www.bbc.com/travel?referer=https%3A%2F%2Fwww.bbc.com%2Fnews%2Flive%2Fworld-53039952')
dm.gather_text(show_status = True, mx_phrazes = 40)
dm.gather_images(web_crawler, mx_images = 10)
dm.generate_images_captions(image_analyzer)
'''





text_classifier = TextClassifierDenseBBC()
text_classifier.build_model()
text_classifier.train_model(show_status = True)

combiner = Combiner(os.path.join(os.getcwd(), 'Data', 'TextFiles' ), text_classifier)
combiner.process_text_data()

text_report, image_report = combiner.get_data_report()



'''

combiner = Combiner()
# build and train text classifier 
text_classifier = TextClassifierDenseBBC()
text_classifier.build_model()
text_classifier.train_model(show_status = True)
# train phaze is finished



combiner = Combiner()
combiner.set_global_text_weight = 0.4
combiner.set_global_image_weight = 0.8



dm = DataMiner(os.getcwd(), 'https://www.motor1.com/')
#dm.gather_images(web_crawler, max_images = 10)
#dm.generate_images_captions(image_analyzer, status = True, save_to_csv = True)
dm.process_text_data(text_classifier, combiner)
'''




'''

from bs4 import BeautifulSoup
from bs4.element import Comment
import urllib.request
import re
import ssl

def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    if re.match(r"[\n]+",str(element)): return False
    return True

def text_from_html(url, file_name, status = True, min_words = 4, max_words = 12, max_sent = 40, save_to_csv = True):
	text_bunch = []

	body = urllib.request.urlopen(url,context=ssl._create_unverified_context()).read()
	soup = BeautifulSoup(body ,"lxml")
	texts = soup.findAll(text=True)
	visible_texts = filter(tag_visible, texts)  
	text = u".".join(t.strip() for t in visible_texts)
	text = text.lstrip().rstrip()
	text = text.split('.')
	for sen in text:
		sen = sen.lstrip().rstrip()
		len_sen = len(sen.split())
		if len_sen >= min_words and len_sen <= max_words:
			text_bunch.append(sen)
	if len(text_bunch) > max_sent:
		text_bunch = text_bunch[:max_sent]
	
	if save_to_csv:
		with open(file_name, mode = 'a') as f:
			text_writer = csv.writer(f, delimiter = ';')
			for phraze in text_bunch:
				text_writer.writerow([phraze])
	if status:
		print('saved to csv\n',text_bunch[:10])
	 

url = 'http://www.nytimes.com/2009/12/21/us/21storm.html'
text_from_html(url, 'm_data.csv', status = True)
'''