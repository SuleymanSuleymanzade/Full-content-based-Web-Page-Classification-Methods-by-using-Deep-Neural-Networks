import numpy as np
import pandas as pd
import csv 
import os
class Combiner:
	def __init__(self, datas_folder, text_classifier):
		self.image_params = []
		self.text_params = []
		self.global_image_weight = 1
		self.global_text_weight = 1
		self.datas_folder = datas_folder
		self.text_classifier = text_classifier 

	def softmax(self, arr):
		return np.exp(arr) / np.sum(np.exp(arr))

	def push_to_image_stack(self, image_classes, image_weight = 1, image_caption = 'Empty', image_classes_by_caption = None, image_prediction = "?"):
		self.image_params.append((image_classes, image_weight, image_caption, image_classes_by_caption, image_prediction))
	def push_to_text_stack(self, text_classes, text_weight = 1, text_prediction = "?"):
		self.text_params.append((text_classes, text_weight, text_prediction))

	def set_global_text_weight(self, text_weight):
		self.global_text_weight = text_weight

	def set_global_image_weight(self, image_weight):
		self.global_image_weight = image_weight

	def get_summary(self, status = False, report = False):
		total_summary_text = np.zeros((1, 5), dtype=float)
		weighted_summary_text = np.zeros((1, 5), dtype=float)
		normilized_summary_text = np.zeros((1, 5), dtype=float)

		total_summary_image = np.zeros((1, 5), dtype=float)
		weighted_summary_image = np.zeros((1, 5), dtype=float)
		normilized_summary_image = np.zeros((1, 5), dtype=float)
		# text part computation
		for text_p in self.text_params:
			total_summary_text += text_p[0] * text_p[1]
			weighted_summary_text += text_p[0] * text_p[1]
			normilized_summary_text += text_p[0] * text_p[1]
		weighted_summary_text *= self.global_text_weight
		normilized_summary_text = self.softmax(normilized_summary_text)
		normilized_summary_text *= self.global_text_weight
		# image part computation
		for image_p in self.image_params:
			total_summary_image += image_p[0] * image_p[1]
			weighted_summary_image += image_p[0] * image_p[1]
			normilized_summary_image += image_p[0] * image_p[1]
		weighted_summary_image *= self.global_image_weight
		normilized_summary_image = self.softmax(normilized_summary_image)
		normilized_summary_image *= self.global_image_weight

		total_summary = total_summary_image + total_summary_text
		weighted_summary = weighted_summary_image + weighted_summary_text
		normilized_summary = normilized_summary_image + normilized_summary_text
		if report:
			print('Report:')
			print(f"total summary:{total_summary}")
			print(f"weighted summary: {weighted_summary}")
			print(f"normilized_summary:{normilized_summary}")
		return {"total_summary":total_summary,"weighted_summary": weighted_summary, "normilized_summary":normilized_summary}

	def get_data_report(self, mode = ['image', 'text']):
		image_report = None
		text_report = None

		if 'image' in mode:			
			image_class_dict = {
				'ImageClasses':[],
				'ImageWeights':[],
				'ImageCaptions':[],
				'ImageClassesByCaptions':[],
				'ImagePredictions':[],
			}			
			for image_param in self.image_params:
				image_class_dict['ImageClasses'].append(image_param[0])
				image_class_dict['ImageWeights'].append(image_param[1])
				image_class_dict['ImageCaptions'].append(image_param[2])
				image_class_dict['ImageClassesByCaptions'].append(image_param[3])
				image_class_dict['ImagePredictions'].append(image_param[4])

			image_report = pd.DataFrame(image_class_dict)

		if 'text' in mode:		
			text_class_dict = {
				'TextID': [],
				'TextClasses':[],
				'TextWeights':[],
				'TextPredictions':[],
			}
			for text_param in self.text_params:
				text_class_dict['TextClasses'].append(text_param[0])
				text_class_dict['TextWeights'].append(text_param[1])
				text_class_dict['TextPredictions'].append(text_param[2])
				text_class_dict['TextID'].append(text_counter)
				text_counter += 1
			text_report = pd.DataFrame(text_class_dict)
		return (text_report, image_report)

	def process_text_data(self, show_status = False):

		'''text classifier must be trained first'''
		text_captions_file = os.path.join(self.datas_folder,  'text_captions.csv')
		images_captions_file = os.path.join(self.datas_folder, 'image_captions.csv')

		if os.path.exists(text_captions_file):
			with open(text_captions_file, mode = 'r') as file:
				caption_reader = csv.reader(file, delimiter = ';')
				for text in caption_reader:
					if text != []:
						prediction = self.text_classifier.predict_text(text[0])
						#print(prediction)
					self.push_to_text_stack(prediction[1], 1, text_prediction = prediction[0])
			if show_status:
				print('text data has been compressed..')

		if os.path.exists(images_captions_file):
			with open(images_captions_file, mode = "r") as file:
				caption_reader = csv.reader(file, delimiter = ';')
				for text in caption_reader:
					if text != []:
						prediction = self.text_classifier.predict_text(text[0])
						#print(prediction)
					self.push_to_image_stack(prediction[1], 1, image_prediction = prediction[0])
				if show_status:
					print('image data has been compressed..')