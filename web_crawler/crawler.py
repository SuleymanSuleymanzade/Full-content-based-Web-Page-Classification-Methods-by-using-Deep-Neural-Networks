from bs4 import BeautifulSoup 
from bs4.element import Comment
import requests
import urllib.request
import ssl
import os 
from datetime import datetime 
import re
import csv 

class MetaSingleton(type):
	_instaces = {}
	def __call__(cls, *args, **kwargs):
		if cls not in cls._instaces:
			cls._instaces[cls] = super(MetaSingleton, cls).__call__(*args, **kwargs)
		return cls._instaces[cls]

class TextProcessing:
	@staticmethod 
	def remove_html_tags(text):
		"""Remove html tags from a string"""
		clean = re.compile('<.*?>')
		return re.sub(clean, '', text)
	@staticmethod
	def remove_tags2(text):
		return re.sub(r'<.*?>','',text)

class WebCrawler:

	__img_res = {".jpg",".png",".gif"}
	def __init__(self, seed = None, strategy = "dfs", deep = 128):
		self.seed = seed # start point
		self.deep = deep # for dfs - deep point, for bfs layers
		self.strategy = strategy

	def add_image_resolution(self, res):
		self.__img_res.add(res)

	def remove_image_resolution(self, res):
		self.__img_res.remove(res)

	def tag_visible(self, element):
		if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
			return False
		if isinstance(element, Comment):
			return False
		if re.match(r"[\n]+",str(element)):
			return False
		return True
	
	def text_from_html(self, url, file_name, status = True, min_words = 4, max_words = 12, max_sent = 40, save_to_csv = True):
		text_bunch = []
		body = urllib.request.urlopen(url,context=ssl._create_unverified_context()).read()
		soup = BeautifulSoup(body ,"lxml")
		texts = soup.findAll(text=True)
		visible_texts = filter(self.tag_visible, texts)  
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
	 
	def get_image_data(self, url, folder_name, status=False, max_images = 128):
		path  = folder_name
		respond = requests.get(url).text
		soup = BeautifulSoup(respond, "lxml")
		images = [image for image in soup.find_all('img')]
		download_links = [link.get('src') for link in images if link.get('src')[-4:] in self.__img_res]
		image_count = 0

		for image_link in download_links:
			image_file = requests.get(image_link)
			image_res = image_link[-4:]
			filename =  str(image_count)+"_"+datetime.now().strftime('%H_%M_%S')+image_res[-4:]
			image_count += 1
			with open(os.path.join(path, filename), "wb") as file:
				if status:
					print(f'writing {filename} file..')
				file.write(image_file.content)
				
				if image_count >= max_images:
					if status:
						print(f'-- {image_count} files was added')
					return
		if status:
			print(f'-- {image_count} files was added')

class DataMiner:
	def __init__(self, directory, webcrawler, website):
		
		self.website = website
		self.directory = directory
		self.webcrawler = webcrawler 
		self.images_captions = []

		if not os.path.exists(os.path.join(self.directory, 'Data', 'Images')):
			os.makedirs(os.path.join((self.directory), 'Data', 'ImageFiles'))

		if not os.path.exists(os.path.join(self.directory, 'Data','TextFiles')):
			os.makedirs(os.path.join((self.directory), 'Data','TextFiles'))		
		
		self.image_files_directory = os.path.join(self.directory, 'Data', 'ImageFiles')
		self.image_captions_file =  os.path.join(self.directory, 'Data', 'TextFiles', 'image_captions.csv')
		self.text_caption_file = os.path.join(self.directory, 'Data', 'TextFiles', 'text_captions.csv')


	def set_website(self, website):
		self.website = website
	def delete_image_captions(self):
		self.images_captions = []

	def gather_images(self, show_status = True, mx_images = 128):
		self.webcrawler.get_image_data(self.website, self.image_files_directory, status = show_status, max_images = mx_images)

	def gather_text(self, show_status = True, mx_phrazes = 128, mn_words = 4, mx_words = 12):
		self.webcrawler.text_from_html(self.website, self.text_caption_file, status = show_status, min_words = mn_words, max_words = mx_words, max_sent = mx_phrazes)

	def generate_images_captions(self, image_analyzer, status = False, save_to_csv = True):
		for image_file in os.listdir(self.image_files_directory):
			file_path = os.path.join(self.image_files_directory, image_file)

			if status:
				print(f'analysing {image_file} file...')
			caption = image_analyzer.get_caption(file_path)
			self.images_captions.append(caption)
			if save_to_csv:
				with open(self.image_captions_file, mode = 'a') as f:
					caption_writer = csv.writer(f, delimiter = ';')
					caption_writer.writerow([caption])
				if status:
					print('saved to csv.')
			if status:
				print(caption)
		if status:
			print('image captioning is done')
		return self.images_captions

