#!/usr/bin/env python
#encoding="utf-8"

import os
import re
import math
import time
import sys
import nltk
import random
import numpy as np
from itertools import islice
from nltk.corpus import stopwords

reload(sys)
sys.setdefaultencoding("utf-8")

def getSet(filePath):
	with open(filePath, 'r') as f :
		dataSet = {line.strip() for line in f}
	return dataSet

def getw2vmodel(filePath):
	word2vecModel = dict()

	with open(filePath, 'r') as f:
		for line in f:
			line = line.strip().split(',')
			if len(line) != 201:
				continue
			word = line[0]
			vector = map(float, line[1:])
			word2vecModel[word] = vector

	return word2vecModel

def dataProcess(filePathCom, filePathDes, record, shuffled = True):
	postive = []
	negative = []

	with open(filePathDes, 'r') as f :
		for line in f :
			line = line.strip()
			negative.append(line)

	with open(filePathCom, 'r') as f :
		for line in f :
			line = line.strip()
			postive.append(line)

	# if shuffled:
	# 	random.shuffle(negative)
	# 	random.shuffle(postive)

	x_pos_train = postive[:record]
	x_pos_dev = postive[record:]

	x_neg_train = negative[:record]
	x_neg_dev = negative[record:]

	return x_pos_train, x_pos_dev, x_neg_train, x_neg_dev

def getpostAndnegative(x_pos_train, x_neg_train, stopwordsPath, word2vecModel):
	stopwords = getSet(stopwordsPath)

	postiveword = dict()
	negativeword = dict()

	for sentence in x_pos_train:
		# fileds = nltk.word_tokenize(sentence)
		fileds = sentence.split(' ')
		words = {word for word in fileds if word not in stopwords and not re.match(r'.*(\d)+.*', word)}
		for word in words :
			if word2vecModel.has_key(word) :
				postiveword[word] = postiveword.get(word, 0) + 1

	for sentence in x_neg_train:
		# fileds = nltk.word_tokenize(sentence)
		fileds = sentence.split(' ')
		words = {word for word in fileds if word not in stopwords and not re.match(r'.*(\d)+.*', word)}
		for word in words :
			if word2vecModel.has_key(word):
				negativeword[word] = negativeword.get(word, 0) + 1

	return postiveword, negativeword

def CalInforGain(postiveNum, negativeNum, totalNum):
	postiveFloat = postiveNum * 1.0 / totalNum
	negativeFloat = negativeNum * 1.0 / totalNum

	if postiveFloat != 0:
		postiveValue = (- 1) * math.log(postiveFloat) * postiveFloat 
	if negativeFloat != 0:
		negativeValue = (- 1) * math.log(negativeFloat) * negativeFloat    # we don't user the calculation of log()/log(2), we use the e instead
	return negativeValue + postiveValue

def CalMutualInfor(postiveNum, negativeNum, totalNum, postiveLen, negativeLen) :
	postMIScore = 10000 * postiveNum * 1.0 / (totalNum * postiveLen)
	negaMIScore = 10000 * negativeNum * 1.0 / (totalNum * negativeLen)
	return postMIScore, negaMIScore

def getPMIandIG(postiveword, negativeword):
	wordInforGain = dict()            # get the information gain of the words
	wordMutualInfor = dict()                   # get the result of the mutual information
	Vocabulay = set(postiveword.keys()) & set(negativeword.keys())

	postiveLen = len(postiveword)   # calculate length of each dict 
	negativeLen = len(negativeword)

	for word in Vocabulay:
		postiveNum = postiveword.get(word, 0)
		negativeNum = negativeword.get(word, 0)
		totalNum = postiveNum + negativeNum 
		if totalNum < 3:
			continue

		wordIG = CalInforGain(postiveNum, negativeNum, totalNum)
		postMIScore, negaMIScore = CalMutualInfor(postiveNum, negativeNum, totalNum, postiveLen, negativeLen)

		wordMutualInfor[word] = postMIScore / negaMIScore #change the value
		wordInforGain[word] = wordIG

	return wordInforGain, wordMutualInfor

def feature(filePathCom, filePathDes, stopwordsPath, word2vecModel, record):
	#dataprocess and get the result
	x_pos_train, x_pos_dev, x_neg_train, x_neg_dev = dataProcess(filePathCom, filePathDes, record)
	postiveword, negativeword = getpostAndnegative(x_pos_train, x_neg_train, stopwordsPath, word2vecModel)
	wordInforGain, wordMutualInfor = getPMIandIG(postiveword, negativeword)
	importantwords = wordInforGain
	return x_pos_train, x_pos_dev, x_neg_train, x_neg_dev, importantwords, wordMutualInfor

def postagToList(word):
	temp = [0.0] * 4
	if re.match(r"NN*", word):
		temp[0] = 1
		return temp
	elif re.match(r'JJ*', word):
		temp[1] = 1
		return temp
	elif re.match(r'VB*', word):
		temp[2] = 1
		return temp
	elif re.match(r'RB*', word):
		temp[3] = 1
		return temp
	else:
		return temp

def getvector(words , word2vecModel, importantwords, wordMutualInfor, sequence_length, dimensional):
	result = list()
	count = 0
	for item in words:
		if len(item) != 2:
			print "error"
			continue
		word = item[0]
		pos_tag_vec = postagToList(item[1])
		temp = list()
		temp1 = word2vecModel.get(word, [0.0] * 200)
		temp2 = importantwords.get(word, 0.0)
		temp3 = wordMutualInfor.get(word, 0.0)
		
		temp.extend(temp1)
		# temp.append(temp2)
		# temp.append(temp3)
		temp.extend(pos_tag_vec)
		 
		if sum(temp) != 0:
			count += 1
			result.extend(temp)

	if count >= sequence_length:
		return result[:sequence_length * dimensional]
	else:
		newTemp = [0.0] * dimensional * (sequence_length - count)
		newTemp.extend(result)
		return newTemp

def getdata(x_pos_train, x_pos_dev, x_neg_train, x_neg_dev, stopwordsPath, word2vecModel, importantwords, wordMutualInfor, sequence_length, dimensional):
	stopwords = getSet(stopwordsPath)
	x_train = []
	y_train = []

	for sentence in x_pos_train:
		line = sentence.strip().replace('-', '')
		fileds = line.split(' ')
		# line = sentence.strip()
		# fileds = nltk.word_tokenize(line)
		wordPostag = nltk.pos_tag(fileds)
		words = {word for word in wordPostag if word[0] not in stopwords and not re.match(r'.*(\d)+.*', word[0])}
		
		# words = [word for word in fileds if word not in stopwords and not re.match(r'.*(\d)+.*', word)]
		vector = getvector(words , word2vecModel, importantwords, wordMutualInfor, sequence_length, dimensional)
		x_train.append(vector)
		y_train.append([0.0, 1.0])

	for sentence in x_neg_train:
		line = sentence.strip().replace('-', '')
		fileds = line.split(' ')
		# line = sentence.strip()
		# fileds = nltk.word_tokenize(line)
		wordPostag = nltk.pos_tag(fileds)
		words = {word for word in wordPostag if word[0] not in stopwords and not re.match(r'.*(\d)+.*', word[0])}
		
		# words = [word for word in fileds if word not in stopwords and not re.match(r'.*(\d)+.*', word)]
		vector = getvector(words , word2vecModel, importantwords, wordMutualInfor, sequence_length, dimensional)
		x_train.append(vector)
		y_train.append([1.0, 0.0])

	x_dev = []
	y_dev = []

	for sentence in x_pos_dev:
		line = sentence.strip().replace('-', '')
		fileds = line.split(' ')

		words = {word for word in fileds if word not in stopwords and not re.match(r'.*(\d)+.*', word)}
		vector = getvector(words , word2vecModel, importantwords, wordMutualInfor, sequence_length, dimensional)
		x_dev.append(vector)
		y_dev.append([0.0, 1.0])

	for sentence in x_neg_dev:
		line = sentence.strip().replace('-', '')
		fileds = line.split(' ')

		words = {word for word in fileds if word not in stopwords and not re.match(r'.*(\d)+.*', word)}
		vector = getvector(words , word2vecModel, importantwords, wordMutualInfor, sequence_length, dimensional)
		x_dev.append(vector)
		y_dev.append([1.0, 0.0])

	return x_train, y_train, x_dev, y_dev

def load_train_label(sequence_length, dimensional, record = 4500):
	#filePath 
	configPath = "../config"
	dataSetPath = "../dataSet"
	filePathCom = os.path.join(dataSetPath, "quote.tok.gt9.5000")
	filePathDes = os.path.join(dataSetPath, "plot.tok.gt9.5000")
	stopwordsPath = os.path.join(configPath, "english")

	w2vPath = os.path.join(dataSetPath, "wikibiake.bin")

	word2vecModel = getw2vmodel(w2vPath)
	print "load word2vec model finisded ... "

	x_pos_train, x_pos_dev, x_neg_train, x_neg_dev, importantwords, wordMutualInfor = feature(filePathCom, filePathDes, stopwordsPath, word2vecModel, record)
	x_train, y_train, x_dev, y_dev = getdata(x_pos_train, x_pos_dev, x_neg_train, x_neg_dev, stopwordsPath, word2vecModel, importantwords, wordMutualInfor, sequence_length, dimensional)

	# X = []
	# Y = []
	# with open("CNN_inputformat_training", 'r') as f:
	# 	for line in f:
	# 		line = line.strip().split("\t")
	# 		if len(line) != 5052:
	# 			continue
	# 		else:
	# 			fileds = map(float, line)
	# 			X.append(fileds[:-2])
	# 			Y.append(fileds[-2:])

	# x_train = X[:9000]
	# y_train = Y[:9000]
	# x_dev = X[9000:]
	# y_dev = Y[9000:]

	return x_train, y_train, x_dev, y_dev

def batch_iter(data, shuffled = True):
	"""
	# we mainly generate a batch of data for training
	"""
	data_size = len(data)
	data_seq = range(data_size)
	if shuffled:
		random.shuffle(data_seq)
	for index in data_seq:
		yield data[index]

def batch_iter2(data, batch_size, num_epochs, shuffle = True):
	"""
	# we mainly generate a batch of data for training
	"""
	begin = time.clock()

	data = np.array(data)
	data_size = data.shape[0]
	num_batchs_each_epoch = int(data_size / batch_size) + 1

	for epochs in range(num_epochs) :
		if shuffle :
			shuffle_indices = np.random.permutation(data_size)
			shuffle_data = data[shuffle_indices]
		else :
			shuffle_data = data
		for num_batch in range(num_batchs_each_epoch) :
			start_index = num_batch * batch_size
			end_index = min((num_batch + 1) * batch_size, data_size)

			yield shuffle_data[start_index : end_index]
	end = time.clock()
	print "generate the batch of data for training end : {:f}".format( end - begin )

def main() :
	load_train_label(25, 202)

if __name__ == '__main__':
	main()