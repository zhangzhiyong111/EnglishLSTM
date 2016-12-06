#!/usr/bin/env python
#encoding="utf-8"

import os
import re
import math
import sys
import random
from itertools import islice
from nltk.corpus import stopwords

reload(sys)
sys.setdefaultencoding("utf-8")

def getSet(filePath) :
	with open(filePath, 'r') as f :
		dataSet = {line.strip() for line in f}
	return dataSet

def getw2vmodel(filePath) :
	word2vecModel = dict()

	input_file = open(filePath)
	for line in islice(input_file, 1, None) :
		line = line.strip().split(' ')
		word = line[0]
		vector = map(float, line[1:])
		if len(vector) != 200:
			continue
		word2vecModel[word] = vector

	return word2vecModel

def dataProcess(filePathCom, filePathDes, stopwordsPath, word2vecModel, record) :
	stopwords = getSet(stopwordsPath)
	postiveword = dict()
	negativeword = dict()

	i = 0
	with open(filePathDes, 'r') as f :
		for line in f :
			i += 1
			if i > record :
				break
			line = line.strip().replace('-', '')
			fileds = line.split(' ')
			words = {word for word in fileds if word not in stopwords and not re.match(r'.*(\d)+.*', word)}
			for word in words :
				if word2vecModel.has_key(word) :
					postiveword[word] = postiveword.get(word, 0) + 1

	i = 0
	with open(filePathCom, 'r') as f :
		for line in f :
			i += 1
			if i > record :
				break
			line = line.strip().replace('-', '')
			fileds = line.split(' ')
			words = {word for word in fileds if word not in stopwords and not re.match(r'.*(\d)+.*', word)}
			for word in words :
				if word2vecModel.has_key(word):
					negativeword[word] = negativeword.get(word, 0) + 1

	return postiveword, negativeword

def CalInforGain(postiveNum, negativeNum, totalNum) :
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

		wordMutualInfor[word] = [postMIScore, negaMIScore, postMIScore - negaMIScore]
		wordInforGain[word] = wordIG

	return wordInforGain, wordMutualInfor

def getImportantWords(wordInforGain, importantWordsThreshold) :
	wordMITemp = sorted(wordInforGain.items(), key = lambda x: x[1])
	words , _ = list(zip(* wordMITemp))
	importantwords = set(words[: importantWordsThreshold])
	return importantwords

def feature(filePathCom, filePathDes, stopwordsPath, word2vecModel, importantWordsThreshold, record):
	#dataprocess and get the result
	postiveword, negativeword = dataProcess(filePathCom, filePathDes, stopwordsPath, word2vecModel, record)
	wordInforGain, wordMutualInfor = getPMIandIG(postiveword, negativeword)

	#get the important words which used to as the feature
	importantwords = getImportantWords(wordInforGain, importantWordsThreshold)
	return importantwords, wordMutualInfor

def getvector(words , word2vecModel, importantwords, wordMutualInfor, sequence_length):
	result = list()
	count = 0
	for word in words:
		temp = list()
		temp1 = word2vecModel.get(word, [0.0] * 200)
		temp2 = [0.0]
		if word in importantwords:
			temp2 = [1.0]
		temp3 = wordMutualInfor.get(word, [0.0] * 3)
		
		temp.extend(temp1)
		temp.extend(temp2)
		temp.extend(temp3)

		if(sum(temp)) != 0:
			count += 1
			result.extend(temp)

	if count >= sequence_length:
		return result[:sequence_length]
	else:
		newTemp = [0.0] * 204 * (sequence_length - count)
		newTemp.extend(result)
		return newTemp

def getdata(filePathCom, filePathDes, stopwordsPath, word2vecModel, importantwords, wordMutualInfor, sequence_length):
	stopwords = getSet(stopwordsPath)
	X = []

	with open(filePathDes, 'r') as f :
		for line in f :
			line = line.strip().replace('-', '')
			fileds = line.split(' ')

			words = {word for word in fileds if word not in stopwords and not re.match(r'.*(\d)+.*', word)}
			vector = getvector(words , word2vecModel, importantwords, wordMutualInfor, sequence_length)
			X.append(vector)

	with open(filePathCom, 'r') as f :
		for line in f :
			line = line.strip().replace('-', '')
			fileds = line.split(' ')

			words = {word for word in fileds if word not in stopwords and not re.match(r'.*(\d)+.*', word)}
			vector = getvector(words , word2vecModel, importantwords, wordMutualInfor, sequence_length)
			X.append(vector)

	x1 = X[:5000]
	x2 = X[5000:]

	random.shuffle(x1)
	random.shuffle(x2)

	y1 = [[1.0, 0.0] for i in range(5000)]
	y2 = [[0.0, 1.0] for i in range(5000)]

	return x1, y1, x2, y2

def load_train_label(sequence_length, record = 4500):
	#filePath 
	configPath = "../config"
	dataSetPath = "../dataSet"
	filePathCom = os.path.join(dataSetPath, "quote.tok.gt9.5000")
	filePathDes = os.path.join(dataSetPath, "plot.tok.gt9.5000")
	stopwordsPath = os.path.join(configPath, "english")

	w2vPath = os.path.join(dataSetPath, "result.bin")

	word2vecModel = getw2vmodel(w2vPath)
	print "load word2vec model finisded ... "

	#parameters setting
	importantWordsThreshold = 50  #you want to get the important words
	importantwords, wordMutualInfor = feature(filePathCom, filePathDes, stopwordsPath, word2vecModel, importantWordsThreshold, record)
	x1, y1, x2, y2 = getdata(filePathCom, filePathDes, stopwordsPath, word2vecModel, importantwords, wordMutualInfor, sequence_length)

	return x1, y1, x2, y2

def batch_iter(data, shuffled = True) :
	"""
	# we mainly generate a batch of data for training
	"""
	data_size = len(data)
	data_seq = range(data_size)
	if shuffled:
		random.shuffle(data_seq)
	for index in data_seq:
		yield data[index]

def main() :
	load_train_label()

if __name__ == '__main__':
	main()