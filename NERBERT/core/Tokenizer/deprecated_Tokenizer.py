import torch
from keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer
import joblib

# TODO meter la funcionalidad de cada metodo en un clase
class Tokenizer():

    def __init__(self, vocabulary='bert-base-multilingual-cased', PathToIndex2Tag = None):
        self.tokenizer = BertTokenizer.from_pretrained(vocabulary)
        self.tag2index = {}
        self.index2tag = joblib.load(PathToIndex2Tag) if PathToIndex2Tag != None else {}
        self.tag2index = {self.index2tag[key]: key for key in self.index2tag.keys()} if PathToIndex2Tag != None else {}


    # TODO deberia ir en otra clase
    def SetTagIndexMapping(self, tags):
        self.tag2index = {tag: index for index, tag in enumerate(tags)}
        self.index2tag = {self.tag2index[key] : key for key in self.tag2index.keys()}
        return self.tag2index, self.index2tag


    def TokenizeData(self, sentences, labels):
        tokenizedTexts = []
        tokenizedLabels = []

        for wordList, label in (zip(sentences, labels)):
            tempLabel = []
            tempToken = []

            tempLabel.append('[CLS]')
            tempToken.append('[CLS]')

            for word, lbl in zip(wordList, label):
                tokenList = self.tokenizer.tokenize(word)
                
                for idx, token in enumerate(tokenList):
                    tempToken.append(token)
                    if idx == 0:
                        tempLabel.append(lbl)
                    else:
                        tempLabel.append('X')

            tempLabel.append('[SEP]')
            tempToken.append('[SEP]') 

            tokenizedTexts.append(tempToken)
            tokenizedLabels.append(tempLabel)

        return tokenizedTexts, tokenizedLabels
    
    def SetTokenEmbedding(self, tokenizedTexts, tokenizedLabels, maxLength=512):
        if maxLength > 512:
            raise Exception('max length con not be bigger than 512')

        inputIds = pad_sequences(
            [self.tokenizer.convert_tokens_to_ids(text) for text in tokenizedTexts],
            maxlen = maxLength,
            dtype = 'long',
            truncating = "post",
            padding = "post"
        )
        tags = pad_sequences(
            [[self.tag2index.get(lbl) for lbl in label] for label in tokenizedLabels],
            maxlen = maxLength,
            value = self.tag2index['O'],
            dtype = 'long',
            padding = 'post',
            truncating = 'post'
        )
        
        return inputIds, tags

    def SetMaskWordEmbedding(self, inputIds):
        attentionMasks = [[int(i>0) for i in ii] for ii in inputIds]
        return attentionMasks

    def SetSegmentEmbedding(self, inputIds):
        segmentIds = [[0] * len(inputId) for inputId in inputIds]
        return segmentIds