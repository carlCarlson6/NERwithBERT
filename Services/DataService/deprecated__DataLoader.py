import context
from common.SentenceGetter import SentenceGetter
from common.utils.TensorUtils import TensorUtils
import pandas as pd
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler
from torch.utils.data import DataLoader as TorchDataLoader
import torch


class DataLoader():
    def __init__(self, file, separator=','):
        self.dataFrame = self.LoadCsv(dataPath=file, csvSeparator=separator)
        self.torchDataLoader = {}


    def LoadCsv(self, dataPath, csvSeparator, encoding='latin1'):
        df = pd.read_csv(dataPath+'.csv', sep=csvSeparator, encoding=encoding).fillna(method='ffill')
        #df = pd.read_csv(dataPath+'.csv', sep=csvSeparator, encoding=encoding)
        return df

    def GetDocIds(self):
        sentenceGetter = SentenceGetter(self.dataFrame)
        sentences = [[s[0] for s in sent] for sent in sentenceGetter.sentences]
        return sentences

    def GetSentences(self):
        sentenceGetter = SentenceGetter(self.dataFrame)
        sentences = [[s[1] for s in sent] for sent in sentenceGetter.sentences]
        return sentences

    def GetLabels(self):
        sentenceGetter = SentenceGetter(self.dataFrame)
        labels = [[s[2] for s in sent] for sent in sentenceGetter.sentences]
        return labels

    def GetTags(self):
        uniqueTags = list(set(self.dataFrame['Tag'].values))
        uniqueTags.append('X')
        uniqueTags.append('[CLS]')
        uniqueTags.append('[SEP]')
        
        tagsValues = set(uniqueTags)
        return tagsValues

    def GetNumLabels(self):
        return len(self.GetTags())

    def PutDataIntoTorch(self, set, inputs, masks, tags, batchSize):
        inputsTensor, masksTensor, tagsTensors = TensorUtils.InputsIntoTensor(inputs=[inputs, masks, tags])
        inputsLongTensor, masksLongTensor, tagsLongTensor = TensorUtils.TensorsToLongTensors([inputsTensor, masksTensor, tagsTensors])

        data = TensorDataset(inputsLongTensor, masksLongTensor, tagsLongTensor)
        sampler = RandomSampler(data)
        self.torchDataLoader[set] = TorchDataLoader(data, sampler=sampler, batch_size=batchSize, drop_last=True)
        
        return len(inputs)