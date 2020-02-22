from Services.DataService.IDataService import IDataService
from common.SentenceGetter import SentenceGetter
from common.utils.TensorUtils import TensorUtils
import pandas as pd
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler
from torch.utils.data import DataLoader as TorchDataLoader
import torch


class DataService(IDataService):
    def __init__(self):
        super().__init__()
        try:
            self.DataFrame = self.LoadCsv(DataPath='', CsvSeparator=',')
            self.SentenceGetter = SentenceGetter(self.DataFrame)

    def LoadCsv(self, DataPath, CsvSeparator, Encoding):
        df = pd.read_csv(DataPath+'.csv', sep=CsvSeparator, encoding=Encoding).fillna(method='ffill')
        return df

    def GetDocIds(self):
        docIds = self.SentenceGetter.GetDocIds()
        return docIds

    def GetSentences(self):
        sentences = self.SentenceGetter.GetSentences()
        return sentences

    def GetLabels(self):
        labels = self.SentenceGetter.GetLabels()
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