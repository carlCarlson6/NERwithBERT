from Services.DataService.IDataService import IDataService
from Services.DataService.TorchDataService import TorchDataService
import common
import pandas as pd
from torch.utils.data import DataLoader as TorchDataLoader
import torch


class DataService(IDataService):
    def __init__(self):
        super().__init__()
        try:
            self.DataFrame = self.LoadCsv(DataPath='', CsvSeparator=',')
            self.SentenceGetter = SentenceGetter(self.DataFrame)
            self.torchDataLoader: dict

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
        uniqueTags = common.TagGetter().GetUniqueTagValues(self.DataFrame)      
        tagsValues = set(uniqueTags)
        return tagsValues

    def GetNumLabels(self):
        return len(self.GetTags())

    def PutDataIntoTorch(self, setName, inputs, masks, tags, batchSize):
        self.torchDataLoader[setName] = TorchDataService().PutDataIntoTorch(inputs, masks, tags, batchSize)
        return self.torchDataLoader[setName]