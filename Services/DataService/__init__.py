from Services.DataService.IDataService import IDataService
from Services.DataService.TorchDataService import TorchDataService
import common
import pandas as pd
from torch.utils.data import DataLoader as TorchDataLoader
import torch


class DataService(IDataService):
    """
        IDataService implementation
    """
    def __init__(self, DataPath='C:\\dev\\NERwithBERT\\DATA\\bio.data\\bio.dataset.v1\\bio_dataset_splitted'):
        super().__init__()
        self.DataFrame: pd.DataFrame
        self.SentenceGetter: common.SentenceGetter
        self.torchDataLoader: dict

        try:
            self.DataFrame = self.LoadCsv(DataPath=DataPath, CsvSeparator=';', Encoding='latin')
            self.SentenceGetter = common.SentenceGetter(self.DataFrame)
            self.torchDataLoader: dict
        except:
            raise Exception('could not load the data')
        
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