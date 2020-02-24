from abc import ABC, abstractmethod
import pandas as pd


class IDataService():
    """
        DataService interface
    """

    @abstractmethod
    def __init__(self):
        self.DataFrame: pd.DataFrame
        pass

    @abstractmethod
    def LoadCsv(self, DataPath, CsvSeparator, Encoding):
        pass

    @abstractmethod
    def GetDocIds(self):
        pass

    @abstractmethod
    def GetSentences(self):
        pass

    @abstractmethod
    def GetLabels(self):
        pass

    @abstractmethod
    def GetTags(self):
        pass

    @abstractmethod
    def GetNumLabels(self):
        pass

    @abstractmethod
    def PutDataIntoTorch(self, Set, Inputs, Masks, Tags, BatchSize):
        pass