from abc import ABC, abstractmethod
from transformers import BertForTokenClassification as BertModel
from core.Tokenizer import ITokenizer
import torch


class IModel(ABC):
    @abstractmethod
    def __init__(self, ModelVersion: str):
        self.Bert = BertModel
        self.Tokenizer = ITokenizer

        self.NumLabels = int

        self.Device = torch.device
        self.NumberGPUS = int

        pass

    @abstractmethod
    def Predict(self):
        pass