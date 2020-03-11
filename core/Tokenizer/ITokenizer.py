from abc import ABC, abstractmethod
from transformers import BertTokenizer
from core.Tokenizer.DataTokenizer import DataTokenizer
from core.Tokenizer.TokenEmbedder import TokenEmbedder
from core.Tokenizer.MaskWordEmbedder import MaskWordEmbedder
from core.Tokenizer.SegmentEmbedder import SegmentEmbedder

class ITokenizer(ABC):
    """
        Tokenizer interface
    """
    @abstractmethod
    def __init__(self):
        self.Tokenizer: BertTokenizer
        pass

    @abstractmethod
    def TokenizeData(self, Sentences: list, Labels: list):
        pass

    @abstractmethod
    def SetTokenEmbedding(self, TokenizedTexts: list, TokenzedLabels: list, MaxLength: int):
        pass

    @abstractmethod
    def SetMaskWordEmbedding(self, InputIds: list):
        pass

    @abstractmethod
    def SetSegmentEmbedding(self, inputIds: list):
        pass