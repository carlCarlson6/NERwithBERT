from abc import ABC, abstractmethod
from transformers import BertTokenizer
from NERBERT.core.Tokenizer.DataTokenizer import DataTokenizer
from NERBERT.core.Tokenizer.TokenEmbedder import TokenEmbedder
from NERBERT.core.Tokenizer.MaskWordEmbedder import MaskWordEmbedder
from NERBERT.core.Tokenizer.SegmentEmbedder import SegmentEmbedder

class ITokenizer(ABC):
    """
        Tokenizer interface
    """
    @abstractmethod
    def __init__(self, Vocabulary: str):
        pass

    @abstractmethod
    def TokenizeData(self, Sentences: list, Labels: list):
        pass

    @abstractmethod
    def SetTokenEmbedding(self, TokenizedTexts: list, TokenzedLabels: list):
        pass

    @abstractmethod
    def SetMaskWordEmbedding(self, InputIds: list):
        pass

    @abstractmethod
    def SetSegmentEmbedding(self, inputIds: list):
        pass