from abc import ABC, abstractmethod
from transformers import BertTokenizer

class ITokenizer(ABC):
    """
        Tokenizer interface
    """
    @abstractmethod
    def __init__(self, Vocabulary: str):
        self.Tokenizer = BertTokenizer
        pass

    @abstractmethod
    def TokenizeData(self, Sentences: list, Labels: list):
        pass

    @abstractmethod
    def SetTokenEmbedding(self, TokenizedTexts: list, TokenzedLabels: list):
        pass


