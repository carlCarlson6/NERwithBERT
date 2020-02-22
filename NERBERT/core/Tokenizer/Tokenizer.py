from NERBERT.core.Tokenizer.ITokenizer import *

class Tokenizer(ITokenizer):
    """
        Implementation of ITokenizer
    """
    def __init__(self, Vocabulary):
        super().__init__(Vocabulary)
        self.DataTokenizer = DataTokenizer()

    def TokenizeData(self, Sentences, Labels):

        self.DataTokenizer

        d = self.DataTokenizer()

        return super().TokenizeData(Sentences, Labels)