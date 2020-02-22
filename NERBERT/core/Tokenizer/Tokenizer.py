from NERBERT.core.Tokenizer.ITokenizer import *

class Tokenizer(ITokenizer):
    """
        Implementation of ITokenizer
    """
    def __init__(self, Vocabulary):
        super().__init__(Vocabulary)
        self.Tokenizer = BertTokenizer.from_pretrained(Vocabulary)
        
    def TokenizeData(self, Sentences, Labels):
        dataTokenizer = DataTokenizer()
        tokenizedText, tokenizedLabels = DataTokenizer.TokenizeData(self.Tokenizer, Sentences, Labels)
        return tokenizedText, tokenizedLabels

    def SetTokenEmbedding(self, TokenizedTexts, TokenzedLabels, MaxLength=512):
        if MaxLength > 512:
            raise Exception('max length con not be bigger than 512')
        
        tokenEmbedder = TokenEmbedder()
        inputIds, tags = tokenEmbedder.SetTokenEmbedding(self.Tokenizer, TokenizedTexts, TokenzedLabels)
        return inputIds, tags

    def SetMaskWordEmbedding(self, InputIds):
        maskWordEmbedder = MaskWordEmbedder()
        attentionMasks = maskWordEmbedder.SetMaskWordEmbedding(InputIds)
        return attentionMasks

    def SetSegmentEmbedding(self, InputIds):
        segmentEmbedder = SegmentEmbedder()
        segmentIds = segmentEmbedder.SetSegmentEmbedding(InputIds)
        return segmentIds