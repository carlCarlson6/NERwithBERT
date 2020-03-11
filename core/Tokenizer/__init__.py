from core.Tokenizer.ITokenizer import *

class Tokenizer(ITokenizer):
    """
        Implementation of ITokenizer
    """
    def __init__(self, TagToIndex: dict, Vocabulary='bert-base-multilingual-cased'):
        super().__init__()
        self.Tokenizer = BertTokenizer.from_pretrained(Vocabulary)
        self.TagToIndex = TagToIndex
        
    def TokenizeData(self, Sentences, Labels):
        dataTokenizer = DataTokenizer()
        tokenizedText, tokenizedLabels = dataTokenizer.TokenizeData(self.Tokenizer, Sentences, Labels)
        return tokenizedText, tokenizedLabels

    def SetTokenEmbedding(self, TokenizedTexts, TokenzedLabels, MaxLength=512):
        if MaxLength > 512:
            raise Exception('max length con not be bigger than 512')    
        tokenEmbedder = TokenEmbedder()
        inputIds, tags = tokenEmbedder.SetTokenEmbedding(self.Tokenizer, self.TagToIndex, TokenizedTexts, TokenzedLabels, MaxLength)
        return inputIds, tags

    def SetMaskWordEmbedding(self, InputIds):
        maskWordEmbedder = MaskWordEmbedder()
        attentionMasks = maskWordEmbedder.SetMaskWordEmbedding(InputIds)
        return attentionMasks

    # TODO quitar ¿?
    def SetSegmentEmbedding(self, InputIds):
        segmentEmbedder = SegmentEmbedder()
        segmentIds = segmentEmbedder.SetSegmentEmbedding(InputIds)
        return segmentIds