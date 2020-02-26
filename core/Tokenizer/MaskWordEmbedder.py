class MaskWordEmbedder:
    def SetMaskWordEmbedding(self, InputIds):
        attentionMasks = [[int(i>0) for i in ii] for ii in InputIds]
        return attentionMasks