class SegmentEmbedder:
    def SetSegmentEmbedding(self, InputIds):
        segmentIds = [[0] * len(inputId) for inputId in InputIds]
        return segmentIds