from Models.Entity import Entity

class Document:
    """
        Entity Data Model
    """
    def __init__(self, docId=str, text=str, labels=str, tokenizedText=None, tokenizedLabels=None, inputs=None, tags=None, masks=None):
        self.DocId = docId
        self.Text = text
        self.Labels = labels

        self.TokenizedText = tokenizedText
        self.TokenizedLabels = tokenizedLabels

        self.Inputs = inputs 
        self.Tags = tags
        self.Masks = masks

        self.TokenizedPredictedTags = None

        self.Predictions = None
        self.Entities = []