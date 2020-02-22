from Entity import Entity

class Document:
    """
        Entity Data Model
    """
    self.DocId = str
    self.Text = str
    self.Labels = str
    self.TokenizedText = None
    self.TokenizedLabels = None
    self.Inputs = None 
    self.Tags = None
    self.Masks = None
    self.TokenizedPredictedTags = None
    self.Predictions = None
    self.Entities = []

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
