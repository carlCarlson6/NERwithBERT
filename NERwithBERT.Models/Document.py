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

    @staticmethod
    def GetDocuments(docIds, texts, labelsList, tokenizedTexts, tokenizedLabelsList, inputsList, tagsList, masksList):
        documents = []

        for docId, text, labels, tokenizedText, tokenizedLabels, inputs, tags, masks in zip(docIds, texts, labelsList, tokenizedTexts, tokenizedLabelsList, inputsList, tagsList, masksList):
            document = Document(docId, text, labels, tokenizedText, tokenizedLabels, inputs, tags, masks)
            documents.append(document)

        return documents

    @staticmethod
    def SetPredictions(documents, predictions):
        for document, prediction in zip(documents, predictions):
            document.Predictions = prediction
            document.TokenizedPredictedTags = [pred[0] for pred in prediction]
            document.SetEntities(prediction)

        return documents

    def SetEntities(self, predictions):
        predictedTags = [prediction[0] for prediction in predictions][1:-1]
        predictedProbs = [prediction[1] for prediction in predictions][1:-1]

        entityData = []

        lastTagChecked = None
        for tokenizedWord, predictedTag, predictedProb in zip(self.TokenizedText, predictedTags, predictedProbs):

            if predictedTag[0:2] == 'B-':
                if entityData == []:
                    entityData.append((tokenizedWord, predictedTag, predictedProb))
                
                else:
                    entity = Entity(text=[data[0] for data in entityData], tags=[data[1] for data in entityData], probabilities=[data[2] for data in entityData])
                    self.Entities.append(entity)

                    entityData = []
                    entityData.append((tokenizedWord, predictedTag, predictedProb))

            else:
                if entityData == []:
                    pass
                
                else:
                    if predictedTag[0:2] == 'I-':
                        entityData.append((tokenizedWord, predictedTag, predictedProb))
                    
                    else:
                        if predictedTag == 'O':
                            entity = Entity(text=[data[0] for data in entityData], tags=[data[1] for data in entityData], probabilities=[data[2] for data in entityData])
                            self.Entities.append(entity)

                            entityData = []
            
                        if predictedTag == 'X':
                            entityData.append((tokenizedWord, predictedTag, predictedProb))