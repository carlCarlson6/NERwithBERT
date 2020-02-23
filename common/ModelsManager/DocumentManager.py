from common.ModelsManager.EntityManager import EntityManager
import Models

class DocumentManager:

    def GetDocuments(self, docIds, texts, labelsList, tokenizedTexts, tokenizedLabelsList, inputsList, tagsList, masksList):
        documents = []

        for docId, text, labels, tokenizedText, tokenizedLabels, inputs, tags, masks in zip(docIds, texts, labelsList, tokenizedTexts, tokenizedLabelsList, inputsList, tagsList, masksList):
            document = Models.Document(docId, text, labels, tokenizedText, tokenizedLabels, inputs, tags, masks)
            documents.append(document)

        return documents

    def SetPredictions(self, documents, predictions):
        for document, prediction in zip(documents, predictions):
            document.Predictions = prediction
            document.TokenizedPredictedTags = [pred[0] for pred in prediction]
            document.SetEntities(prediction)

        return documents

    # TODO mirar de refactorizar esta parte
    def SetEntities(self, predictions, tokenizedText):
        entities = []
        entityManager = EntityManager()
        entityData = []

        predictedTags = [prediction[0] for prediction in predictions][1:-1]
        predictedProbs = [prediction[1] for prediction in predictions][1:-1]

        for tokenizedWord, predictedTag, predictedProb in zip(tokenizedText, predictedTags, predictedProbs):
            if predictedTag[0:2] == 'B-':
                if entityData == []:
                    entityData.append((tokenizedWord, predictedTag, predictedProb))             
                else:                    
                    entity = Models.Entity(
                        text=entityManager.SetText(entityData), 
                        tag=entityManager.SetTag(entityData), 
                        probability=entityManager.SetProbability(entityData)
                    )
                    entities.append(entity)
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
                            entity = Models.Entity(
                                text=entityManager.SetText(entityData), 
                                tag=entityManager.SetTag(entityData), 
                                probability=entityManager.SetProbability(entityData)
                            )       
                            entities.append(entity)
                            entityData = []         
                        if predictedTag == 'X':
                            entityData.append((tokenizedWord, predictedTag, predictedProb))

        return entities