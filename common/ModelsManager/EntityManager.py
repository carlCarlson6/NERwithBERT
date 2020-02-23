import numpy as np
import Models

class EntityManager:
    
    def SetText(self, entityData):
        text = [data[0] for data in entityData]
        completeText = ''
        for word in text:
            if '##' in word:
                completeText = completeText + word[2:]
            else:
                completeText = completeText + ' ' + word
        return completeText

    def SetTag(self, entityData):
        tags = [data[1] for data in entityData]
        initialTag = tags[0]
        tag = initialTag[2:]       
        return tag
    
    def SetProbability(self, entityData):
        probabilities = [data[2] for data in entityData]
        prob = np.mean(probabilities)
        return prob
        