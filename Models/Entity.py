import numpy as np

class Entity:
    """
        Entity Data Model
    """
    def __init__(self, text, tags, probabilities):
        self.Text = self.SetText(text)
        self.Tag = self.SetTag(tags)
        self.Probability = self.SetProbability(probabilities)

    def SetText(self, text):
        completeText = ''
        for word in text:
            if '##' in word:
                completeText = completeText + word[2:]
            else:
                completeText = completeText + ' ' + word

        return completeText

    def SetTag(self, tags):
        initialTag = tags[0]
        tag = initialTag[2:]
        
        return tag
    
    def SetProbability(self, probabilities):
        prob = np.mean(probabilities)
        return prob
