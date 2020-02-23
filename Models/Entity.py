import numpy as np

class Entity:
    """
        Entity Data Model
    """
    def __init__(self, text, tags, probability):
        
        self.Text = text
        self.Tag = tags
        self.Probability = probability

        # TODO colocar esto en DocumentManager.SetEntities
        # self.Text = entityManager.SetText(text)
        # self.Tag = entityManager.SetTag(tags)
        # self.Probability = entityManager.SetProbability(probabilities)

