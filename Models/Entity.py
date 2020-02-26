import numpy as np

class Entity:
    """
        Entity Data Model
    """
    def __init__(self, text, tags, probability):
        
        self.Text = text
        self.Tag = tags
        self.Probability = probability