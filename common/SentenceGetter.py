class SentenceGetter(object):   
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(index, word, tag) for index, word, tag in zip(s["Sentence"].values.tolist(), s["Word"].values.tolist(), s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence").apply(agg_func)
        self.sentences = [s for s in self.grouped]
    
    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

    def GetDocIds(self):
        ids = [sentence[0][0] for sentence in self.sentences]
        return ids

    def GetSentences(self):
        sentences = [[s[1] for s in sent] for sent in self.sentences]
        return sentences

    def GetLabels(self):
        labels = [[s[2] for s in sent] for sent in self.sentences]
        return labels