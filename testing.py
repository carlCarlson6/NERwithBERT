import Models
from Services import DataService
from core.Tokenizer import Tokenizer

directory= 'C:\\dev\\NERwithBERT\\DATA\\bio.data\\bio.dataset.v1\\bio_dataset_splitted'
fileName = 'bio_dataset_splitted'

dataService = DataService(directory)
docIds = dataService.GetDocIds()
sentences = dataService.GetSentences()
labels = dataService.GetLabels()
tag2index, index2tag = dataService.GetTagIndexMapping()

tokenizer = Tokenizer(TagToIndex=tag2index)
tokenizedText, tokenizedLabels = tokenizer.TokenizeData(sentences, labels)
inputIds, tags = tokenizer.SetTokenEmbedding(tokenizedText, tokenizedLabels)
segmentIds = tokenizer.SetSegmentEmbedding(inputIds)
attentionMasks = tokenizer.SetMaskWordEmbedding(inputIds)

k = 0