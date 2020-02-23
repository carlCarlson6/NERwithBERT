import context
import Services.DataService
from Services.DataLoader import DataService
from NERBERT.core.Tokenizer import Tokenizer

directory: 'C:\\dev\\AMLPOC\\AMLPOC.BERT\\AMLPOC.BERT.DATA\\bio.data\\bio.dataset.v1\\'
fileName = 'bio_dataset_splitted'

dataService = DataService()
docIds = dataService.GetDocIds()
sentences = dataService.GetSentences()
labels = dataService.GetLabels()

tokenizer = Tokenizer()