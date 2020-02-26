#import context
import torch
import torch.nn.functional as F
from transformers import BertForTokenClassification as BertModel
from transformers import AdamW
from tqdm import tqdm, trange
from seqeval.metrics import f1_score
from seqeval.metrics import classification_report,accuracy_score,f1_score
from NERBERT.core.Tokenizer.Tokenizer import Tokenizer
from datetime import datetime
import os, time, joblib, math
    

class Model():
    def __init__(self, numLabels, model='bert-base-multilingual-cased', loadNew=True):
        if loadNew:
            self.model = BertModel.from_pretrained(model, num_labels=numLabels)
            self.tokenizer = Tokenizer(model)
        else:
            self.model = None
            self.tokenizer = None

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.numLabels = numLabels

        if self.device.type == 'cuda':
            self.model.cuda()

        self.numberGpus = torch.cuda.device_count()
        if self.numberGpus > 1:
            self.model = torch.nn.DataParallel(self.model)

        self.modelName = None


    def Train(self, dataLoader, epochs=5, maxGradNOrm=1.0, fullFineTunning=True):
        numTrainOptimizationSteps = int(int(math.ceil(dataLoader.sampler.num_samples / dataLoader.batch_size) / 1) * epochs)

        if dataLoader.batch_size > dataLoader.sampler.num_samples:
            raise Exception('batch size can not be bigger than the total of training examples')

        if fullFineTunning:
            # finetune all model layers
            param_optimizer = list(self.model.named_parameters())
            no_decay = ['bias', 'gamma', 'beta']
            optimizer_grouped_parameters = [
                {
                    'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                    'weight_decay_rate': 0.01
                },
                {
                    'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                    'weight_decay_rate': 0.0
                }
            ] 
        else:
            # finetune only classifier layers
            param_optimizer = list(self.model.classifier.named_parameters()) 
            optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
            
        optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5)

        self.model.train();

        print("***** Running training *****")
        print("  Num examples = %d"%(dataLoader.sampler.num_samples))
        print("  Batch size = %d"%(dataLoader.batch_size))
        
        for _ in trange(epochs, desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            
            for step, batch in enumerate(dataLoader):
                # add batch to gpu
                batch = tuple(t.to(self.device) for t in batch)
                batchInputIds, batchInputMasks, batchLabels = batch
                # forward pass
                #outputs = self.model(inputs_ids=batchInputIds, attention_mask = batchInputMasks, labels = batchLabels)
                outputs = self.model(input_ids=batchInputIds, attention_mask = batchInputMasks, labels = batchLabels)
                loss, scores = outputs[:2]
                if self.numberGpus > 1:
                    # When multi gpu, average it
                    loss = loss.mean()
                # backward pass
                loss.backward()
                # track train loss
                tr_loss += loss.item()
                nb_tr_examples += batchInputIds.size(0)
                nb_tr_steps += 1
                # gradient clipping
                torch.nn.utils.clip_grad_norm_(parameters = self.model.parameters(), max_norm = maxGradNOrm)
                # update parameters
                optimizer.step()
                optimizer.zero_grad()

            # print train loss per epoch
            print("Train loss: {}".format(tr_loss/nb_tr_steps))

        return self


    def Evaluate(self, data):
        self.model.eval();

        evaluationLoss, evaluationLoss, nbEvalSteps, nbEvalExamples = 0, 0, 0, 0
        predictions = []

        print("***** Running evaluation *****")
        print("  Num examples ", len(data))
 
        for document in data:
            inputs = document.Inputs
            masks = document.Masks

            with torch.no_grad():
                inputs = torch.LongTensor(inputs)
                inputs = torch.unsqueeze(inputs, 0)
                masks = torch.LongTensor(masks)
                masks = torch.unsqueeze(masks, 0)

                outputs = self.model(inputs, token_type_ids=None, attention_mask=masks,)
                logits = outputs[0]
          
            softmaxLogits = F.softmax(logits, dim=2)
            argMaxLogits = torch.max(softmaxLogits, dim=2)[1].detach().cpu().numpy()
            probsLogits = torch.max(softmaxLogits, dim=2)[0].detach().cpu().numpy()
            
            masks = masks.to('cpu').numpy()
            for i, mask in enumerate(masks):
                predTemp = []

                for j, m in enumerate(mask):
                    if m:
                        predictedTag = self.tokenizer.index2tag[argMaxLogits[i][j]]
                        predictedProb = probsLogits[i][j]
                        prediction = (predictedTag, predictedProb)
                        predTemp.append(prediction)
                
                    else:
                        break

                predictions.append(predTemp)

        return predictions


    def Save(self, path):
        """
        If no specific path and name for the model is provided the model will be save on DATA/MODEL/modelName
        where modelName = yymmddHMS (year-month-day-hour-minutessecods) 
        """

        modelName = datetime.today().strftime('%y%m%d%H%M%S')
        self.modelName = modelName

        try:
            pathToModel = os.path.join(path, modelName)

            if not os.path.exists(pathToModel):
                    os.makedirs(pathToModel)

            model_to_save = self.model.module if hasattr(self.model, 'module') else self.model

            outputModelFile = os.path.join(pathToModel, "pytorch_model.bin")
            outputConfigFile = os.path.join(pathToModel, "config.json")

            torch.save(model_to_save.state_dict(), outputModelFile)
            model_to_save.config.to_json_file(outputConfigFile)

            self.tokenizer.tokenizer.save_vocabulary(pathToModel)

            joblib.dump(self.tokenizer.index2tag, pathToModel+'\\index2tag.joblib')

            return True

        except Exception as exception:
            print('ups ): something happened')
            print(exception)
            return False


    @staticmethod
    def Load(Path, NumLabels):
        try:
            pathToModel = Path
            pathToVocabulary = os.path.join(Path, 'vocab.txt')
            pathToIndex2Tag = os.path.join(Path, 'index2tag.joblib')

            bertModel = BertModel.from_pretrained(pathToModel)
            bertTokenizer = Tokenizer(vocabulary=pathToVocabulary, PathToIndex2Tag=pathToIndex2Tag)

            loadedModel = Model(numLabels=NumLabels, loadNew=False)
            loadedModel.model = bertModel
            loadedModel.tokenizer = bertTokenizer

            return loadedModel

        except Exception as exception:
            print('ups ): something happened')
            print(exception)
            return None

