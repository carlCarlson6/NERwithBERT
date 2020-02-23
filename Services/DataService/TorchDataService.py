from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler
import common

class TorchDataService:
    def PutDataIntoTorch(self, inputs, masks, tags, batchSize):
        torchDataLoader = None

        inputsTensor, masksTensor, tagsTensors = common.utils.TensorUtils.InputsIntoTensor(inputs=[inputs, masks, tags])
        inputsLongTensor, masksLongTensor, tagsLongTensor = common.utils.TensorUtils.TensorsToLongTensors([inputsTensor, masksTensor, tagsTensors])

        data = TensorDataset(inputsLongTensor, masksLongTensor, tagsLongTensor)
        sampler = RandomSampler(data)
        
        torchDataLoader = TorchDataLoader(data, sampler=sampler, batch_size=batchSize, drop_last=True)        
        return torchDataLoader