import torch

class TensorUtils():

    @staticmethod
    def InputsIntoTensor(inputs):
        inputs2Tensor = list(map(torch.tensor, inputs))
        return inputs2Tensor

    @staticmethod
    def TensorsToLongTensors(inputs):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        inputsLongTensors = []

        if device.type == 'cuda':
            for inpuT in inputs:
                inputsLongTensors.append(inpuT.type(torch.cuda.LongTensor)) 

        else:
            for inpuT in inputs:
                inputsLongTensors.append(inpuT.type(torch.LongTensor)) 

        return inputsLongTensors
