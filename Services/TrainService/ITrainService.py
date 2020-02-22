from abc import ABC, abstractmethod


class ITrainService(ABC):
    @abstractmethod
    def __init__(self):
        super().__init__()