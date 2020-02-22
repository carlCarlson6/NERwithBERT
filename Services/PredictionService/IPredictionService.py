from abc import ABC, abstractmethod


class IPredictionService(ABC):
    @abstractmethod
    def __init__(self):
        super().__init__()
