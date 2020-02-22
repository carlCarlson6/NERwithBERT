from abc import ABC, abstractmethod


class IMetricsService(ABC):
    @abstractmethod
    def __init__(self):
        super().__init__()