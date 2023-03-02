from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from torch import Tensor

class BaseDataset(ABC):
    class_names: list[str]

    @abstractmethod
    def get_class_weights_inverse(self) -> Tensor:
        pass

    @abstractmethod    
    def split(self, test_size=0.2) -> tuple[Dataset, Dataset]:
        pass