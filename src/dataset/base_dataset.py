from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from torch import Tensor
from typing import Literal


class BaseDataset(ABC):
    class_names: list[str]

    @abstractmethod
    def get_class_weights_inverse(self) -> Tensor:
        pass

    @abstractmethod
    def split(
        self, test_size=0.2, mode: Literal["subject", "random"] = "subject"
    ) -> tuple[Dataset, Dataset]:
        pass
