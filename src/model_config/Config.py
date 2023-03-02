from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass


class ModelConfig(metaclass=ABCMeta):
    @property
    @abstractmethod
    def model_name(cls):
        raise NotImplementedError
