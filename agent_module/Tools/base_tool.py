import os 
from abc import ABC, abstractmethod


class Tool_Base(ABC):
    
    @abstractmethod
    def process():
        """Process the input data and output the response"""
        pass 

    
    def __repr__(self):
        return self.__class__.__name__
