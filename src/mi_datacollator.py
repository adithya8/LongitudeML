"""
    DEPRECATED SCRIPT
    Class for data collators. It won't be required for now. 
"""
from dataclasses import dataclass
import torch.nn as nn

DATA_COLLATION_FUNCTIONS = ["strict-length", "all-lengths"]

class MIDataCollator:
    def __call__(self, features):
        if self.collation_function == "strict-length":
            return self.strict_length_collation(features)
        elif self.collation_function == "all-lengths":
            return self.all_lengths_collation(features)
        else:
            raise ValueError("Collation function {} not available. Available choices: {}".format(self.collation_function, ",".join(DATA_COLLATION_FUNCTIONS)) )


@dataclass
class StrictLengthCollator(MIDataCollator):
    """
        Collates data to pre-defined lengths. No sequence is longer than the specified length.
    """
    length:int
    collation_function:str = "strict-length"
    
    def __post_init__(self):
        if self.length is None:
            self.length = -1
            raise Warning("Length not specified. Using full timesteps")
        elif self.length == -1:
            raise Warning("Using full timesteps")
        elif self.length == 0 or self.length < -1:
            raise ValueError("Invalid Length {} ".format(self.length))
        else:
            print ("Using {} timesteps".format(self.length))
            
        assert self.collation_function == "strict-length", "Collation function must be strict-length for this class"
            
    def strict_length_collation(self, features):
        if self.length == -1:
            print ("Using full timesteps")
            # Use all the timesteps of the sequence
            pass
        elif self.length >=1:
            print ("Using {} timesteps".format(self.length))
            # Use the first self.length timesteps of the sequence
            pass
    

@dataclass
class AllLengthsCollator(MIDataCollator):
    """
        Multi Instance Learning Collator. Collates data to all possible lengths.
    """
    prob:float=1.0 # Probability of using a particular length (like dropout). Default is 1.0, i.e. all lengths are used.
    collation_function:str="all-lengths"
    
    def __post_init__(self):
        if self.prob < 0 or self.prob > 1:
            raise ValueError("Probability {} value out of bounds".format(self.prob))
        
        assert self.collation_function == "all-lengths", "Collation function must be all-lengths for this class"
    
    def all_lengths_collation(self, features):
        print ("Using all possible lengths of timesteps")
        pass