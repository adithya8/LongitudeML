import torch
import torch.nn as nn

class MIDataCollator:
    def __init__(self, collation_function:str):
        """
            Data Collator Abstract Class for MI
            
            Args:
                collation_function: The collation function to use.
        """
        # TODO: MASK, PAD tokens
        self.collation_function = collation_function
    
    def __call__(self, features):
        if self.collation_function is "strict-length":
            return self.strict_length_collation(features)
        elif self.collation_function is "all-lengths":
            return self.all_lengths_collation(features)
        else:
            raise ValueError("Collation function {} not implemented".format(self.collation_function))


class StrictLengthCollator(MIDataCollator):
    """
        Collates data to pre-defined lengths. No sequence is longer than the specified length.
    """
    collation_function:str
    length:int
    
    def strict_length_collation(self, features):
        if self.length == -1:
            print ("Using full timesteps")
            # Use all the timesteps of the sequence
            pass
        elif self.length >=1:
            print ("Using {} timesteps".format(self.length))
            # Use the first self.length timesteps of the sequence
            pass
        else:
            raise ValueError("Length {} not supported".format(self.length))
    

class AllLengthsCollator(MIDataCollator):
    """
        Multi Instance Learning Collator. Collates data to all possible lengths.
    """
    collation_function:str
    prob:float=1.0 # Probability of using a particular length (like dropout). Default is 1.0, i.e. all lengths are used.
    
    def all_lengths_collation(self, features):
        print ("Using all possible lengths of timesteps")
        pass