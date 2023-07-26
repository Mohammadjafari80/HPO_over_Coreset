from .base_selection import CoresetSelection 
import numpy as np

class RandomCoresetSelection(CoresetSelection):
    
    def get_coreset(self):
        random_idx = np.random.choice(len(self.dataset), self.count, replace=False)
        return random_idx
