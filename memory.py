#Author: Kiur
import torch.utils.data as D
import torch

class StaticData(D.Dataset):
    """
    A object to store the dataset for training RL algorithms.
    """

    def __init__(self, state_s, act_s, rw_s):
        
        self.states = state_s
        self.acts = act_s
        self.rws = rw_s
    
    def __len__(self):
        return len(self.states)-1

    def __getitem__(self,idx):

        sample = {"state": self.states[idx],
                  "action": self.acts[idx],
                  "reward": self.rws[idx+1],
                  "n_state": self.states[idx+1],
                  "n_action": self.acts[idx+1]}
        return sample
