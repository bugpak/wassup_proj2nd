import numpy as np
import torch
  
class PatchTSDataset(torch.utils.data.Dataset):
  def __init__(self, ts:np.array, patch_length:int=16, n_patches:int=6, prediction_length:int=4):
    self.patch_length = patch_length
    self.n_patches = n_patches
    self.window_size = int(patch_length * n_patches / 2)  # look-back window length
    self.prediction_length = prediction_length
    self.data = ts

  def __len__(self):
    return len(self.data) - self.window_size- self.prediction_length + 1

  def __getitem__(self, idx):
    look_back = self.data[idx:(idx+self.window_size)] #[1,...,48]
    look_back = np.concatenate([look_back]+[look_back[-int(self.patch_length/4):]]*2) #[0,...,48]+[4]*2
    x = np.array([look_back[idx*int(self.patch_length/2):(idx+2)*int(self.patch_length/2)] for idx in range(self.n_patches)])
    #[0:16],[8:24],[16:32],[24:40],[32:48],[40:56] -> [41,42,43,44,45,46,47,48,48,48,48,48,48,48,48,48]
    #                                                 [41,42,43,44,45,46,47,48,45,46,47,48,45,46,47,48]
    y = self.data[(idx+self.window_size):(idx+self.window_size+self.prediction_length)]
    return x, y