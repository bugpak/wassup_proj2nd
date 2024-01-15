import torch

def mape(input, target):
  return (torch.abs(input - target)/target).mean() * 100