import torch
from torch.utils.data import Dataset
import librosa
import numpy as np


class DS(Dataset): 
  def __init__(self, train=True):
      dataset_dir = "/workspace/dataset/양자화데이터셋/train.npy" if train else "/workspace/dataset/양자화데이터셋/test.npy"
      self.data = np.load(dataset_dir)
    

  # 총 데이터의 개수를 리턴
  def __len__(self): 
    return len(self.data)

  # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
  def __getitem__(self, idx): 
    noisy_wav = self.data[idx,:,0]
    clean_wav = self.data[idx,:,1]
    noisy_spec = np.abs(librosa.stft(noisy_wav, n_fft=256, hop_length=64, window='hann'))**2
    clean_spec = np.abs(librosa.stft(clean_wav, n_fft=256, hop_length=64, window='hann'))**2
    return torch.tensor(noisy_spec).unsqueeze(dim=0), torch.tensor(clean_spec).unsqueeze(dim=0)