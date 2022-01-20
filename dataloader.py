import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils import data
import os

def MinMaxScaler(data):
  """Min Max normalizer.
  
  Args:
    - data: original data
  
  Returns:
    - norm_data: normalized data
  """
  numerator = data - np.min(data, 0)
  denominator = np.max(data, 0) - np.min(data, 0)
  norm_data = numerator / (denominator + 1e-7)
  return norm_data

def real_data_loading(data_name, seq_len):
  """Load and preprocess real-world datasets.
  
  Args:
    - data_name: stock or energy
    - seq_len: sequence length
    
  Returns:
    - data: preprocessed data.
  """  
  assert data_name in ['stock','energy']
  
  if data_name == 'stock':
    ori_data = np.loadtxt('./data/stock/stock_data.csv', delimiter = ",",skiprows = 1)
  elif data_name == 'energy':
    ori_data = np.loadtxt('./data/energy/energy_data.csv', delimiter = ",",skiprows = 1)
        
  # Flip the data to make chronological data
  ori_data = ori_data[::-1]
  # Normalize the data
  ori_data = MinMaxScaler(ori_data)
    
  # Preprocess the dataset
  temp_data = []    
  # Cut data by sequence length
  for i in range(0, len(ori_data) - seq_len):
    _x = ori_data[i:i + seq_len]
    temp_data.append(_x)
        
  # Mix the datasets (to make it similar to i.i.d)
  idx = np.random.permutation(len(temp_data))    
  data = []
  for i in range(len(temp_data)):
    data.append(temp_data[idx[i]])
    
  return data

class Energy(data.Dataset):
    def __init__(self, device, split='train'):
        super(Energy, self).__init__()
        self.data = pd.read_csv('./data/ENB2012_data.csv')
        self.device = device

        l = self.data.shape[0]
        if split == 'train':
            self.data = self.data.iloc[:7*l//10, :]
        elif split == 'val':
            self.data = self.data.iloc[7*l//10:4*l//5, :]
        elif split == 'test':  
            self.data = self.data.iloc[4*l//5:, :]
        elif split == 'all':
            pass
        else:
            raise Exception('Split undefined, not in [train, val, test, all].')

    def __len__(self):
        return self.data.shape[0] 

    def __getitem__(self, i):
        d = self.data.iloc[[i]]
        d = np.asarray(d)[0]
        x = torch.tensor(d[:-2]).float()
        y = torch.tensor(d[-2:]).float()
        return x.to(self.device), y.to(self.device)

    def getXY(self):
        d = np.asarray(self.data)
        x = torch.tensor(d[:, :-2]).float()
        y = torch.tensor(d[:, -2:]).float()
        return x.to(self.device), y.to(self.device)

def load_power():
    def load_data():
        file = os.path.join('./data', 'power', 'data.npy')
        return np.load(file)

    def load_data_split_with_noise():
        rng = np.random.RandomState(42)

        data = load_data()
        rng.shuffle(data)
        N = data.shape[0]

        data = np.delete(data, 3, axis=1)
        data = np.delete(data, 1, axis=1)
        ############################
        # Add noise
        ############################
        # global_intensity_noise = 0.1*rng.rand(N, 1)
        voltage_noise = 0.01 * rng.rand(N, 1)
        # grp_noise = 0.001*rng.rand(N, 1)
        gap_noise = 0.001 * rng.rand(N, 1)
        sm_noise = rng.rand(N, 3)
        time_noise = np.zeros((N, 1))
        # noise = np.hstack((gap_noise, grp_noise, voltage_noise, global_intensity_noise, sm_noise, time_noise))
        # noise = np.hstack((gap_noise, grp_noise, voltage_noise, sm_noise, time_noise))
        noise = np.hstack((gap_noise, voltage_noise, sm_noise, time_noise))
        data += noise

        N_test = int(0.1 * data.shape[0])
        data_test = data[-N_test:]
        data = data[0:-N_test]
        N_validate = int(0.1 * data.shape[0])
        data_validate = data[-N_validate:]
        data_train = data[0:-N_validate]

        return data_train, data_validate, data_test

    def load_data_normalised():
        data_train, data_validate, data_test = load_data_split_with_noise()
        data = np.vstack((data_train, data_validate))
        mu = data.mean(axis=0)
        s = data.std(axis=0)
        data_train = (data_train - mu) / s
        data_validate = (data_validate - mu) / s
        data_test = (data_test - mu) / s

        return data_train, data_validate, data_test

    return load_data_normalised()

def save_splits():
    train, val, test = load_power()
    splits = (
        ('train', train),
        ('val', val),
        ('test', test)
    )
    for split in splits:
        name, data = split
        file = os.path.join('./data', 'power', '{}.npy'.format(name))
        np.save(file, data)


def print_shape_info():
    train, val, test = load_power()
    print(train.shape, val.shape, test.shape)


class PowerDataset(data.Dataset):
    def __init__(self, split='train', frac=None):
        path = os.path.join('./data', 'power', '{}.npy'.format(split))
        self.data = np.load(path).astype(np.float32)
        self.n, self.dim = self.data.shape
        if frac is not None:
            self.n = int(frac * self.n)

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return self.n

class TimeSeriesDataset(data.Dataset):
    def __init__(self, device, dataset='energy', split='train'):
        if dataset not in ['energy', 'stock']:
            raise ValueError('Dataset not in ["energy", "stock"]')
        self.dataset = dataset
        self.data = np.load('./data/' + dataset + '/data_24.npy')
        self.device = device
        l = self.data.shape[0]
        if split == 'train':
            self.data = self.data[:7*l//10]
        elif split == 'val':
            self.data = self.data[7*l//10:4*l//5]
        elif split == 'test':  
            self.data = self.data[4*l//5:]
        elif split == 'all':
            pass
        else:
            raise Exception('Split undefined, not in [train, val, test, all].')
    
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, i):
        x = torch.tensor(self.data[i][:12, :]).float()
        y = torch.tensor(self.data[i][12, :]).float()
        #print(x.shape, y.shape)
        return x.to(self.device), y.to(self.device)

    def getXY(self):
        x = torch.tensor(self.data[:, :-1, :]).float()
        y = torch.tensor(self.data[:, -1, :]).float()
        return x.to(self.device), y.to(self.device)

if __name__ == '__main__':
    '''
    data = real_data_loading('stock', 24)
    data = np.array(data)
    np.save('./data/stock/data_24.npy', data)
    print(data.shape)
    '''
    import seaborn as sns
    ds = TimeSeriesDataset(device='cpu',dataset='energy')
    x, y = ds.getXY()
    d = torch.cat([x, y.unsqueeze(1)], axis=1).numpy()
    d = d.reshape(-1, d.shape[-1])
    #d = np.random.normal(0, 5, size=(1000, 5))
    print(d.shape)
    #c = np.corrcoef(d.T)
    c = np.var(d.T, axis=1)
    idx = np.argpartition(c, -4)[-4:]
    print(c.shape)
    print(c)
    print(idx)
    #ax = sns.heatmap(
    #    c, 
    #    vmin=-1, vmax=1, center=0,
    #    cmap=sns.diverging_palette(20, 220, n=200),
    #    square=True)
    #fig = ax.get_figure()
    #fig.savefig("./ncor.png") 
    #dl = data.DataLoader(ds, batch_size=2, shuffle=True, drop_last=True)
    #for idx, batch in enumerate(dl):
    #    x = batch
    #    break
