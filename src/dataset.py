import torch

import pandas as pd
import config
from torch.utils.data import DataLoader, Dataset

'''
定义获取数据集
'''

# 1.定义dataset
class InputMethodDataset(Dataset):
    def __init__(self, data_path):
        # [{"input":[8249,4235,11839,8994,13067],"target":4911},{"input":[4235,11839,8994,13067,4911],"target":9933}]
        self.data = pd.read_json(data_path, orient='records', lines=True).to_dict('records')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_tensor = torch.tensor(self.data[idx]['input'], dtype=torch.long)
        target_tensor = torch.tensor(self.data[idx]['target'], dtype=torch.long)
        return input_tensor, target_tensor


# 2.获取dataloader方法

def get_dataloader(train=True):
    data_path = config.PROCESSED_DIR / ('index_train.jsonl' if train else config.PROCESSED_DIR / 'index_test.jsonl')
    dataset = InputMethodDataset(data_path)
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True,num_workers=4)


if __name__ == '__main__':
    train_dataloader = get_dataloader()
    print(len(train_dataloader))

    for inputs, targets in train_dataloader:
        # [batch_size,seq_len]
        print(f'inputs.shape:{inputs.shape}, targets.shape:{targets.shape}')
        break
