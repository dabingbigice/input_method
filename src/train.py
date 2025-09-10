import os
import time

import torch
from tqdm import tqdm

import config
from torch import nn
from dataset import get_dataloader
from model import InputMethodModel
from torch.utils.tensorboard import SummaryWriter

# 创建写入器，指定日志保存目录
writer = SummaryWriter(log_dir=config.LOGS_DIR / time.strftime("%Y%m%d-%H%M%S"))


def train_one_epoch(model, dataloader, loss_function, optimizer, device):
    '''
    训练一轮
    :param model:
    :param dataloader:
    :param loss_function:
    :param optimizer:
    :param device:
    :return:每个batch的平均值
    '''
    total_loss = 0
    model.train()
    model.to(device)
    for inputs, target in tqdm(dataloader, desc='train'):
        # inputs.shape[batch_size,sql_len]
        # target.shape[batch_size]
        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        # outputs.shape[batch_size,vocal_size]
        loss = loss_function(outputs, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # 批平均loss
    return total_loss / len(dataloader)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # 准备数据
    dataloader = get_dataloader(True)

    # 获取词表
    with open(config.PROCESSED_DIR / 'vocab.txt', 'r', encoding='utf-8') as f:
        vocal_list = [line.strip() for line in f.readlines()]
    # 模型创建
    model = InputMethodModel(vocab_size=len(vocal_list))

    # 损失函数
    loss_function = nn.CrossEntropyLoss()

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # 开始训练
    best_loss = float('inf')
    for epoch in range(config.EPOCHS):
        print(f'==============Epoch{epoch}==============')

        # 训练一轮的逻辑
        avg_loss = train_one_epoch(model, dataloader, loss_function, optimizer, device)

        model.state_dict()

        print(f'loss:{avg_loss}')
        if not os.path.exists(config.MODELS_DIR):
            os.mkdir(config.MODELS_DIR)
        # 记录训练结果
        writer.add_scalar('loss', avg_loss, epoch)
        if avg_loss < best_loss:
            best_loss = avg_loss

            torch.save(model.state_dict(), config.MODELS_DIR / f'model.pt')

            print(f'best loss {best_loss},_保存成功')
        else:
            print('无须保持')


if __name__ == '__main__':
    train()
