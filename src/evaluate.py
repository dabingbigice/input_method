import jieba
import torch

from src import config
from src.model import InputMethodModel
from dataset import get_dataloader
from predict import predict_batch


def evaluate_model(model, dataloader, device):
    '''

    :param model:
    :param dataloader:
    :param device:
    :return: top1,top5
    '''
    model.eval()
    total_count = 0
    top1_acc_count = 0
    top5_acc_count = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.tolist()
            top5_indexes_list = predict_batch(model, inputs)
            for idx, top5_indexes in enumerate(top5_indexes_list):
                total_count += 1
                if targets[idx] == top5_indexes[0]:
                    top1_acc_count += 1
                    print('top1')

                if targets[idx] in top5_indexes:
                    top5_acc_count += 1
                    print('top5')

    return top1_acc_count / total_count, top5_acc_count / total_count


def run_evaluate():
    print()
    ######################################
    # 加载资源
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 获取词表
    with open(config.PROCESSED_DIR / 'vocab.txt', 'r', encoding='utf-8') as f:
        vocal_list = [line.strip() for line in f.readlines()]
    model = InputMethodModel(vocab_size=len(vocal_list))
    model.load_state_dict(torch.load(config.MODELS_DIR / 'model.pt'))
    ######################################
    dataloader = get_dataloader(False)
    top1_acc, top5_acc = evaluate_model(model, dataloader, device)
    print(f'{'=' * 10}')
    print(f'top1准确率:{top1_acc}')
    print(f'top1准确率:{top5_acc}')


if __name__ == '__main__':
    run_evaluate()
