import torch
import jieba
import config
from src.model import InputMethodModel


def predict_batch(model, input_tensor):
    '''
    批量预测
    :param model:
    :param input_tensor:
    :return:[[1,2,3,4,5],[2,3,4,5,6]]
    '''
    model.eval()
    with torch.no_grad():
        # 输入索引给模型经过嵌入层
        # 输出词向量进行rnn计算
        # 对计算结果进行分类获取
        outputs = model(input_tensor)
        # outputs.shape[batch_size,vocab_size]

        top5_index = torch.topk(outputs, 5)[1]
        # top5_index=[values],[index]

    top5_index_list = top5_index.tolist()
    return top5_index_list


def predict(text, model, device, vocal_list):
    # 数据
    word_list = jieba.lcut(text)
    # 词转换为索引
    word2index = {word: inndex for word, inndex in enumerate(vocal_list)}
    index2word = {inndex: word for word, inndex in enumerate(vocal_list)}
    index_list = [word2index.get(word, 0) for word in word_list]

    input_tensor = torch.tensor(index_list).unsqueeze(0).to(device)
    # input_tensor.shape:[batch_size,seq_len]
    top5_index_list = predict_batch(model, input_tensor)
    top5_words = [index2word[index] for index in top5_index_list[0]]
    return top5_words


def run_predict():
    ######################################
    # 加载资源
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 获取词表
    with open(config.PROCESSED_DIR / 'vocab.txt', 'r', encoding='utf-8') as f:
        vocal_list = [line.strip() for line in f.readlines()]
    model = InputMethodModel(vocab_size=len(vocal_list))
    model.load_state_dict(torch.load(config.MODELS_DIR / 'model.pt'))
    ######################################
    print('请输入下一个word：(输入exit退出)')
    history_input = ''
    while True:
        user_input = input("> ")
        if user_input == 'exit':
            print('程序已退出')
            break
        if user_input.strip() == '':
            print('请输入下一个词')
            continue
        history_input += user_input
        print(f'历时输入:{history_input}')
        # 输入历时数据，预测下一个词
        print(predict(history_input, model, device, vocal_list))


if __name__ == '__main__':
    top5_words = predict("我们团队正在")
    print(top5_words)
