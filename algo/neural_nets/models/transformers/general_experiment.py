# Created by Hansi at 7/13/2020
import pandas as pd
import torch

from algo.neural_nets.models.transformers.args.args import MODEL_TYPE, MODEL_NAME, args
from algo.neural_nets.models.transformers.common.run_model import ClassificationModel
# from algo.neural_nets.models.transformers.common.utils import tokenize_text
from project_config import TRAINING_DATA_PATH, VALIDATION_DATA_PATH


def tokenize_text(data, tokenizer):
    max_tokens = 0
    for text in data:
        tokens = tokenizer.tokenize(text)
        if len(tokens) > max_tokens:
            max_tokens = len(tokens)
    return max_tokens


def get_max_seq_length():
    train = pd.read_csv(TRAINING_DATA_PATH, sep='\t')
    dev = pd.read_csv(VALIDATION_DATA_PATH, sep='\t')

    model = ClassificationModel(MODEL_TYPE, MODEL_NAME, args=args, use_cuda=torch.cuda.is_available())
    tokenizer = model.tokenizer
    train_max = tokenize_text(train["Text"], tokenizer)
    dev_max = tokenize_text(dev["Text"], tokenizer)

    print('training set max seq length: ', train_max)
    print('dev set max seq length: ', dev_max)


if __name__ == '__main__':
    get_max_seq_length()
