import utils
import config
import logging
import numpy as np

import torch
from torch.utils.data import DataLoader

from train import train, test
from data_loader import MTDataset
from utils import english_tokenizer_load
from model import make_model
from datasets import load_dataset

def run():
    utils.set_logger(config.log_path)
    raw_datasets = load_dataset('iwslt2017', 'iwslt2017-zh-en')
    train_dataset = MTDataset(raw_datasets['train']['translation'])

    dev_dataset = MTDataset(raw_datasets['validation']['translation'])
    test_dataset = MTDataset(raw_datasets['test']['translation'])

    logging.info("-------- Dataset Build! --------")
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size,
                                collate_fn=train_dataset.collate_fn)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=config.batch_size,
                                collate_fn=dev_dataset.collate_fn)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=config.batch_size,
                                collate_fn=test_dataset.collate_fn)

    model = make_model(config.src_vocab_size, config.tgt_vocab_size,
                        config.d_model, config.d_hidden, config.dropout)
    criterion = torch.nn.CrossEntropyLoss(ignore_index = 0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    train(train_dataloader, dev_dataloader, model, criterion, optimizer)
    test(test_dataloader, model, criterion)


if __name__ == "__main__":
    import os
    import warnings
    warnings.filterwarnings('ignore')
    run()
