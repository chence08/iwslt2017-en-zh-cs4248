import torch
import json
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from utils import english_tokenizer_load
from utils import chinese_tokenizer_load

import config
DEVICE = config.device

class Batch:
    """Object for holding a batch of data with mask during training."""
    def __init__(self, src_text, trg_text, src,src_len, trg=None, pad=0):
        self.src_text = src_text
        self.trg_text = trg_text
        self.src_len = src_len

        src = src.to(DEVICE)
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            trg = trg.to(DEVICE)
            self.trg = trg
            self.trg_y = trg[1::]
            self.ntokens = (self.trg_y != pad).data.sum()


class MTDataset(Dataset):
    """The MTDataset class inherits from the Dataset class 
    dataset : list of dictionary
     
    """
    def __init__(self, dataset):
        #dataset is a list of dictionary
        
        self.out_en_sent, self.out_cn_sent = self.get_dataset(dataset, sort=True)
        self.sp_eng = english_tokenizer_load()
        self.sp_chn = chinese_tokenizer_load()
        self.PAD = self.sp_eng.pad_id()  # 0
        self.BOS = self.sp_eng.bos_id()  # 2
        self.EOS = self.sp_eng.eos_id()  # 3

    @staticmethod
    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    def get_dataset(self, dataset, sort=False):
        out_en_sent = []
        out_cn_sent = []
        for idx, _ in enumerate(dataset):
            out_en_sent.append(dataset[idx]['en'])
            out_cn_sent.append(dataset[idx]['zh'])
        if sort:
            sorted_index = self.len_argsort(out_en_sent)
            out_en_sent = [out_en_sent[i] for i in sorted_index]
            out_cn_sent = [out_cn_sent[i] for i in sorted_index]
        return out_en_sent, out_cn_sent
    #change here
    def __getitem__(self, idx):
        eng_text = self.out_en_sent[idx]
        chn_text = self.out_cn_sent[idx]
        return [eng_text, chn_text]

    def __len__(self):
        return len(self.out_en_sent)

    def collate_fn(self, batch):
        src_text = [x[0] for x in batch]
        tgt_text = [x[1] for x in batch]

        src_tokens = [[self.BOS] + self.sp_eng.EncodeAsIds(sent) + [self.EOS] for sent in src_text]
        tgt_tokens = [[self.BOS] + self.sp_chn.EncodeAsIds(sent) + [self.EOS] for sent in tgt_text]
        src_len = torch.ByteTensor([ len(l) for l in src_tokens])
        batch_input = pad_sequence([torch.LongTensor(np.array(l_)) for l_ in src_tokens],padding_value=self.PAD)
        batch_target = pad_sequence([torch.LongTensor(np.array(l_)) for l_ in tgt_tokens],padding_value=self.PAD)
        return Batch(src_text, tgt_text, batch_input, src_len, batch_target, self.PAD)
