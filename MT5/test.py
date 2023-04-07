from datasets.load import load_dataset
from evaluate import load
import logging
import pandas as pd
from simpletransformers.t5 import T5Model, T5Args
from pprint import pprint

raw_datasets = load_dataset('iwslt2017', 'iwslt2017-zh-en')

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


model_args = T5Args()
model_args.max_length = 512
model_args.length_penalty = 1
model_args.num_beams = 10

model = T5Model("mt5", "mt5_more_epochs/checkpoint-115634-epoch-2", args=model_args)
bleu = load('sacrebleu')

en_zh_test = pd.DataFrame(raw_datasets['test']['translation'])
zh_truth = en_zh_test['zh'].tolist()
en_input = en_zh_test['en'].tolist()

zh_preds = model.predict(en_input) # time consuming
en_zh_bleu = bleu.compute(predictions=zh_preds, references=zh_truth, tokenize='zh')
print("----------------------------------------------")
print("English to Chinese: ", en_zh_bleu)

en_preds = model.predict(zh_truth)
zh_en_bleu = bleu.compute(predictions=en_preds, references=en_input)
print("----------------------------------------------")
print("Chinese to English: ", zh_en_bleu)