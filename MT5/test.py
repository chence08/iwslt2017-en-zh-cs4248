from datasets.load import load_dataset
import logging
import sacrebleu
import pandas as pd
from simpletransformers.t5 import T5Model, T5Args

raw_datasets = load_dataset('iwslt2017', 'iwslt2017-zh-en')

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


model_args = T5Args()
model_args.max_length = 512
model_args.length_penalty = 1
model_args.num_beams = 10

model = T5Model("mt5", "outputs", args=model_args)

en_zh_test = pd.DataFrame(raw_datasets['test']['translation'])
zh_truth = en_zh_test['zh'].tolist()
en_input = en_zh_test['en'].tolist()

zh_preds = model.predict(en_input)
en_zh_bleu = sacrebleu.corpus_bleu(zh_preds, zh_truth)
print("----------------------------------------------")
print("English to Chinese: ", en_zh_bleu.score)

en_preds = model.predict(zh_truth)
zh_en_bleu = sacrebleu.corpus_bleu(en_preds, en_input)
print("----------------------------------------------")
print("Chinese to English: ", zh_en_bleu.score)
