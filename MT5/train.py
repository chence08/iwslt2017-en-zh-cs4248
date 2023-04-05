from datasets.load import load_dataset
import pandas as pd
import logging
from simpletransformers.t5 import T5Args, T5Model

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

raw_datasets = load_dataset('iwslt2017', 'iwslt2017-zh-en')

train_df = pd.DataFrame(raw_datasets['train']['translation'])
train_df.columns = ['input_text', 'target_text']
reverse_df = train_df.copy()
reverse_df.columns = ['target_text', 'input_text']
train_df['prefix'] = 'translate english to chinese'
reverse_df['prefix'] = 'translate chinese to english'
train_df = pd.concat([train_df, reverse_df])

eval_df = pd.DataFrame(raw_datasets['validation']['translation'])
eval_df.columns = ['input_text', 'target_text']
reverse_df = eval_df.copy()
reverse_df.columns = ['target_text', 'input_text']
eval_df['prefix'] = 'translate english to chinese'
reverse_df['prefix'] = 'translate chinese to english'
eval_df = pd.concat([eval_df, reverse_df])

model_args = T5Args()
model_args.max_seq_length = 96
model_args.train_batch_size = 20
model_args.eval_batch_size = 20
model_args.num_train_epochs = 4
model_args.evaluate_during_training = True
model_args.evaluate_during_training_steps = 5000
model_args.use_multiprocessing = False
model_args.fp16 = False
model_args.save_steps = -1
model_args.save_model_every_epoch = True
model_args.save_eval_checkpoints = False
model_args.no_cache = True
model_args.reprocess_input_data = True
model_args.overwrite_output_dir = False
model_args.preprocess_inputs = False
model_args.num_return_sequences = 1
model_args.wandb_project = "MT5 English-Chinese Translation"

model = T5Model("mt5", "outputs", args=model_args)

model.train_model(train_df, eval_data=eval_df, output_dir='mt5_more_epochs')
