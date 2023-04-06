from tqdm import tqdm
from utils import chinese_tokenizer_load
import logging
import config
import sacrebleu
import torch

def run_epoch(data, model, criterion, optimizer=None, is_training=False,clip=1):
    total_loss = 0.
    i = 0
    for batch in tqdm(data):
      i+=1
      output = model(batch.src, batch.trg)
      output_dim = output.shape[-1]
      output = output[1:].view(-1, output_dim)
      trg = batch.trg[1:].view(-1)
      loss = criterion(output, trg)

      if is_training:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

      total_loss += loss
    return total_loss/i

def train(train_data, dev_data, model, criterion, optimizer):
    best_bleu_score = 0.0
    early_stop = config.early_stop
    for epoch in range(1, config.epoch_num + 1):
        model.train()
        train_loss = run_epoch(train_data, model, criterion, optimizer=optimizer, is_training = True)
        logging.info("Epoch: {}, loss: {}".format(epoch, train_loss))
        
        model.eval()
        dev_loss = run_epoch(dev_data, model,criterion)
        bleu_score = evaluate(dev_data, model)
        logging.info('Epoch: {}, Dev loss: {}, Bleu Score: {}'.format(epoch, dev_loss, bleu_score))
        if bleu_score > best_bleu_score:
            torch.save(model.state_dict(), config.model_path)
            best_bleu_score = bleu_score
            early_stop = config.early_stop
            logging.info("-------- Save Best Model! --------")
        else:
            early_stop -= 1
            logging.info("Early Stop Left: {}".format(early_stop))
        if early_stop == 0:
            logging.info("-------- Early Stop! --------")
            break
        

def evaluate(data, model, mode='dev'):
    sp_chn = chinese_tokenizer_load()
    trg = []
    res = []
    with torch.no_grad():
        for batch in tqdm(data):
            decode_result = model(batch.src,batch.trg,0)
            decode_result = torch.argmax(decode_result,dim=2)
            decode_result = decode_result.transpose(1,0).tolist()
            translation = decode(decode_result,sp_chn)
            res.extend(translation)
            trg.extend(batch.trg_text)
    trg = [trg]
    bleu = sacrebleu.corpus_bleu(res, trg, tokenize='zh')
    return float(bleu.score)

def decode(data,tokenizer):
    translation = []
    for sentence in data:
        batch_result = []
        batch_result.append(tokenizer.decode_ids(sentence))
        translation.extend(batch_result)
    return translation

        
def test(data, model, criterion):
    with torch.no_grad():
        model.load_state_dict(torch.load(config.model_path))
        model.eval()
        test_loss = run_epoch(data, model,criterion)
        bleu_score = evaluate(data, model, 'test')
        logging.info('Test loss: {},  Bleu Score: {}'.format(test_loss, bleu_score))