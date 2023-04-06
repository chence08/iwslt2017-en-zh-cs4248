import random
import torch
import config

DEVICE = config.device

class Encoder(torch.nn.Module):
  def __init__(self, input_size, embedding_size, hidden_size, num_layers, dropout):
    super().__init__()
    self.hidden_size = hidden_size
    self.embedding_size = embedding_size
    self.num_layers = num_layers

    self.embedding = torch.nn.Embedding(input_size, embedding_size)
    self.LSTM = torch.nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, dropout=dropout)
    self.dropout = torch.nn.Dropout(dropout)

  def forward(self, input_seq):
    word_embeddings = self.embedding(input_seq)
    word_embeddings = self.dropout(word_embeddings)
    outputs, (hidden,cell) = self.LSTM(word_embeddings)
    return hidden, cell
  
class Decoder(torch.nn.Module):
  def __init__(self, output_size, embedding_size, hidden_size, num_layers, dropout):
    super().__init__()

    self.hidden_size = hidden_size
    self.embedding_size = embedding_size
    self.num_layers = num_layers
    self.output_size = output_size

    self.embedding = torch.nn.Embedding(output_size, embedding_size)
    self.LSTM = torch.nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, dropout=dropout)
    self.out = torch.nn.Linear(hidden_size, output_size)
    self.dropout = torch.nn.Dropout(dropout)
  
  def forward(self, input, hidden, cell):
    input = input.unsqueeze(0)
    word_embeddings = self.embedding(input)
    word_embeddings = self.dropout(word_embeddings)
    outputs, (hidden,cell) = self.LSTM(word_embeddings,(hidden, cell))
    
    outputs = self.out(outputs.squeeze(0))
    return outputs, hidden, cell

class EncoderDecoder(torch.nn.Module):
  def __init__(self,encoder,decoder,device):
    super().__init__()

    self.encoder = encoder
    self.decoder = decoder
    self.device = device
  def forward(self,input,target_output,teacher_forcing_ratio=0.5):
    batch_size = target_output.shape[1]
    target_len = target_output.shape[0]
    target_vocab_size = self.decoder.output_size

    outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(self.device)
    hidden, cell = self.encoder(input)
    target_in = target_output[0,:]
    for t in range(1, target_len):
            #insert input token embedding, previous hidden and previous cell states
            #receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(target_in, hidden, cell)
            #place predictions in a tensor holding predictions for each token
            outputs[t] = output
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            #get the highest predicted token from our predictions
            top1 = output.argmax(1) 
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            target_in = target_output[t] if teacher_force else top1   
    return outputs

def init_weights(m):
    for name, param in m.named_parameters():
        torch.nn.init.uniform_(param.data, -0.08, 0.08)

def make_model(src_vocab, tgt_vocab, N=2, d_model=256, d_hidden=512, dropout=0.1):
  enc = Encoder(src_vocab, d_model, d_hidden, N, dropout)
  dec = Decoder(tgt_vocab, d_model, d_hidden, N, dropout)
  model = EncoderDecoder(enc, dec, DEVICE).to(DEVICE)
  model.apply(init_weights)
  return model