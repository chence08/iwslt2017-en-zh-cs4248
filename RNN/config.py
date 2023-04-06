import torch

d_model = 256
d_hidden = 512
dropout = 0.1
padding_idx = 0
bos_idx = 2
eos_idx = 3
src_vocab_size = 32000
tgt_vocab_size = 32000
batch_size = 32
epoch_num = 40
early_stop = 5
lr = 3e-4

data_dir = './data'
model_path = './experiment/model.pth'
log_path = './experiment/train.log'
output_path = './experiment/output.txt'

# gpu_id and device id is the relative id
# thus, if you wanna use os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'
# you should set CUDA_VISIBLE_DEVICES = 2 as main -> gpu_id = '0', device_id = [0, 1]
gpu_id = '0'
device_id = [0, 1]

# set device
if gpu_id != '':
    device = torch.device(f"cuda:{gpu_id}")
else:
    device = torch.device('cpu')
