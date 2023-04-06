Most of the code present are 

The code from the Old_Notebook_(RNN).ipynb has been incorporated with code from the Original Transformer section of the project which is [forked](https://github.com/chence08/ChineseNMT).

Additionally, a key difference in the training is that this model do not incorporate batch_first. It also only does greedy decoding.

Ensure you have the packages in requirements.txt installed.

To train model, just run main.py.

Code referenced from the following:
1. RNN Model: https://github.com/bentrevett/pytorch-seq2seq/blob/master/4%20-%20Packed%20Padded%20Sequences%2C%20Masking%2C%20Inference%20and%20BLEU.ipynb with a modification of the GRU being non-bidirectional (to increase training speed).