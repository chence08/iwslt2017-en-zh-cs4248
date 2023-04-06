Most of the code present are 

The code from the Old_Notebook_(RNN).ipynb has been incorporated with code from the Original Transformer section of the project which is [forked](https://github.com/chence08/ChineseNMT).

The main changes involves in removing the code regarding masking, attention mostly as RNN LSTM implementation
do not require it.

Additionally, a key difference in the training is that this model do not incorporate batch_first i.e tensors having
its first dimension indicating the batch_size as LSTM default implementation do not assume this. It also only does greedy decoding.

Ensure you have the packages in requirements.txt installed.

To train model, just run main.py.

Code referenced from the following:
1. RNN-LSTM Model: https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb
