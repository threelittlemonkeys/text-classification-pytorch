# The Transformer in PyTorch

A PyTorch implementation of the Self-Attentive Encoder for text classification.

Supported features:
- Mini-batch training with CUDA

## Usage

Training data should be formatted as below:
```
source_sequence \t label
source_sequence \t label
...
```

To prepare data:
```
python prepare.py training_data
```

To train:
```
python train.py model vocab.src vocab.tgt training_data.csv num_epoch
```

To predict:
```
python predict.py model.epochN vocab.src vocab.tgt test_data
```
