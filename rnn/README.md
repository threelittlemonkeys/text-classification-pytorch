# RNNs for Text Classification in PyTorch

A PyTorch implementation of Recurrent Neural Networks (RNNs) for text classification.

Supported features:
- Lookup, CNNs, RNNs and/or self-attentive encoding in the embedding layer
- Mini-batch training with CUDA
- Global attention (Luong et al 2015)
- Multi-head attention (Vaswani et al 2017)
- Self attention (Vaswani et al 2017)

## Usage

Training data should be formatted as below:
```
sentence \t label
sentence \t label
...
```

To prepare data:
```
python3 prepare.py training_data
```

To train:
```
python3 train.py model word_to_idx tag_to_idx training_data.csv (validation_data) num_epoch
```

To predict:
```
python3 predict.py model.epochN word_to_idx tag_to_idx test_data
```

To evaluate:
```
python3 evaluate.py model.epochN word_to_idx tag_to_idx test_data
```
