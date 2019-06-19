# CNNs for Text Classification in PyTorch

A minimal PyTorch implementation of Convolutional Neural Networks (CNNs) for text classification.

Supported features:
- Lookup, CNNs, RNNs and/or self-attentive encoding in the embedding layer
- Mini-batch training with CUDA

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
python3 train.py model char_to_idx word_to_idx tag_to_idx training_data.csv (validation_data) num_epoch
```

To predict:
```
python3 predict.py model.epochN char_to_idx word_to_idx tag_to_idx test_data
```

To evaluate:
```
python3 evaluate.py model.epochN char_to_idx word_to_idx tag_to_idx test_data
```
