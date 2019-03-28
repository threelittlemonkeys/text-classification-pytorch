# CNNs for Text Classification in PyTorch

A minimal PyTorch implementation of Convolutional Neural Networks (CNNs) for text classification.

Supported features:
- Character and/or word embeddings in the input layer
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
python prepare.py training_data
```

To train:
```
python train.py model char_to_idx word_to_idx tag_to_idx training_data.csv num_epoch
```

To predict:
```
python predict.py model.epochN char_to_idx word_to_idx tag_to_idx test_data
```

To evaluate:
```
python evaluate.py model.epochN char_to_idx word_to_idx tag_to_idx test_data
```
