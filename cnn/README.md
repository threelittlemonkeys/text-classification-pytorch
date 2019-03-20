# CNNs for Text Classification in PyTorch

A minimal PyTorch implementation of Convolutional Neural Networks (CNNs) for text classification.

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
python train.py model word_to_idx tag_to_idx training_data.tsv num_epoch
```

To predict:
```
python predict.py model.epochN word_to_idx tag_to_idx test_data
```

To evaluate:
```
python evaluate.py model.epochN word_to_idx tag_to_idx test_data
```
