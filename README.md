# Model name
This code is based on [OpenNMT](https://github.com/OpenNMT/OpenNMT-py).
  
## Requirements
  
## Dataset
Data formatted for OpenNMT can be downloaded in the [gdrive link](https://drive.google.com/drive/folders/1GvFBVvOa2YPy_X9aJ6KYLoz_CnqZN796).
## Preprocessing
Assuming that the data is located in `data/rotowire` directory,
```
preprocess.py -train_src data/rotowire/src_train.txt -train_tgt data/rotowire/tgt_train.txt -valid_src data/rotowire/src_valid.txt -valid_tgt data/rotowire/tgt_valid.txt -save_data data/rotowire/demo -dynamic_dict -src_seq_length 700 -tgt_seq_length 1000
```
  
## Training
  
## Evaluation
  

