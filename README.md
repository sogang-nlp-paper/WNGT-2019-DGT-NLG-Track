# Model name
This code is based on [OpenNMT](https://github.com/OpenNMT/OpenNMT-py).
  
## Requirements
  
## Dataset
[Original Rotowire dataset](https://github.com/harvardnlp/boxscore-data) is in json format.
Data formatted in txt for OpenNMT can be downloaded in the [gdrive link](https://drive.google.com/drive/folders/1GvFBVvOa2YPy_X9aJ6KYLoz_CnqZN796).
This repository contains both data in `data/rotowire` directory.
  
## Preprocessing
Assuming that the data is located in `data/rotowire` directory,
```
python preprocess.py -train_src data/rotowire/src_train.txt -train_tgt data/rotowire/tgt_train.txt -valid_src data/rotowire/src_valid.txt -valid_tgt data/rotowire/tgt_valid.txt -save_data data/rotowire/demo -dynamic_dict -src_seq_length 700 -tgt_seq_length 1000
```
  
## Training
To train a model similar to Wiseman et al.,
```
python train.py -data data/rotowire/demo -src_word_vec_size 600 -tgt_word_vec_size 600 -share_embeddings -feat_merge mlp -feat_vec_size 600 -encoder_type entity_mean -batch_size 60 -rnn_size 600 --copy_attn -gpu_ranks 0 -truncated_decoder 100 --save_model models/entity_mean
```
  
## Generation
Assuming that the trained model is saved in `models/entity_mean_step_80000.pt`,
```
python translate.py -model models/entity_mean_step_80000.pt -src data/rotowire/src_test.txt -out models/pred.txt -verbose -max_length 850 -min_length 150
```
  
## Evaluation
To perform evaluation, we need a extraction model and relations extracted from gold data.
This repository contains both items. Original files can be download in [model link](https://github.com/harvardnlp/data2text#evaluating-generated-summaries) and [extracted relations link](https://github.com/harvardnlp/data2text#evaluating-generated-summaries).
- Note that `data_utils.py` is in Python2.*
- Note that torch is not compatiable with CUDA10. To make it compatiable, refer to this [link](https://github.com/torch/cutorch/issues/834#issuecomment-428767642).
 ```
python2 tools/data_utils.py -mode make_ie_data -input_path 'data/rotowire' -output_fi 'data/rotowire/roto-ie.h5'
python2 tools/data_utils.py -mode prep_gen_data -gen_fi models/pred.txt -dict_pfx data/rotowire/roto-ie -output_fi models/pred.h5 -input_path data/rotowire -test
th tools/extractor.lua -gpuid 1 -datafile data/rotowire/roto-ie.h5 -preddata models/pred.h5 -dict_pfx data/rotowire/roto-ie -just_eval
 ```

