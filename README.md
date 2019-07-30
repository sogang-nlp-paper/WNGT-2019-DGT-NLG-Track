# Model name
This code is based on [OpenNMT](https://github.com/OpenNMT/OpenNMT-py).
  
## Requirements
  
## Dataset
We need rotowire data in two formats: json and txt. Download [original json format dataset](https://github.com/harvardnlp/boxscore-data) to use in evaluation phase. Data formatted in txt for OpenNMT can be downloaded in the [gdrive link](https://drive.google.com/drive/folders/1GvFBVvOa2YPy_X9aJ6KYLoz_CnqZN796).
`data/rotowire/` directory must contain `src_train.txt`, `src_valid.txt`, `src_test.txt`, `tgt_train.txt`, `tgt_valid.txt`, `tgt_test.txt`, `train.json`, `valid.json`, `test.json`.
  
## Preprocessing
Assuming that the data is located in `data/rotowire/` directory,
```bash
python preprocess.py -train_src data/rotowire/src_train.txt -train_tgt data/rotowire/tgt_train.txt -valid_src data/rotowire/src_valid.txt -valid_tgt data/rotowire/tgt_valid.txt -save_data data/rotowire/demo -dynamic_dict -src_seq_length 700 -tgt_seq_length 1000
```
This will create `demo.train.0.pt`, `demo.valid.0.pt`, `demo.vocab.pt` in `data/rotowire/`.
  
## Training
To train a model similar to Wiseman et al.,
```bash
python train.py -data data/rotowire/demo -src_word_vec_size 600 -tgt_word_vec_size 600 -feat_merge mlp -feat_vec_size 600 -encoder_type entity_mean -rnn_size 600 -context_gate both -coverage_attn -copy_attn -reuse_copy_attn -early_stopping 3 -batch_size 64 -truncated_decoder 100 -optim adam -learning_rate 0.001 -gpu_ranks 0 -save_checkpoint_steps 0 -proc_name hwii:opt_cov_cont -save_model models/opt_cov_cont/model
```
Alternatively,
```bash
./run_experiment.sh experiment_name gpu_id # manually configure parameters in the shell file
```
This will create several models(`*.pt` files) in `models/opt_cov_cont/` directory.
  
## Generation
Assuming that the trained model is saved in `models/opt_conv_cont/model_step_80000.pt`,
```bash
python translate.py -model models/opt_cov_cont/model_step_80000.pt -src data/rotowire/src_test.txt -output models/opt_cov_cont/pred.txt -verbose -max_length 850 -min_length 150 -gpu 0
```
This will create `pred.txt` in `models/` directory.
  
## Evaluation
To perform evaluation, we need pretrained extraction model(s) and relations extracted from gold data.
This repository does not contain both items. Download the pretrained models following [the pretrained link](https://github.com/harvardnlp/data2text#evaluating-generated-summaries) and put the pretrained models(`*.t7`) in `tools/` directory. [Extracted relations link](https://github.com/harvardnlp/data2text#evaluating-generated-summaries) provide extracted tuples. Download `roto-gold-val.tuples.txt` and `roto-gold-test.tuples.txt` and put it in `data/rotowire` directory.
- Note that `data_utils.py` and `non_rg_metrics.py` is in Python2.
- Using the provided Dockerfile to build environment for running `evaluator.lua` is higly recommended.
 ```bash
# prepare data for evaluation
python2 tools/data_utils.py -mode make_ie_data -input_path 'data/rotowire' -output_fi 'data/rotowire/roto-ie.h5'
python2 tools/data_utils.py -mode prep_gen_data -gen_fi models/opt_cov_cont/pred.txt -dict_pfx data/rotowire/roto-ie -output_fi models/opt_cov_cont/pred.h5 -input_path data/rotowire -test

# docker
docker build -t hwijeen:1.0 .
nvidia-docker run -ti -d -v /home/hwijeen/WNGT2019/data:/home/WNGT2019/data -v /home/hwijeen/WNGT2019/models/:/home/WNGT2019/models --name hwijeen hwijeen:1.0

# perform evaluation
th tools/extractor.lua -gpuid 1 -datafile data/rotowire/roto-ie.h5 -preddata models/opt_cov_cont/pred.h5 -dict_pfx data/rotowire/roto-ie -just_eval |tee models/opt_cov_cont/rg_result.txt # run on docker
python2 tools/non_rg_metrics.py data/rotowire/roto-gold-test.tuples.txt models/opt_cov_cont/pred.h5-tuples.txt |tee models/opt_cov_cont/cs_co_result.txt
perl tools/multi-bleu.perl data/rotowire/tgt_test.txt < models/opt_cov_cont/pred.txt |tee models/opt_cov_cont/bleu.txt
 ```
`extractor.lua` measures Relation Generation(RG) score. Precision(`P%` in the original paper) and the number(`#`) corresponds to `nodup prec` and `nodup correct / 727`, respectiely(727 is the number of test samples).
`non_rg_metrics.py` calculates Content Selection(CS) and Content Ordering(CO) scores. Precision(`P%`) corresponds to `prec` and recall(`R%`) `rec`. Content ordering measured in normalized Damerau-Levenshtein distance(`DLD%`) is `avg score`.  
Alternatively,
```
./run_evaluation.sh experiment_name
```
This will show results in stdout and the results will also be saved as `rg_results.txt`, `cs_co_result.txt`, `bleu.txt` in `models/experiment_name/`.
