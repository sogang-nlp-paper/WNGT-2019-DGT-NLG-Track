# usage ./run_evaluation experiment_name
# settings
alias python2=/home/nlpgpu5/anaconda3/envs/hwijeen_2.7/bin/python
export NLTK_DATA=/home/hwijeen/data/nltk_data

echo '##### preparing data for evaluation... #####'
python2 tools/data_utils.py -mode prep_gen_data -gen_fi models/$1/pred.txt -dict_pfx data/rotowire/roto-ie -output_fi models/$1/pred.h5 -input_path data/rotowire -test

echo '##### evaluatig RG metrics(on docker)... this takes a while #####'
nvidia-docker run -ti --rm -v /home/hwijeen/WNGT2019/data:/home/WNGT2019/data -v /home/hwijeen/WNGT2019/models/:/home/WNGT2019/models hwijeen:latest th tools/extractor.lua -gpuid 1 -datafile data/rotowire/roto-ie.h5 -preddata models/$1/pred.h5 -dict_pfx data/rotowire/roto-ie -just_eval |tee models/$1/rg_result.txt

echo '##### evaluatig CS/CO metrics... #####'
python2 tools/non_rg_metrics.py data/rotowire/roto-gold-test.tuples.txt models/$1/pred.h5-tuples.txt |tee models/$1/cs_co_result.txt

echo '##### evaluatig BLEU metrics... #####'
perl tools/multi-bleu.perl data/rotowire/tgt_test.txt < models/$1/pred.txt |tee models/$1/bleu.txt

