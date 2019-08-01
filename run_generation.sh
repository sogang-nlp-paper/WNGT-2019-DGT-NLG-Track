# usage ./run_generation.sh exp_name model_step
python translate.py -model models/$1/model_step_$2.pt -src data/rotowire/src_test.txt -output models/$1/pred.txt -verbose -max_length 850 -min_length 150 -gpu 0
