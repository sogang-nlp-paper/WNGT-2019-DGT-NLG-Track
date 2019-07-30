# usage ./run_experiment experiment_name gpu_id
# setting
export CUDA_VISIBLE_DEVICES=$2

# opt_cov_cont
#nohup python train.py -data data/rotowire/demo -src_word_vec_size 600 -tgt_word_vec_size 600 -feat_merge mlp -feat_vec_size 600 -encoder_type entity_mean  -rnn_size 600 -context_gate both -coverage_attn -copy_attn -reuse_copy_attn -early_stopping 3 -batch_size 64 -truncated_decoder 100 -optim adam -learning_rate 0.001 -start_decay_steps 40000 -gpu_rank 0 -proc_name hwii:$1 -save_model models/$1/model > models/$1.txt &

# wiseman
#nohup python train.py -data data/rotowire/demo -src_word_vec_size 600 -tgt_word_vec_size 600 -feat_merge mlp -feat_vec_size 600 -encoder_type entity_mean  -rnn_size 600 -copy_attn -reuse_copy_attn -early_stopping 3 -batch_size 64 -truncated_decoder 100 -optim sgd -gpu_rank 0 -proc_name hwii:$1 -save_model models/$1/model > models/$1.txt &

# export CUDA_VISIBLE_DEVICES=$2,$3
# transformer without positonal encoding
# nohup python train.py -data data/rotowire/demo -encoder_type transformer -decoder_type transformer -layers 6 -heads 8 -rnn_size 512 -src_word_vec_size 512 -tgt_word_vec_size 512 -feat_merge mlp -feat_vec_size 512 -transformer_ff 2048 -dropout 0.1 -accum_count 2 -copy_attn -reuse_copy_attn -early_stopping 3 -batch_size 8 -optim adam -learning_rate 2 -adam_beta2 0.998 -warmup_steps 8000 -decay_method noam -label_smoothing 0.1 -param_init_glorot -train_steps 200000 -proc_name hwii:$1 -save_model models/$1/model -world_size 2 -gpu_rank 0 1 > models/$1.txt &

# reviewnet
nohup python train.py -data data/rotowire/demo -encoder_type entity_mean -rnn_size 600 -src_word_vec_size 600 -tgt_word_vec_size 600 -feat_merge mlp -feat_vec_size 600 -early_stopping 3 -batch_size 64 -truncated_decoder 100 -optim adam -learning_rate 0.001 -gpu_rank 0 -proc_name hwii:$1 -save_model models/$1/model -review_steps 16 -review_type input -review_net > models/$1.txt &
