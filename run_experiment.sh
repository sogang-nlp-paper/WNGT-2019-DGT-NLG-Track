# usage ./run_experiment experiment_name
echo 'experiment name:' $1
echo 'gpu id:' $2

export CUDA_VISIBLE_DEVICES=$2

# opt_cov_cont
#nohup python train.py -data data/rotowire/demo -src_word_vec_size 600 -tgt_word_vec_size 600 -feat_merge mlp -feat_vec_size 600 -encoder_type entity_mean  -rnn_size 600 -context_gate both -coverage_attn -copy_attn -reuse_copy_attn -early_stopping 3 -batch_size 64 -truncated_decoder 100 -optim adam -learning_rate 0.001 -gpu_rank 0 -proc_name hwii:$1 -save_model models/$1/model > models/$1.txt &

# wiseman
nohup python train.py -data data/rotowire/demo -src_word_vec_size 600 -tgt_word_vec_size 600 -feat_merge mlp -feat_vec_size 600 -encoder_type entity_mean  -rnn_size 600 -copy_attn -reuse_copy_attn -early_stopping 3 -batch_size 64 -truncated_decoder 100 -optim sgd -gpu_rank 0 -proc_name hwii:$1 -save_model models/$1/model > models/$1.txt &
