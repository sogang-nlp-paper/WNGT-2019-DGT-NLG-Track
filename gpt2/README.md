# Model name
This code is based on [pytorch-transformers](https://github.com/huggingface/pytorch-transformers) from huggingface.
  

## Dataset
We use rotowire data from [gdrive link](https://drive.google.com/drive/folders/1GvFBVvOa2YPy_X9aJ6KYLoz_CnqZN796).
`data/rotowire/` directory must contain `src_train.txt`, `src_valid.txt`, `src_test.txt`, `tgt_train.txt`, `tgt_valid.txt`, `tgt_test.txt`.
  
Assuming that the data is located in `data/rotowire/` directory,

original rotowire dataset url: [original json format dataset](https://github.com/harvardnlp/boxscore-data)
 
## Training
To train a model,
```bash
python run_openai_gpt2.py \
    --model_name gpt2 \
    --do_train \
    --dataset_path data/rotowire \
    --output_dir temp \
    --train_batch_size 4 \
    --eval_batch_size 4 \
    --num_train_epochs 100 \
    --early_stop
```

## Generation
To generate summary,
```bash
python run_openai_gpt2.py \
    --model_name gpt2 \
    --do_generate \
    --dataset_path data/rotowire \
    --output_dir temp \
    --generate_model_file pytorch_model.bin
```
