code_dir=thumt
work_dir=thumt1_code
data_dir=/path/to/data/chen/chat_en2de_ctx3
vocab_dir=$data_dir
export PYTHONPATH=$work_dir/$code_dir:$PYTHONPATH
model_name=zh2en_model_base_share_dev_shuf_final10w
python $work_dir/$code_dir/thumt/bin/trainer.py \
  --model transformer \
  --output $work_dir/models/$model_name \
  --input $train_data/train_all.ch.nodup.norm.tok.clean.newbpe.shuf $train_data/train_all.en.nodup.norm.tok.clean.newbpe.shuf \
  --vocabulary $vocab_dir/vocab.enzh.chat.share100 $vocab_dir/vocab.enzh.chat.en100.txt \
  --validation $data_dir/dev_bpe.32k.zh \
  --references $data_dir/dev.tok.en \
  --parameters=device_list=[0,1,2,3],update_cycle=2,eval_steps=5000,train_steps=100000,batch_size=4096,max_length=128,constant_batch_size=False,residual_dropout=0.1,attention_dropout=0.1,relu_dropout=0.1,hidden_size=512,filter_size=2048,num_heads=8,max_relative_dis=16,shared_source_target_embedding=True

chmod 777 -R ${work_dir}/models/$model_name
