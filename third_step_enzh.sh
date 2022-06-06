code_dir=thumt-dialog-wo-sp-decoder-w-mask-all-final-grad-bottom-update-share-four-adp
work_dir=$PWD
data_dir=/path/to/target_data/ende/chat_${src}2${tgt}_ctx3 # for evaluation
train_data=$data_dir
vocab_dir=$train_data
crg=$1
mrg=$2 #True #True
coh=$3
clus=$4 
cluts=False
clm=False
start_steps=$5
train_steps=$6
kl_steps2=5000
kl_steps1=$kl_steps2
export PYTHONPATH=$work_dir/$code_dir:$PYTHONPATH
model_name=target_zh2en_model_base_10w_sent_indomainchat_10w_w_task_chat5k_w_task_crg${crg}_mrg${mrg}_coh${coh}_clus${clus}_grad_adp$7
perfix=.nodup.clean
python $work_dir/$code_dir/thumt/bin/trainer.py \
  --model transformer \
  --output $work_dir/models/$model_name \
  --input $train_data/train_bpe.32k.zh $train_data/train_bpe.32k.en \
  --vocabulary $vocab_dir/vocab.enzh.chat.share100 $vocab_dir/vocab.enzh.chat.en100.txt $vocab_dir/position.txt \
  --validation $data_dir/dev_bpe.32k.zh \
  --references $data_dir/dev.tok.en \
  --context_source $train_data/train_ctx_src_bpe.32k.zh \
  --dialog_src_context $train_data/train_ctx_bpe.32k.zh \
  --dialog_tgt_context $train_data/train_ctx_bpe.32k.en \
  --sample $train_data/train_bpe.32k.en \
  --dev_context_source $data_dir/dev_ctx_src_bpe.32k.zh \
  --dev_dialog_src_context $data_dir/dev_ctx_bpe.32k.zh \
  --dev_dialog_tgt_context $data_dir/dev_ctx_bpe.32k.en \
  --dev_sample $data_dir/dev_bpe.32k.en \
  --parameters=device_list=[0,1,2,3],update_cycle=2,eval_steps=50,train_steps=$train_steps,batch_size=4096,max_length=128,constant_batch_size=False,residual_dropout=0.1,attention_dropout=0.1,relu_dropout=0.1,hidden_size=512,filter_size=2048,num_heads=8,shared_source_target_embedding=True,kl_annealing_steps=$kl_steps1,kl_annealing_steps2=$kl_steps2,use_crg=$crg,use_mrg=$mrg,use_coherence=$coh,use_clus=$clus,use_cluts=$cluts,use_clm=$clm,start_steps=$start_steps,max_relative_dis=16

chmod 777 -R ${work_dir}/models/$model_name
