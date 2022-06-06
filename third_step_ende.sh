code_dir=thumt-sml
work_dir=$PWD
data_dir=/path/to/target_data/ende/chat_${src}2${tgt}_ctx3 # for evaluation
train_data=$data_dir
vocab_dir=$train_data
crg=$1
mrg=$2 #True #True
coh=$3
clus=$4 
src=$5
tgt=$6
cluts=False
clm=False
start_steps=$7
train_steps=$8

kl_steps2=5000
kl_steps1=$kl_steps2
export PYTHONPATH=$work_dir/$code_dir:$PYTHONPATH
model_name=target_zh2en_model_base_10w_sent_indomainchat_10w_chat5k_w_task_crgTrue_mrgTrue_cohTrue_clusTrue_original_sml$9

python $work_dir/$code_dir/thumt/bin/trainer.py \
  --model transformer \
  --output $work_dir/models/$model_name \
  --input $train_data/train.tok.bpe.32000.${src} $train_data/train.tok.bpe.32000.${tgt} \
  --vocabulary $vocab_dir/vocab.both.share100 $vocab_dir/vocab.both.share100 $vocab_dir/position.txt \
  --validation $data_dir/dev.tok.bpe.32000.${src} \
  --references $data_dir/dev.tok.${tgt} \
  --context_source $train_data/train_ctx_src.tok.bpe.32000.${src} \
  --dialog_src_context $train_data/train_ctx.tok.bpe.32000.${src} \
  --dialog_tgt_context $train_data/_ctx.tok.bpe.32000.${tgt} \
  --sample $train_data/train_${tgt}_sample.txt.norm.tok.bpe \
  --dev_context_source $data_dir/dev_ctx_src.tok.bpe.32000.${src} \
  --dev_dialog_src_context $data_dir/dev_ctx.tok.bpe.32000.${src} \
  --dev_dialog_tgt_context $data_dir/dev_ctx.tok.bpe.32000.${tgt} \
  --dev_sample $data_dir/dev.tok.bpe.32000.${tgt} \
  --parameters=device_list=[0,1,2,3],update_cycle=2,eval_steps=50,train_steps=$train_steps,batch_size=4096,max_length=128,constant_batch_size=False,residual_dropout=0.1,attention_dropout=0.1,relu_dropout=0.1,hidden_size=512,filter_size=2048,num_heads=8,shared_source_target_embedding=True,kl_annealing_steps=$kl_steps1,kl_annealing_steps2=$kl_steps2,use_crg=$crg,use_mrg=$mrg,use_coherence=$coh,use_clus=$clus,use_cluts=$cluts,use_clm=$clm,start_steps=$start_steps,max_relative_dis=16

chmod 777 -R ${work_dir}/models/$model_name
