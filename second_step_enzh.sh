code_dir=thumt-sml
work_dir=$PWD
data_dir=/path/to/target_data/ende/chat_${src}2${tgt}_ctx3 # for evaluation
train_data=/path/to/in_domain_data/chatnmt/chat4pretrain_ende/
vocab_dir=$train_data
#train_data=$data_dir
crg=$1
mrg=$2 #True #True
coh=$3
clus=$4
cluts=False
clm=False
src=zh
tgt=en
mode=$5
typ=$6
kl_steps1=100000
kl_steps2=100000
#typ=$8
sta=$8
end=200000
if [ $mode == "base" ]
then
    rd=0.1
    hd=8
    hs=512
    fs=2048
    mrd=16
else
    rd=0.3
    hd=16
    hs=1024
    fs=4096
    mrd=8
fi
export PYTHONPATH=$work_dir/$code_dir:$PYTHONPATH
model_name=second_zh2en_model_${mode}_10w_sent_indomainchat_10w_crg${crg}_mrg${mrg}_coh${coh}_clus${clus}_grad${typ}
perfix=.nodup.clean
python $work_dir/$code_dir/thumt/bin/trainer.py \
  --model transformer \
  --output $work_dir/models/$model_name \
  --input $train_data/train_dialog_${src}.txt$perfix $train_data/train_dialog_${tgt}.txt$perfix \
  --vocabulary $vocab_dir/vocab.enzh.chat.share100 $vocab_dir/vocab.enzh.chat.${tgt}100.txt $vocab_dir/position.txt \
  --validation $data_dir/dev_bpe.32k.zh \
  --references $data_dir/dev.tok.en \
  --context_source $train_data/train_dialog_zh_ctx_src.txt$perfix \
  --dialog_src_context $train_data/train_dialog_zh_ctx.txt$perfix \
  --dialog_tgt_context $train_data/train_dialog_en_ctx.txt$perfix \
  --sample $train_data/train_dialog_en_sample.txt$perfix \
  --dev_context_source $data_dir/dev_ctx_src_bpe.32k.zh \
  --dev_dialog_src_context $data_dir/dev_ctx_bpe.32k.zh \
  --dev_dialog_tgt_context $data_dir/dev_ctx_bpe.32k.en \
  --dev_sample $data_dir/dev_bpe.32k.${tgt} \
  --parameters=device_list=[0,1,2,3],update_cycle=8,eval_steps=5000,train_steps=$end,batch_size=1024,max_length=128,constant_batch_size=False,residual_dropout=$rd,attention_dropout=0.1,relu_dropout=0.1,hidden_size=$hs,filter_size=$fs,num_heads=${hd},shared_source_target_embedding=True,kl_annealing_steps=$kl_steps1,kl_annealing_steps2=$kl_steps2,use_crg=$crg,use_mrg=$mrg,use_coherence=$coh,use_clus=$clus,use_cluts=$cluts,use_clm=$clm,start_steps=$sta,max_relative_dis=$mrd

#,use_coherence=$use_coherence,use_trans_selection=$use_trans_selection,alpha=$alpha

chmod 777 -R ${work_dir}/models/$model_name
