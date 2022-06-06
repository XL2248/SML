code_dir=thumt-sml
crg=$1
mrg=$2 #True #True
coh=$3
clus=$4
cluts=False
clm=False
src=$5
tgt=$6
mode=$7
tstep=200000
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

work_dir=$PWD
data_dir=/path/to/target_data/ende/chat_${src}2${tgt}_ctx3 # for evaluation
train_data=/path/to/in_domain_data/chatnmt/chat4pretrain_ende/
vocab_dir=$train_data
#train_data=$data_dir
kl_steps1=100000
kl_steps2=100000
if [ $src == 'en' ]
then
    sstep=100000
else
    sstep=100000
fi
export PYTHONPATH=$work_dir/$code_dir:$PYTHONPATH
model_name=init_model/${src}2${tgt}_model_${mode}_crg${crg}_mrg${mrg}_coh${coh}_clus${clus}_grad
python $work_dir/$code_dir/thumt/bin/trainer.py \
  --model transformer \
  --output $work_dir/models_ende/$model_name \
  --input $train_data/train_deen_${src}.txt.nodup.clean $train_data/train_deen_${tgt}.txt.nodup.clean \
  --vocabulary $vocab_dir/vocab.both.share100 $vocab_dir/vocab.both.share100 $vocab_dir/position.txt \
  --validation $data_dir/dev.tok.bpe.32000.${src} \
  --references $data_dir/dev.tok.${tgt} \
  --context_source $train_data/train_deen_${src}_ctx_src.txt.nodup.clean \
  --dialog_src_context $train_data/train_deen_${src}_ctx.txt.nodup.clean \
  --dialog_tgt_context $train_data/train_deen_${tgt}_ctx.txt.nodup.clean \
  --sample $train_data/train_deen_${tgt}_sample.txt.nodup.clean \
  --dev_context_source $data_dir/dev_ctx_src.tok.bpe.32000.${src} \
  --dev_dialog_src_context $data_dir/dev_ctx.tok.bpe.32000.${src} \
  --dev_dialog_tgt_context $data_dir/dev_ctx.tok.bpe.32000.${tgt} \
  --dev_sample $data_dir/dev.tok.bpe.32000.${tgt} \
  --parameters=device_list=[0,1,2,3],update_cycle=8,eval_steps=5000,train_steps=$tstep,batch_size=1024,max_length=128,constant_batch_size=False,residual_dropout=${rd},attention_dropout=0.1,relu_dropout=0.1,hidden_size=${hs},filter_size=${fs},num_heads=${hd},shared_source_target_embedding=True,kl_annealing_steps=$kl_steps1,kl_annealing_steps2=$kl_steps2,use_crg=$crg,use_mrg=$mrg,use_coherence=$coh,use_clus=$clus,use_cluts=$cluts,use_clm=$clm,start_steps=$sstep,max_relative_dis=$mrd

#,use_coherence=$use_coherence,use_trans_selection=$use_trans_selection,alpha=$alpha

chmod 777 -R ${work_dir}/models_ende/$model_name
