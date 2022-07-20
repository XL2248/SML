code_dir=thumt-sml
work_dir=$PWD
export PYTHONPATH=$work_dir/$code_dir:$PYTHONPATH
crg=True
mrg=True #True #True
coh=True
clus=True
vocab_data_dir=path_to_vocab_file
data_dir=path_to_data_file
checkpoint_dir=path_to_checkpoint_file
data_name=test

#for idx in $Step
output_dir=$work_dir/$checkpoint_dir
LOG_FILE=$output_dir/infer_${data_name}.log
Step="xxxxx"
for idx in $Step
do
    echo model_checkpoint_path: \"model.ckpt-$idx\" > $work_dir/$checkpoint_dir/checkpoint |tee -a ${LOG_FILE}
    cat $work_dir/$checkpoint_dir/checkpoint
    echo decoding with checkpoint-$idx |tee -a ${LOG_FILE}
    python $work_dir/$code_dir/thumt/bin/translator.py \
        --models transformer \
        --checkpoints $work_dir/$checkpoint_dir \
        --input $data_dir/${data_name}_bpe.32k.en \
        --vocabulary $vocab_dir/vocab.enzh.chat.share100 $vocab_dir/vocab.enzh.chat.en100.txt $vocab_dir/position.txt \
        --dev_context_source $data_dir/"$data_name"_ctx_src_bpe.32k.en \
        --dev_dialog_src_context $data_dir/"$data_name"_ctx_bpe.32k.en \
        --dev_dialog_tgt_context $data_dir/"$data_name"_ctx_bpe.32k.de \
        --dev_sample $data_dir/"$data_name"_bpe.32k.de \
        --output $output_dir/"$data_name".out.de.$idx \
        --parameters=decode_batch_size=64,use_crg=$crg,use_mrg=$mrg,use_coherence=$coh,use_clus=$clus
    echo evaluating with checkpoint-$idx |tee -a ${LOG_FILE}
    sed -r "s/(@@ )|(@@ ?$)//g" $output_dir/"$data_name".out.de.$idx > $output_dir/${data_name}.out.de.delbpe.$idx
    echo "multi-bleu:" |tee -a ${LOG_FILE}
    $data_dir/multi-bleu.perl $data_dir/${data_name}.tok.de < $output_dir/${data_name}.out.de.delbpe.$idx |tee -a ${LOG_FILE}
    echo finished of checkpoint-$idx |tee -a ${LOG_FILE}
done