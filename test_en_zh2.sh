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
        --vocabulary $vocab_dir/vocab.enzh.chat.share100 $vocab_dir/vocab.enzh.chat.zh100.txt $vocab_dir/position.txt \
        --dev_context_source $data_dir/"$data_name"_ctx_src_bpe.32k.en \
        --dev_dialog_src_context $data_dir/"$data_name"_ctx_bpe.32k.en \
        --dev_dialog_tgt_context $data_dir/"$data_name"_ctx_bpe.32k.zh \
        --dev_sample $data_dir/"$data_name"_bpe.32k.zh \
        --output $output_dir/"$data_name".out.zh.$idx \
        --parameters=decode_batch_size=128,use_crg=$crg,use_mrg=$mrg,use_coherence=$coh,use_clus=$clus
    echo evaluating with checkpoint-$idx |tee -a ${LOG_FILE}
#    cd $train_dir
    cat $output_dir/"$data_name".out.zh.$idx
    sed -r "s/(@@ )|(@@ ?$)//g" $output_dir/"$data_name".out.zh.$idx > $output_dir/${data_name}.out.zh.delbpe.$idx
    cat $output_dir/${data_name}.out.zh.delbpe.$idx
    perl $data_dir/chi_char_segment.pl < $output_dir/${data_name}.out.zh.delbpe.$idx > $output_dir/${data_name}.out.zh.delbpe.char.$idx
    echo "multi-bleu:" |tee -a ${LOG_FILE}
    $data_dir/multi-bleu.perl $data_dir/${data_name}.tok.zh.char < $output_dir/${data_name}.out.zh.delbpe.char.$idx |tee -a ${LOG_FILE}
    echo finished of checkpoint-$idx |tee -a ${LOG_FILE}
done