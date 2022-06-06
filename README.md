# SML
Our code is basically based on the publicly available toolkit: THUMT-Tensorflow[1] (python version 3.6). Please refer to it in Github for the required dependency. (Just seach it on Github.)

The following steps are training our model and then test its performance in terms of BLEU, TER, and Sentence Similarity (Still updating).

Take En->De as an example
# Training

Our work involves three-stage training
## The first stage
1) bash first_step_ende.sh # set the training_step=100,000; Suppose the generated checkpoint file is located in path1

## The second stage (i.e., fine-tuning on the in-domain chat translation data)
2) bash second_step_ende.sh # Here, set the training_step=1; Suppose the generated checkpoint file is located in path2
3) python thumt1_code/thumt/scripts/combine_add.py --model path2 --part path1 --output path3  # copy the weight of the first stage to the second stage.
4) bash second_step_ende.sh # Here, set the --output=path3 and the training_step=200,000; Suppose the generated checkpoint file is path4


## The third stage (i.e., fine-tuning on the target chat translation data)
5) bash third_step_ende.sh # Here, set the training_step=1; Suppose the generated checkpoint file is located in path5
6) python thumt1_code/thumt/scripts/combine_add.py --model path4 --part path5 --output path6  # copy the weight of the first stage to the second stage.
7) bash third_step_ende.sh # Here, set the --output=path6 and the training_step=205,000; Suppose the generated checkpoint file is path7


# Test by multi-blue.perl
8) bash test_en_de2.sh # set the checkpoint file path to path7 in this script. # Suppose the predicted file is located in path8 at checkpoint step xxxxx

# Test by SacreBLEU and TER
Required TER: v0.7.25; Sacre-BLEU: version.1.4.13 (BLEU+case.mixed+numrefs.1+smooth.exp+tok.13a+version.1.4.13)

9) python SacreBLEU_TER_Coherence_Evaluation_code/cal_bleu_ter4ende.py # Please correctly set the golden file and predicted file in this file and in sacrebleu_ende.py, respectively.


# Coherence Evaluation by Sentence Similarity
Required: gensim; MosesTokenizer

10) python SacreBLEU_TER_Coherence_Evaluation_code/train_word2vec.py # firstly downloading the corpus in [2] and then training the word2vec.
11) python SacreBLEU_TER_Coherence_Evaluation_code/eval_coherence.py # putting the file containing three precoding utterances and the predicted file in corresponding location and then running it.


# Reference
[1] Zhixing Tan, Jiacheng Zhang, Xuancheng Huang, Gang Chen, Shuo Wang, Maosong Sun, Huanbo Luan, and Yang Liu. 2020. THUMT: An open-source toolkit for neural machine translation. In Proceedings of AMTA, pages 116–122.
[2] Bill Byrne, Karthik Krishnamoorthi, ChinnadhuraiSankar, Arvind Neelakantan, Ben Goodrich, DanielDuckworth, Semih Yavuz, Amit Dubey, KyuYoungKim, and Andy Cedilnik. 2019. Taskmaster-1: Toward a realistic and diverse dialog dataset. In Proceedings of EMNLP-IJCNLP, pages 4516–4525.

@inproceedings{liang-etal-2022-scheduled,
    title = "Scheduled Multi-task Learning for Neural Chat Translation",
    author = "Liang, Yunlong  and
      Meng, Fandong  and
      Xu, Jinan  and
      Chen, Yufeng  and
      Zhou, Jie",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.300",
    doi = "10.18653/v1/2022.acl-long.300",
    pages = "4375--4388",
    abstract = "Neural Chat Translation (NCT) aims to translate conversational text into different languages. Existing methods mainly focus on modeling the bilingual dialogue characteristics (e.g., coherence) to improve chat translation via multi-task learning on small-scale chat translation data. Although the NCT models have achieved impressive success, it is still far from satisfactory due to insufficient chat translation data and simple joint training manners. To address the above issues, we propose a scheduled multi-task learning framework for NCT. Specifically, we devise a three-stage training framework to incorporate the large-scale in-domain chat translation data into training by adding a second pre-training stage between the original pre-training and fine-tuning stages. Further, we investigate where and how to schedule the dialogue-related auxiliary tasks in multiple training stages to effectively enhance the main chat translation task. Extensive experiments on four language directions (English-Chinese and English-German) verify the effectiveness and superiority of the proposed approach. Additionally, we will make the large-scale in-domain paired bilingual dialogue dataset publicly available for the research community.",
}
