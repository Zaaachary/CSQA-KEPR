# bsub -n 1 -q HPC.S1.GPU.X785.sha -o run_logs/train/train.ranking.0503-time.log -gpu num=1:mode=exclusive_process sh szcs/train_ranking_zfli.sh
cd ../Ranking_Model

model_type='electra'
problem_type='classification_BCE'
# 'regression', 'classification_BCE', 'classification_MCE'

task_name=$model_type"_LR"
# _Multi_Classification; _BC_Hard_Case; _Binary_Classification; _Regression"

init_model_root="/SISDC_GPFS/Home_SE/hy-suda/zfli/Models/init_model"
output_root="/SISDC_GPFS/Home_SE/hy-suda/zfli/Models/output_model"

if [ $model_type = 'albert' ] ; then
    PTM_name_or_path=/SISDC_GPFS/Home_SE/hy-suda/zfli/Models/init_model/albert-xxlarge-v2
else
    PTM_name_or_path=/SISDC_GPFS/Home_SE/hy-suda/zfli/Models/init_model/electra-large-discriminator
fi

CUDA_VISIBLE_DEVICES=0 python run_train.py \
    --dataset_path /SISDC_GPFS/Home_SE/hy-suda/zfli/CODE/ProtoQA/proto-qa-research/DATA/ranking_data/CommonGen/random/v1 \
    --PTM_name_or_path $PTM_name_or_path \
    --wkdt_path /SISDC_GPFS/Home_SE/hy-suda/zfli/CODE/ProtoQA/proto-qa-research/DATA/wiktionary.dict \
    --keyword_path /SISDC_GPFS/Home_SE/hy-suda/zfli/CODE/ProtoQA/proto-qa-research/DATA/generation_data/with_keyword_v1\
    --output_path $output_root/Ranking_Model \
    --task_name $task_name \
    --problem_type $problem_type \
    --overwrite_cache \
    --max_len 80 \
    --epoch 1 \
    --learning_rate 3e-6 \
    --adam_epsilon 1e-6 \
    --warmup_proportion 0.01\
    --train_batch_size_per_gpu 8 \
    --gradient_accumulation_step 1 \
    --dev_batch_size_per_gpu 32 \
    --seed 518 \
    --gpus 0 \
    --experiment commongen
    # --experiment soft_label
    # --add_wkdt \
    # --experiment pos_keyword_top1 \

    # --experiment weight_scale_no_shuffle \
