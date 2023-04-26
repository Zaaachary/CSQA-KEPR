# bsub -n 1 -q HPC.S1.GPU.X785.sha -o run_logs/train/train.generation.0506-time.log -gpu num=1:mode=exclusive_process sh szcs/train_generation_zfli_tr.sh
# bsub -n 1 -q HPC.S1.GPU.X795.suda -o run_logs/train/train.generation.0506-time.log -gpu num=1:mode=exclusive_process sh szcs/train_generation_zfli_tr.sh

# seed 220406; 4082022; 408920; 42; 220410

nvidia-smi
cd ../Generation_Model

# model_type='bart'
# task_name='bart_wkdt_dpr'
# task_name='bart_wkdt_multi_dpr_sep'
# task_name='bart_wkdt_multi4_dpr'
# task_name='bart_wkdt_dpr'

model_type='bart'
task_name='Project-Final'

init_model_root="/SISDC_GPFS/Home_SE/hy-suda/zfli/Models/init_model"
output_root="/SISDC_GPFS/Home_SE/hy-suda/zfli/Models/output_model"

if [ $model_type = 'gpt2' ] ; then
    PTM_name_or_path=$init_model_root/gpt2-large/
    # PTM_name_or_path=$init_model_root/gpt2-medium/
elif [ $model_type = 'bart-base' ] ; then
    PTM_name_or_path=$init_model_root/bart-base/
    model_type='bart'
elif [ $model_type = 'bart' ] ; then
    PTM_name_or_path=$init_model_root/bart-large/
else 
    PTM_name_or_path=/SISDC_GPFS/Home_SE/hy-suda/pre-train_model/T5-3B
    # PTM_name_or_path=/SISDC_GPFS/Home_SE/hy-suda/pre-train_model/T5-base
fi

# CUDA_VISIBLE_DEVICES=0,1,2 python run_train.py \
#     --PTM_name_or_path $PTM_name_or_path \
#     --dataset_path /SISDC_GPFS/Home_SE/hy-suda/zfli/CODE/ProtoQA/proto-qa-research/DATA/generation_data/with_keyword_v1_dpr \
#     --wkdt_path /SISDC_GPFS/Home_SE/hy-suda/zfli/CODE/ProtoQA/proto-qa-research/DATA/wiktionary.dict \
#     --output_path $output_root/Generation_Model \
#     --model_type $model_type \
#     --task_name $task_name \
#     --overwrite_cache \
#     --max_src_len 60 \
#     --max_tgt_len 60 \
#     --epoch 1 \
#     --learning_rate 1e-5 \
#     --adam_epsilon 1e-6 \
#     --weight_decay 0.0 \
#     --warmup_proportion 0.025 \
#     --train_batch_size_per_gpu 1 \
#     --gradient_accumulation_step 1 \
#     --dev_batch_size_per_gpu 32 \
#     --seed 10232 \
#     --T5_split 3\
#     --gpus 0  
    # --fp16 \
    # --dataset_path /SISDC_GPFS/Home_SE/hy-suda/zfli/CODE/Project-Final/Data/Raw_data/ProtoQA/with_keyword_v1_dpr \
CUDA_VISIBLE_DEVICES=0 python run_train.py \
    --PTM_name_or_path $PTM_name_or_path \
    --dataset_path /SISDC_GPFS/Home_SE/hy-suda/zfli/CODE/Project-Final/Data/Constructed_data/ProtoQA/Prefix_desc_v1 \
    --wkdt_path /SISDC_GPFS/Home_SE/hy-suda/zfli/CODE/ProtoQA/proto-qa-research/DATA/wiktionary.dict \
    --output_path $output_root/Generation_Model \
    --model_type $model_type \
    --task_name $task_name \
    --overwrite_cache \
    --max_src_len 110 \
    --max_tgt_len 125 \
    --epoch 1 \
    --learning_rate 1e-5 \
    --adam_epsilon 1e-6 \
    --weight_decay 0.0 \
    --warmup_proportion 0.025 \
    --train_batch_size_per_gpu 4 \
    --gradient_accumulation_step 2 \
    --dev_batch_size_per_gpu 32 \
    --seed 1028 \
    --gpus 0 \
    --do_eval \
    --data_seed 42 \
    --eval_target /SISDC_GPFS/Home_SE/hy-suda/zfli/CODE/Project-Final/Data/Constructed_data/ProtoQA/Prefix_desc_v1/dev.crowdsourced.jsonl \
    --experiment keyword_wkdt_dpr_rm_bad_prefix
    
    # --eval_target /SISDC_GPFS/Home_SE/hy-suda/zfli/CODE/Project-Final/Data/Raw_data/ProtoQA/with_keyword_v1_dpr/dev.crowdsourced.jsonl \
    # --experiment keyword_wkdt_dpr_rm_bad_prefix
    # --T5_split 3\
    # --experiment keyword_wkdt_multi_dpr_rm_bad
    # --experiment keyword_wkdt_multi_4_dpr_rm_bad
    # --experiment rm_bad


    # --experiment keyword_wkdt_multi_pos_rm_bad

    # origin 40 45
    # wkdt 80 85
    # wkdt multi 110 115
    # wkdt multi_3 4 120 125

