# bsub -n 1 -q HPC.S1.GPU.X785.sha -o run_logs/infer/infer.generation.0505-time.log -gpu num=1:mode=exclusive_process sh szcs/infer_generation_zfli_inf.sh

# bsub -n 1 -q HPC.S1.GPU.X795.suda -o run_logs/infer/infer.generation.date-time.log -gpu num=1:mode=exclusive_process sh szcs/infer_generation_zfli_inf.sh

nvidia-smi
cd ../Generation_Model

init_model_root="/SISDC_GPFS/Home_SE/hy-suda/zfli/Models/init_model"

# model_type='gpt2'
model_type='bart'

if [ $model_type = 'gpt2' ] ; then
    PTM_name_or_path=$init_model_root/gpt2-large/
elif [ $model_type = 'bart-base' ] ; then
    PTM_name_or_path=$init_model_root/bart-base/
    model_type='bart'
else
    PTM_name_or_path=$init_model_root/bart-large/
fi

# CUDA_VISIBLE_DEVICES=0 python run_infer.py \
#     --wkdt_path /SISDC_GPFS/Home_SE/hy-suda/zfli/CODE/ProtoQA/proto-qa-research/DATA/wiktionary.dict\
#     --dataset_path /SISDC_GPFS/Home_SE/hy-suda/zfli/CODE/ProtoQA/proto-qa-research/DATA/generation_data/evaluate/demo.jsonl \
#     --model_path /SISDC_GPFS/Home_SE/hy-suda/zfli/Models/output_model/ProtoQA_Submission/GPT2_bz=1x4x2_ep=1_lr=1e-05_ae=1e-06_seed=220406_keyword_wkdt_multi_dpr_rm_bad/checkpoints/epoch=00-step=5864-val_loss=1.746466.ckpt \
#     --output_path /SISDC_GPFS/Home_SE/hy-suda/zfli/CODE/Research/ \
#     --PTM_name_or_path $PTM_name_or_path \
#     --model_type $model_type \
#     --device 0 \
#     --max_src_len 110 \
#     --eval_length 188 \
#     --beam_nums 30 \
#     --sample_nums 15 \

CUDA_VISIBLE_DEVICES=0 python run_infer.py \
    --last_model \
    --wkdt_path /SISDC_GPFS/Home_SE/hy-suda/zfli/CODE/ProtoQA/proto-qa-research/DATA/wiktionary.dict\
    --dataset_path /SISDC_GPFS/Home_SE/hy-suda/zfli/CODE/Project-Final/Data/Constructed_data/ProtoQA/Prefix_desc_v1/dev.crowdsourced.jsonl \
    --model_path /SISDC_GPFS/Home_SE/hy-suda/zfli/Models/output_model/Generation_Model/Project-Final/bz=1x4x2_ep=1_lr=1e-05_ae=1e-06_seed=1027_keyword_wkdt_dpr_rm_bad_prefix/checkpoints \
    --PTM_name_or_path $PTM_name_or_path \
    --model_type $model_type \
    --device 0 \
    --max_src_len 110 \
    --eval_length 188 \
    --beam_nums 30 \
    --sample_nums 15 \
    --experiment keyword_wkdt_dpr
    # --experiment keyword_wkdt_dpr

    # --experiment keyword_wkdt_multi_3_dpr

    # --dataset_path /SISDC_GPFS/Home_SE/hy-suda/zfli/CODE/ProtoQA/proto-qa-research/DATA/generation_data/with_keyword_v1_dpr/dev.crowdsourced.jsonl\
    # --experiment keyword_wkdt_multi_pos
    # --experiment keyword_wkdt_multi_pos
    
    # --experiment keyword_wkdt_pos
# keyword wkdt  100
# keyword wkdt multi 120