# bsub -n 1 -q HPC.S1.GPU.X785.sha -o run_logs/infer/infer.ranking.0505-time.log -gpu num=1:mode=exclusive_process sh szcs/infer_ranking_zfli_inf.sh
cd ../Ranking_Model

init_model_root="/SISDC_GPFS/Home_SE/hy-suda/zfli/Models/init_model"

model_type='electra'

if [ $model_type = 'albert' ] ; then
    PTM_name_or_path=$init_model_root/albert-xxlarge-v2
else
    PTM_name_or_path=$init_model_root/electra-large-discriminator
fi

    # --multi_target \
    # --last_model \
CUDA_VISIBLE_DEVICES=1 python run_infer.py \
    --wkdt_path /SISDC_GPFS/Home_SE/hy-suda/zfli/CODE/ProtoQA/proto-qa-research/DATA/wiktionary.dict \
    --PTM_name_or_path $PTM_name_or_path \
    --max_len 140 \
    --output_path /SISDC_GPFS/Home_SE/hy-suda/zfli/Models/Finetuned_model/ProtoQA/T5-3B_wkdt_single_6660_rmbad_dpr/dev_set/wkdt_single_15.rmsame.ranked2.jsonl\
    --target_path /SISDC_GPFS/Home_SE/hy-suda/zfli/Models/Finetuned_model/ProtoQA/T5-3B_wkdt_single_6660_rmbad_dpr/dev_set/wkdt_single_15.rmsame.jsonl \
    --model_path /SISDC_GPFS/Home_SE/hy-suda/zfli/Models/output_model/Ranking_Model/electra_LR/data=top2_bz=1x8x1_ep=1_lr=3e-06_ae=1e-06_seed=518_-16GB-LS_soft_label/checkpoints/epoch=00-step=3581-val_loss=0.8934-auc=0.6694.ckpt \
    --dataset_path /SISDC_GPFS/Home_SE/hy-suda/zfli/CODE/ProtoQA/proto-qa-research/DATA/generation_data/with_keyword_v1_dpr/dev.crowdsourced.jsonl \
    --keyword_path /SISDC_GPFS/Home_SE/hy-suda/zfli/CODE/ProtoQA/proto-qa-research/DATA/generation_data/with_keyword_v1_dpr\
    --device 0\

    # --dataset_path /SISDC_GPFS/Home_SE/hy-suda/zfli/CODE/ProtoQA/proto-qa-research/DATA/generation_data/with_keyword_v1_dpr/dev.crowdsourced.jsonl \

# CUDA_VISIBLE_DEVICES=0 python run_infer.py \
#     --wkdt_path ../wiktionary.dict \
#     --PTM_name_or_path $PTM_name_or_path \
#     --max_len 140 \
#     --output_path /SISDC_GPFS/Home_SE/hy-suda/zfli/Models/output_model/ProtoQA_Submission/F1_test.jsonl\
#     --target_path /SISDC_GPFS/Home_SE/hy-suda/zfli/CODE/ProtoQA/proto-qa-research/DATA/ranking_data/BC_Hard_Case/v3.1/dev_ans.jsonl \
#     --model_path /SISDC_GPFS/Home_SE/hy-suda/zfli/Models/output_model/ProtoQA_Submission/ELECTRA_data=v3.1_bz=1x8x1_ep=1_lr=3e-06_ae=1e-06_seed=3407_-16GB-LS/checkpoints/epoch=00-step=3979-val_loss=0.1346-auc=0.9882.ckpt \
#     --dataset_path /SISDC_GPFS/Home_SE/hy-suda/zfli/CODE/ProtoQA/proto-qa-research/DATA/ranking_data/BC_Hard_Case/v3.1/dev_ref.jsonl \
#     --device 0 \


    # --dataset_path /SISDC_GPFS/Home_SE/hy-suda/zfli/CODE/ProtoQA/proto-qa-research/DATA/generation_data/with_keyword_v1_dpr/test.questions.jsonl \
    # --target_path /SISDC_GPFS/Home_SE/hy-suda/zfli/CODE/ProtoQA/proto-qa-research/DATA/ranking_data/benchmark/benchmark2.0.jsonl \
    # --output_path /SISDC_GPFS/Home_SE/hy-suda/zfli/Models/output_model/Submission/Electra_data=v3.1_bz=1x8x1_ep=1_lr=3e-06_ae=1e-06_seed=3407_-16GB-LS/benchmark2.0/bart-wkdt.jsonl\
    # --dataset_path /SISDC_GPFS/Home_SE/hy-suda/zfli/CODE/ProtoQA/proto-qa-research/DATA/generation_data/with_keyword_v1/dev.crowdsourced.jsonl \
    # --add_wkdt\
    # --experiment keyword_pos
    # --experiment pos
    # --dataset_path /SISDC_GPFS/Home_SE/hy-suda/zfli/CODE/ProtoQA/proto-qa-research/DATA/generation_data/with_keyword_v1/test.questions.jsonl \
    # --target_path /SISDC_GPFS/Home_SE/hy-suda/zfli/CODE/ProtoQA/proto-qa-research/DATA/ranking_data/small_model \

    # --target_path /SISDC_GPFS/Home_SE/hy-suda/zfli/Models/output_model/Ranking_Model/target \

    # --add_wkdt \

    # --multi_model
    # --max_len 120