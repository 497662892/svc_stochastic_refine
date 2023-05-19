# ===================== Conversion for M4Singer =====================
# Note:
# --model_file:
#   <root_path>/model/ckpts/<target_dataset (training dataset)>/<expriement_name>/<epoch>.pt
# --target_singer_f0_file: 
#   <root_path>/preprocess/<source_dataset>/F0/test_f0.pkl
# --inference_dir:
#   <root_path>/model/ckpts/<source_dataset>/Transformer_eval_conversion

python -u converse.py \
--source_dataset 'M4Singer' --dataset_type 'test' \
--model_file '/mntnfs/med_data5/yiwenhu/diffsvc/Singing-Voice-Conversion-BingliangLi/model/ckpts/Opencpop/Transformer_lr_2.5e-05/87.pt' \
--target_singer_f0_file '/mntnfs/med_data5/yiwenhu/diffsvc/Singing-Voice-Conversion-BingliangLi/preprocess/M4Singer/F0/test_f0.pkl' \
--inference_dir '/mntnfs/med_data5/yiwenhu/diffsvc/Singing-Voice-Conversion-BingliangLi/model/ckpts/M4Singer/Transformer_eval_conversion/'