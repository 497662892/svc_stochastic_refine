# ===================== Inference for M4Singer =====================
# Note:
# --resume:
#   <root_path>/model/ckpts/<training_dataset>/<expriement_name>/<epoch>.pt

python -u main.py --debug False --evaluate True \
--dataset 'M4Singer' --converse True --model 'Transformer' \
--resume '/mntnfs/med_data5/yiwenhu/diffsvc/Singing-Voice-Conversion-BingliangLi/model/ckpts/Opencpop/Transformer_lr_2.5e-05/87.pt'