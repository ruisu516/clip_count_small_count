# @title train clip
model_name="clip-vit-base-patch32"
ref_obj="dogs"
python train_clip.py \
    --num_epochs 20 \
    --batch_size 128 \
    --lr 1e-4 \
    --ref_obj $ref_obj \
    --train_data_path "../../testadapt/my_data/my_train_data_${ref_obj}.pth"\
    --eval_data_path "../../testadapt/my_data/my_val_data_${ref_obj}.pth" \
    --model "openai/${model}" \
    --model_save_path "./${model}_clip_text_projection_${ref_obj}"