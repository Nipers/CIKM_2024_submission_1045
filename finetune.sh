# nreimers/MiniLMv2-L12-H384-distilled-from-RoBERTa-Large
nohup python finetune.py \
    --model_name MOARERecRoberta \
    --pretrain_ckpt ./pretrained.ckpt \
    --model_path ./MiniLM \
    --data_path ./Scientific \
    --num_train_epochs 15 \
    --batch_size 4 \
    --devices 0 1 2 3 4 5 6 7 \
    --gradient_accumulation_steps 64 \
    --fp16 \
    --output_dir finetune_logs \
    --finetune_negative_sample_size 128 \
    --use_img \
    --img_emb_path img_emb_beit3.npy \
    --img_marks_path has_img_beit3.npy \
    --num_adapter_layer 12 \
    --num_adapter 8 \
    --use_moa \
    --adapter_top_k 8 \
    --adapter_type LoRA \
    --use_gate \
    --use_mlm \
    --dataloader_num_workers 0 > finetune_Scientific.log 2>&1 &
