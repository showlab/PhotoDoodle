export MODEL_DIR="./PhotoDoodle_Pretrain" # you may need to modity this in order to train your own model
export OUTPUT_DIR="outputs/sksedgeeffect"
export CONFIG="./default_config.yaml"
export TRAIN_DATA="data/sksedgeeffect/meta.jsonl"
export LOG_PATH="$OUTPUT_DIR/log"

CUDA_VISIBLE_DEVICES=1 accelerate launch --config_file $CONFIG train_lora_flux_pe.py \
    --pretrained_model_name_or_path $MODEL_DIR \
    --width 512 \
    --height 768 \
    --source_column="source" \
    --target_column="target" \
    --caption_column="text" \
    --output_dir=$OUTPUT_DIR \
    --logging_dir=$LOG_PATH \
    --mixed_precision="bf16" \
    --train_data_dir=$TRAIN_DATA \
    --rank=128 \
    --learning_rate=1e-4 \
    --train_batch_size=1 \
    --num_validation_images=2 \
    --validation_image "1.png" \
    --validation_prompt "add yellow flames to the cat by sksedgeeffect" \
    --num_train_epochs=10000 \
    --validation_steps=2000 \
    --checkpointing_steps=2000 \
