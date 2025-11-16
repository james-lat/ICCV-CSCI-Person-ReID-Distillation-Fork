export NUM_GPU=1
export PORT=12355

export CONFIG="configs/ltcc_eva02_l_cloth.yml"
export ROOT="/mnt/SSD2/LTCC"
export DATASET="ltcc"
export OUT_DIR="logs/ltcc/distill_evaB_from_evaL_RUN-1244"
export SEED=1244
export COLOR=26

PYTHONUNBUFFERED=1 stdbuf -oL -eL \
python -u -W ignore -m torch.distributed.launch \
  --nproc_per_node=$NUM_GPU --master_port="$PORT" \
  train_two_step_distillation.py \
    --config_file "$CONFIG" \
    --no-head \
    DATA.ROOT "$ROOT" \
    DATA.DATASET "$DATASET" \
    OUTPUT_DIR "$OUT_DIR" \
    SOLVER.SEED "$SEED" \
    MODEL.NAME "eva02_img_extra_token_base" \
    TRAIN.COLOR_PROFILE "$COLOR" \
    TRAIN.TEACH1 "$DATASET" \
    TRAIN.DIR_TEACH1 "$ROOT" \
    TRAIN.TEACH1_MODEL "eva02_img_extra_token" \
    TRAIN.TEACH1_MODEL_WT "$WT_L" \
    TRAIN.TEACH1_LOAD_AS_IMG True \
    TRAIN.KD_WEIGHT 0.5 \
    DATA.PIN_MEMORY True \
  | stdbuf -oL -eL tee -a "$OUT_DIR/run_$(date +%F_%H-%M-%S).log"
