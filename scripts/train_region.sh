#!/bin/bash

#SBATCH -N 1
#SBATCH -n 5
#SBATCH -M priv
#SBATCH -p priv_para
#SBATCH --gres=gpu:1
#SBATCH --no-requeue

module load nvidia/cuda/10.2

export CUDA_VISIBLE_DEVICES=0

DATASET_NAME='f30k'
DATASET_ROOT="/media/panzx/新加卷/PanZhengxin/datasets/CrossModalRetrieval"
DATA_PATH=${DATASET_ROOT}/${DATASET_NAME}
VOCAB_PATH=${DATASET_ROOT}/"vocab"
BUTD_WEIGHT_PATH=${DATASET_ROOT}/"weights"

## wasserstain with region features
SAVE_PATH='./scripts/runs/wassestain/f30k_region_gru'
echo "Experiment of Wassestain with region featrues save in: "${SAVE_PATH}
python3 ./train.py \
  --data_path=${DATA_PATH} --data_name=${DATASET_NAME} --text_enc_type=bigru \
  --vocab_path=${VOCAB_PATH} --logger_name=${SAVE_PATH}/log --model_name=${SAVE_PATH} \
  --num_epochs=25 --lr_update=15 --learning_rate=5e-4 --precomp_enc_type=mlp --workers=5 \
  --log_step=200 --embed_size=1024 --vse_mean_warmup_epochs=1 --batch_size=128 \
  --aggr_type=ot --alpha=0.05 --belta=3 \
  --mask --wemb_type=glove \
  --criterion=ContrastiveLoss --margin=0.05

python3 ./eval.py --dataset=${DATASET_NAME} --model_path=${SAVE_PATH}/model_best.pth --data_path=${DATA_PATH}
