DATASET_NAME='f30k'
DATASET_ROOT="/media/panzx/新加卷/PanZhengxin/datasets/CrossModalRetrieval"
DATA_PATH=${DATASET_ROOT}/${DATASET_NAME}
VOCAB_PATH=${DATASET_ROOT}/"vocab"
BUTD_WEIGHT_PATH=${DATASET_ROOT}/"weights"
# SAVE_PATH='./scripts/ckpts/coco_BUTD_512x512'
SAVE_PATH='./wasserstain/scripts/open_source/f30k_res152_256x256/model_best.pth'

export CUDA_VISIBLE_DEVICES=0 

python3 ./eval.py --dataset=${DATASET_NAME} --model_path=${SAVE_PATH}/checkpoint.pth --data_path=${DATA_PATH} 