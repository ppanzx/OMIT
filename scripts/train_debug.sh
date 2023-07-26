export CUDA_VISIBLE_DEVICES=0 

DATASET_NAME='coco'
DATASET_ROOT="/media/panzx/新加卷/PanZhengxin/datasets/CrossModalRetrieval"
DATA_PATH=${DATASET_ROOT}/${DATASET_NAME}
VOCAB_PATH=${DATASET_ROOT}/"vocab"
BUTD_WEIGHT_PATH=${DATASET_ROOT}/"weights"

SAVE_PATH='./scripts/ckpts/coco_CHAN_butd_ft'
# python3 train.py \
#   --data_path=${DATA_PATH} --data_name=${DATASET_NAME}  --vocab_path=${VOCAB_PATH}\
#   --logger_name=${SAVE_PATH}/log --model_name=${SAVE_PATH} \
#   --num_epochs=25 --lr_update=15 --learning_rate=5e-4 --precomp_enc_type=backbone  --workers=16 --backbone_source=detector \
#   --vse_mean_warmup_epochs=1 --backbone_warmup_epochs=0 --embedding_warmup_epochs=1  --optim=adam --backbone_lr_factor=0.01 \
#   --input_scale_factor=2.0  --backbone_path=${BUTD_WEIGHT_PATH}/original_updown_backbone.pth --log_step=200 --batch_size=60 \
#   --coding_type=VHACoding --alpha=0.1 --pooling_type=LSEPooling --belta=0.1 \
#   --drop --wemb_type=glove --criterion=ContrastiveLoss --margin=0.05 \

python3 ./eval.py --dataset=${DATASET_NAME} --model_path=${SAVE_PATH}/model_best.pth --data_path=${DATA_PATH}