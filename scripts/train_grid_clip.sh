DATASET_NAME='f30k'
DATASET_ROOT="/home/wufangyu/project/pzx/dataset/CrossModalRetrieval"
DATA_PATH=${DATASET_ROOT}/${DATASET_NAME}
VOCAB_PATH=${DATASET_ROOT}/"vocab"
BUTD_WEIGHT_PATH=${DATASET_ROOT}/"weights"

SAVE_PATH='./scripts/runs/f30k_CHAN_butd_ft'
CUDA_VISIBLE_DEVICES=0 python3 train.py \
  --data_path=${DATA_PATH} --data_name=${DATASET_NAME}  --vocab_path=${VOCAB_PATH} \ 
  --logger_name=${SAVE_PATH}/log --model_name=${SAVE_PATH} --precomp_enc_type=clip --text_enc_type=clip \
  --num_epochs=25 --lr_update=15 --learning_rate=5e-4 --workers=16 --backbone_source=detector \
  --vse_mean_warmup_epochs=1 --backbone_warmup_epochs=0 --embedding_warmup_epochs=1  --optim=adam --backbone_lr_factor=0.01 \
  --input_scale_factor=1.0  --backbone_path=${BUTD_WEIGHT_PATH}/original_updown_backbone.pth --log_step=200 --batch_size=60 \
  --coding_type=VHACoding --alpha=0.1 --pooling_type=LSEPooling --belta=0.1 \
  --mask --visual_mask_ratio=0.2 --wemb_type=glove --criterion=ContrastiveLoss --margin=0.05 \

CUDA_VISIBLE_DEVICES=0 python3 ./eval.py --dataset=${DATASET_NAME} --model_path=${SAVE_PATH}/model_best.pth --data_path=${DATA_PATH}