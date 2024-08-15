DATASET_NAME=$1
PRED=$2
DEBUG=$3
CKPT=$4


source ./args.sh $DATASET_NAME $PRED

FREQ_SAVE_ITER=20000
NGPU=1
CUDA_VISIBLE_DEVICES=1
mpiexec -n $NGPU python scripts/ddbm_train_mrm.py --exp=$EXP \
 --attention_resolutions $ATTN --class_cond False --use_scale_shift_norm True \
  --dropout 0.1 --ema_rate 0.9999 --batch_size $BS \
   --lr 0.0001 --num_channels $NUM_CH --num_head_channels 64 \
    --num_res_blocks $NUM_RES_BLOCKS --resblock_updown True ${COND:+ --condition_mode="${COND}"} ${MICRO:+ --microbatch="${MICRO}"} \
     --pred_mode=$PRED  --schedule_sampler $SAMPLER ${UNET:+ --unet_type="${UNET}"} \
    --use_fp16 $USE_16FP --attention_type $ATTN_TYPE --weight_decay 0.0 --weight_schedule bridge_karras \
     ${BETA_D:+ --beta_d="${BETA_D}"} ${BETA_MIN:+ --beta_min="${BETA_MIN}"}  \
       ${CH_MULT:+ --channel_mult="${CH_MULT}"} \
      --num_workers=$NGPU  --sigma_data $SIGMA_DATA --sigma_max=$SIGMA_MAX --sigma_min=$SIGMA_MIN \
      --save_interval_for_preemption=$FREQ_SAVE_ITER --save_interval=$SAVE_ITER \
      ${CKPT:+ --resume_checkpoint="${CKPT}"}  ${DEBUG:+ --debug="${DEBUG}"} \
      --data_path $DATA_PATH ${HUMAN_DATA_PATH:+ --human_data_path="${HUMAN_DATA_PATH}"} --load_pose=True --arch=$ARCH --normalize=$NORMALIZE