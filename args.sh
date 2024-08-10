BS=64


DATASET_NAME=$1
PRED=$2
NGPU=1


SIGMA_MAX=80.0
SIGMA_MIN=0.002
SIGMA_DATA=0.5
COV_XY=0


NUM_CH=256
ATTN=32,16,8
SAMPLER=real-uniform # real-uniform??
NUM_RES_BLOCKS=2
USE_16FP=False # True
ATTN_TYPE=flash

DATA_DIR=YOUR_DATASET_PATH
DATASET=edges2handbags
NUM_CH=192
NUM_RES_BLOCKS=3
EXP="test_${NUM_CH}d"
SAVE_ITER=1000
    
if  [[ $PRED == "ve" ]]; then
    EXP+="_ve"
    COND=concat
elif  [[ $PRED == "vp" ]]; then
    EXP+="_vp"
    COND=concat
    BETA_D=2
    BETA_MIN=0.1
    SIGMA_MAX=1
    SIGMA_MIN=0.0001
elif  [[ $PRED == "ve_simple" ]]; then
    EXP+="_ve_simple"
    COND=concat
elif  [[ $PRED == "vp_simple" ]]; then
    EXP+="_vp_simple"
    COND=concat
    BETA_D=2
    BETA_MIN=0.1
    SIGMA_MAX=1
    SIGMA_MIN=0.0001
else
    echo "Not supported"
    exit 1
fi


# BS=192
