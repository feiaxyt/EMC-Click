#!/usr/bin/env bash
CONFIG=models/emcclick/hrnet18s_att_cclvis.py
EXP_NAME=hrnet18s_att_cclvis
nGPUS=4
nBS=64
nWORKERS=4
PORT=`expr $RANDOM + 5000`
echo $PORT
python -m torch.distributed.launch --nproc_per_node=$nGPUS --master_port=$PORT \
    train.py $CONFIG \
    --ngpus=$nGPUS \
    --workers=$nWORKERS \
    --batch-size=$nBS \
    --exp-name=$EXP_NAME