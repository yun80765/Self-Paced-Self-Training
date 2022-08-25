#!/bin/sh
network=WRN28_2
dataset=cifar10
num_labeled=400
num_classes=10
epochs=700
holder=20
n=`expr $num_labeled \* $num_classes`
nohup python main.py \
--arch $network --dataset $dataset --num_labeled $num_labeled --epochs $epochs --num_classes $num_classes \
--doParallel --seed 821 --nesterov --weight-decay 0.0005 --percentiles_holder $holder \
--batch_size 512 --lr_rampdown_epochs 750 --classwise_curriculum \
--add_name WRN28_${dataset}_AUG_MIX_SWA_${n}_${holder} --mixup --swa \
>${network}_${dataset}_${n}_${holder}.outs 2>&1 &