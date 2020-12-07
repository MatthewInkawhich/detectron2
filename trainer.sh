#!/bin/bash

#./tools/plain_train_net.py --num-gpus 4 --config-file "configs/ITD-COCO-Detection/retinanet_R50.yaml"
#./tools/plain_train_net.py --num-gpus 4 --config-file "configs/ITD-COCO-Detection/retinanet_R50_stride3x3_nf.yaml"

#CUDA_VISIBLE_DEVICES=1 ./tools/itd_train_net.py --num-gpus 1 --config-file "configs/ITD-COCO-Detection/play.yaml"
#CUDA_VISIBLE_DEVICES=1 ./tools/plain_train_net.py --num-gpus 1 --config-file "configs/ITD-COCO-Detection/play_base.yaml"
./tools/itd_train_net.py --num-gpus 4 --config-file "configs/ITD-COCO-Detection/baseline_itdretinanet_R50.yaml"
