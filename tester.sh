#!/bin/bash

CUDA_VISIBLE_DEVICES=1 ./tools/itd_main.py --num-gpus 1 --config-file "configs/ITD-COCO-Detection/baseline_itdretinanet_R50_dup.yaml" --eval-only MODEL.WEIGHTS ./out/ITD-COCO-Detection/baseline_itdretinanet_R50_dup/model_0124999.pth
