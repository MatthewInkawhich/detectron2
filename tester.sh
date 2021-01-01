#!/bin/bash

#CUDA_VISIBLE_DEVICES=1 ./tools/itd_main.py --num-gpus 1 --config-file "configs/ITD-COCO-Detection/baseline_itdretinanet_R50_dup.yaml" --eval-only MODEL.WEIGHTS ./out/ITD-COCO-Detection/baseline_itdretinanet_R50_dup/model_final.pth

#./tools/itd_main.py --num-gpus 4 --config-file "configs/ITD-COCO-Detection/baseline_itdretinanet_R50.yaml" --eval-only MODEL.WEIGHTS ./out/ITD-COCO-Detection/baseline_itdretinanet_R50/model_final.pth


#CUDA_VISIBLE_DEVICES=1 ./tools/itd_main.py --num-gpus 1 --config-file "configs/ITD-COCO-Detection/strides/retinanet_R50_A.yaml" --eval-only MODEL.WEIGHTS ./out/ITD-COCO-Detection/baseline_itdretinanet_R50/model_final.pth
#CUDA_VISIBLE_DEVICES=1 ./tools/itd_main.py --num-gpus 1 --config-file "configs/ITD-COCO-Detection/ksizes/retinanet_R50_ABCD.yaml" --eval-only MODEL.WEIGHTS ./out/ITD-COCO-Detection/baseline_itdretinanet_R50/model_final.pth
#CUDA_VISIBLE_DEVICES=1 ./tools/itd_main.py --num-gpus 1 --config-file "configs/ITD-COCO-Detection/ksizes/play.yaml" --eval-only MODEL.WEIGHTS ./out/ITD-COCO-Detection/baseline_itdretinanet_R50/model_final.pth

#./tools/itd_main.py --num-gpus 4 --config-file "configs/ITD-COCO-Detection/strides/retinanet_R50_A.yaml" --eval-only MODEL.WEIGHTS ./out/ITD-COCO-Detection/baseline_itdretinanet_R50/model_final.pth
#./tools/itd_main.py --num-gpus 4 --config-file "configs/ITD-COCO-Detection/itd_dilation_A.yaml" --eval-only MODEL.WEIGHTS ./out/ITD-COCO-Detection/baseline_itdretinanet_R50/model_final.pth
#./tools/itd_main.py --num-gpus 4 --config-file "configs/ITD-COCO-Detection/itd_ksize_A.yaml" --eval-only MODEL.WEIGHTS ./out/ITD-COCO-Detection/baseline_itdretinanet_R50/model_final.pth
./tools/itd_main.py --num-gpus 4 --config-file "configs/ITD-COCO-Detection/ksizes/play.yaml" --eval-only MODEL.WEIGHTS ./out/ITD-COCO-Detection/baseline_itdretinanet_R50/model_final.pth
