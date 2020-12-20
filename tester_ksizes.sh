#!/bin/bash

# Core
./tools/itd_main.py --num-gpus 4 --config-file "configs/ITD-COCO-Detection/ksizes/retinanet_R50_A.yaml" --eval-only MODEL.WEIGHTS ./out/ITD-COCO-Detection/baseline_itdretinanet_R50/model_final.pth
echo; echo 'DONE: A'; echo; echo; echo
./tools/itd_main.py --num-gpus 4 --config-file "configs/ITD-COCO-Detection/ksizes/retinanet_R50_AB.yaml" --eval-only MODEL.WEIGHTS ./out/ITD-COCO-Detection/baseline_itdretinanet_R50/model_final.pth
echo; echo 'DONE: AB'; echo; echo; echo
./tools/itd_main.py --num-gpus 4 --config-file "configs/ITD-COCO-Detection/ksizes/retinanet_R50_ABC.yaml" --eval-only MODEL.WEIGHTS ./out/ITD-COCO-Detection/baseline_itdretinanet_R50/model_final.pth
echo; echo 'DONE: ABC'; echo; echo; echo
./tools/itd_main.py --num-gpus 4 --config-file "configs/ITD-COCO-Detection/ksizes/retinanet_R50_ABCD.yaml" --eval-only MODEL.WEIGHTS ./out/ITD-COCO-Detection/baseline_itdretinanet_R50/model_final.pth
echo; echo 'DONE: ABCD'; echo; echo; echo


# Rest
./tools/itd_main.py --num-gpus 4 --config-file "configs/ITD-COCO-Detection/ksizes/retinanet_R50_B.yaml" --eval-only MODEL.WEIGHTS ./out/ITD-COCO-Detection/baseline_itdretinanet_R50/model_final.pth
echo; echo 'DONE: B'; echo; echo; echo
./tools/itd_main.py --num-gpus 4 --config-file "configs/ITD-COCO-Detection/ksizes/retinanet_R50_C.yaml" --eval-only MODEL.WEIGHTS ./out/ITD-COCO-Detection/baseline_itdretinanet_R50/model_final.pth
echo; echo 'DONE: C'; echo; echo; echo
./tools/itd_main.py --num-gpus 4 --config-file "configs/ITD-COCO-Detection/ksizes/retinanet_R50_D.yaml" --eval-only MODEL.WEIGHTS ./out/ITD-COCO-Detection/baseline_itdretinanet_R50/model_final.pth
echo; echo 'DONE: D'; echo; echo; echo

./tools/itd_main.py --num-gpus 4 --config-file "configs/ITD-COCO-Detection/ksizes/retinanet_R50_AC.yaml" --eval-only MODEL.WEIGHTS ./out/ITD-COCO-Detection/baseline_itdretinanet_R50/model_final.pth
echo; echo 'DONE: AC'; echo; echo; echo
./tools/itd_main.py --num-gpus 4 --config-file "configs/ITD-COCO-Detection/ksizes/retinanet_R50_AD.yaml" --eval-only MODEL.WEIGHTS ./out/ITD-COCO-Detection/baseline_itdretinanet_R50/model_final.pth
echo; echo 'DONE: AD'; echo; echo; echo
./tools/itd_main.py --num-gpus 4 --config-file "configs/ITD-COCO-Detection/ksizes/retinanet_R50_BC.yaml" --eval-only MODEL.WEIGHTS ./out/ITD-COCO-Detection/baseline_itdretinanet_R50/model_final.pth
echo; echo 'DONE: BC'; echo; echo; echo
./tools/itd_main.py --num-gpus 4 --config-file "configs/ITD-COCO-Detection/ksizes/retinanet_R50_BD.yaml" --eval-only MODEL.WEIGHTS ./out/ITD-COCO-Detection/baseline_itdretinanet_R50/model_final.pth
echo; echo 'DONE: BD'; echo; echo; echo
./tools/itd_main.py --num-gpus 4 --config-file "configs/ITD-COCO-Detection/ksizes/retinanet_R50_CD.yaml" --eval-only MODEL.WEIGHTS ./out/ITD-COCO-Detection/baseline_itdretinanet_R50/model_final.pth
echo; echo 'DONE: CD'; echo; echo; echo

./tools/itd_main.py --num-gpus 4 --config-file "configs/ITD-COCO-Detection/ksizes/retinanet_R50_ABD.yaml" --eval-only MODEL.WEIGHTS ./out/ITD-COCO-Detection/baseline_itdretinanet_R50/model_final.pth
echo; echo 'DONE: ABD'; echo; echo; echo
./tools/itd_main.py --num-gpus 4 --config-file "configs/ITD-COCO-Detection/ksizes/retinanet_R50_ACD.yaml" --eval-only MODEL.WEIGHTS ./out/ITD-COCO-Detection/baseline_itdretinanet_R50/model_final.pth
echo; echo 'DONE: ACD'; echo; echo; echo
./tools/itd_main.py --num-gpus 4 --config-file "configs/ITD-COCO-Detection/ksizes/retinanet_R50_BCD.yaml" --eval-only MODEL.WEIGHTS ./out/ITD-COCO-Detection/baseline_itdretinanet_R50/model_final.pth
echo; echo 'DONE: BCD'; echo; echo; echo
