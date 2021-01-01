#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Detectron2 training script with a plain training loop.

This script reads a given config file and runs the training or evaluation.
It is an entry point that is able to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as a library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.

Compared to "train_net.py", this script supports fewer default features.
It also includes fewer abstraction, therefore is easier to add custom logic.
"""

import logging
import os
import random
import itertools
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, ITDDetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    itd_inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)

logger = logging.getLogger("detectron2")


############################################################################
### GET EVALUATOR
############################################################################
def get_evaluator(cfg, dataset_name, output_folder=None, quiet=False):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None and not quiet:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        assert (
            torch.cuda.device_count() >= comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        assert (
            torch.cuda.device_count() >= comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesSemSegEvaluator(dataset_name)
    if evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    if evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, cfg, True, output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)



############################################################################
### GET VALID OPTION COMBOS
############################################################################
def get_valid_combos(cfg):
    valid_stride_combos = get_valid_stride_combos(cfg)
    valid_dilation_combos = get_valid_dilation_combos(cfg)
    valid_ksize_combos = get_valid_ksize_combos(cfg)
    valid_combos = list(itertools.product(valid_stride_combos, valid_dilation_combos, valid_ksize_combos))
    return valid_combos


def get_valid_stride_combos(cfg):
    stride_config = cfg.MODEL.ITD_BACKBONE.STRIDE_CONFIG
    stride_options = cfg.MODEL.ITD_BACKBONE.STRIDE_OPTIONS
    downsample_bounds = cfg.MODEL.ITD_BACKBONE.DOWNSAMPLE_BOUNDS
    # Initialize counts tensor
    num_dyn_blocks = sum([1 if x[0] == 1 else 0 for x in stride_config])
    num_stride_options = len(stride_options)
    stride_options_scales = [[1/x[1][0], 1/x[1][1]] if x[0] else x[1] for x in stride_options]
    # Create list of all possible stride options
    option_list = list(range(num_stride_options))
    all_combos = list(itertools.product(option_list, repeat=num_dyn_blocks))
    valid_combos = []
    # Trim stride options that are invalid due to bounds
    for i in range(len(all_combos)):
        valid = True
        curr_downsample = [4, 4]  # [dH, dW] Stem downsamples H and W by 4x
        adaptive_idx = 0
        # Iterate over network configs to check downsample rate
        for layer_idx in range(len(stride_config)):
            # If the curr layer is adaptive
            if stride_config[layer_idx][0] == 1:
                stride = stride_options_scales[all_combos[i][adaptive_idx]]
                curr_downsample = [s1*s2 for s1, s2 in zip(curr_downsample, stride)]
                adaptive_idx += 1 
            # If the curr layer is NOT adaptive
            else:
                stride_option_idx = stride_config[layer_idx][1]
                stride = stride_options_scales[stride_option_idx]
                curr_downsample = [s1*s2 for s1, s2 in zip(curr_downsample, stride)]
            # Check if curr_downsample is now out of bounds
            curr_bounds = downsample_bounds[layer_idx]
            if curr_downsample[0] > curr_bounds[0] or curr_downsample[1] > curr_bounds[0] or curr_downsample[0] < curr_bounds[1] or curr_downsample[1] < curr_bounds[1]:
                valid = False
                break   # Out of bounds, do NOT consider this stride combo
        if valid:
            valid_combos.append(all_combos[i])
    return valid_combos


def get_valid_dilation_combos(cfg):
    dilation_config = cfg.MODEL.ITD_BACKBONE.DILATION_CONFIG
    dilation_options = cfg.MODEL.ITD_BACKBONE.DILATION_OPTIONS
    # Initialize counts tensor
    num_dyn_blocks = sum([1 if x[0] == 1 else 0 for x in dilation_config])
    num_dilation_options = len(dilation_options)
    # Create list of all possible options
    option_list = list(range(num_dilation_options))
    all_combos = list(itertools.product(option_list, repeat=num_dyn_blocks))
    return all_combos


def get_valid_ksize_combos(cfg):
    ksize_config = cfg.MODEL.ITD_BACKBONE.KSIZE_CONFIG
    ksize_options = cfg.MODEL.ITD_BACKBONE.KSIZE_OPTIONS
    # Initialize counts tensor
    num_dyn_blocks = sum([1 if x[0] == 1 else 0 for x in ksize_config])
    num_ksize_options = len(ksize_options)
    # Create list of all possible options
    option_list = list(range(num_ksize_options))
    all_combos = list(itertools.product(option_list, repeat=num_dyn_blocks))
    return all_combos



############################################################################
### DO TEST
############################################################################
def do_test(cfg, model, valid_combos):
    results_worst = OrderedDict()
    results_median = OrderedDict()
    results_best = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        # Initialize data_loader object
        #data_loader = build_detection_test_loader(cfg, dataset_name, itd=True, mini=True)
        data_loader = build_detection_test_loader(cfg, dataset_name, itd=True)
        # Initialize 3 separate but identical evaluators
        evaluator_tmp = get_evaluator(cfg, dataset_name, quiet=True)
        evaluator_worst = get_evaluator(cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name, "worst"))
        evaluator_median = get_evaluator(cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name, "median"))
        evaluator_best = get_evaluator(cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name, "best"))
        # Run inference and evaluate
        results_worst_i, results_median_i, results_best_i = itd_inference_on_dataset(cfg, model, data_loader, evaluator_tmp, evaluator_worst, evaluator_median, evaluator_best, valid_combos)
        # Log results in respective OrderedDict
        results_worst[dataset_name] = results_worst_i
        results_median[dataset_name] = results_median_i
        results_best[dataset_name] = results_best_i
        # Print results
        if comm.is_main_process():
            logger.info("\n\n[WORST] Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_worst_i)
            logger.info("\n\n[MEDIAN] Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_median_i)
            logger.info("\n\n[BEST] Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_best_i)
    # Handle length=1 case
    if len(results_worst) == 1:
        results_worst = list(results_worst.values())[0]
    if len(results_median) == 1:
        results_median = list(results_median.values())[0]
    if len(results_worst) == 1:
        results_best = list(results_best.values())[0]
    return results_worst, results_median, results_best



############################################################################
### DO TRAIN
############################################################################
def do_train(cfg, model, valid_combos, resume=False):
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = ITDDetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )

    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = (
        [
            CommonMetricPrinter(max_iter),
            JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(cfg.OUTPUT_DIR),
        ]
        if comm.is_main_process()
        else []
    )

    # compared to "train_net.py", we do not support accurate timing and
    # precise BN here, because they are not trivial to implement in a small training loop
    data_loader = build_detection_train_loader(cfg)
    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            storage.iter = iteration

            #print("\n\niteration:", iteration)
            #for i in range(len(data)):
            #    for k, v in data[i].items():
            #        print(k, v)
            #        if k == "image":
            #            print(v.shape)

            config_combo = select_config_combo(valid_combos)
            loss_dict = model(data, config_combo)
            losses = sum(loss_dict.values())

            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            if (
                cfg.TEST.EVAL_PERIOD > 0
                and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
                and iteration != max_iter - 1
            ):
                do_test(cfg, model, valid_combos)
                # Compared to "train_net.py", the test results are not dumped to EventStorage
                comm.synchronize()

            if iteration - start_iter > 5 and (
                (iteration + 1) % 20 == 0 or iteration == max_iter - 1
            ):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)



############################################################################
### DO TRAIN
############################################################################
def select_config_combo(valid_combos):
    choice = random.choice(valid_combos)
    return choice



############################################################################
### SETUP
############################################################################
def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Expand user $HOME in cfg.MODEL.WEIGHTS path
    cfg.MODEL.WEIGHTS = cfg.MODEL.WEIGHTS.replace("~", os.path.expanduser("~"))
    cfg.freeze()
    default_setup(
        cfg, args
    )  # if you don't like any of the default setup, write your own setup code
    return cfg



############################################################################
### MAIN
############################################################################
def main(args):
    cfg = setup(args)
    valid_combos = get_valid_combos(cfg)
    #valid_combos = [((3, 0, 0, 0), (), ())]
    #valid_combos = [((), (), (3, 3, 3, 3))]
    logger.info("Valid Combos: {} {}".format(valid_combos, len(valid_combos)))

    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    logger.info("Param Count: {}".format(sum(p.numel() for p in model.parameters())))
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return do_test(cfg, model, valid_combos)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )

    do_train(cfg, model, valid_combos, resume=args.resume)
    return do_test(cfg, model, valid_combos)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
