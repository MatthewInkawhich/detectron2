#!/usr/bin/env python
"""
Use this script to analyze the truth_log files we save during testing.
"""

import logging
import os
import csv
import random
import itertools
import argparse
from collections import OrderedDict
import torch
from detectron2.config import get_cfg


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
### SETUP
############################################################################
def setup(config_file):
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    # Expand user $HOME in cfg.MODEL.WEIGHTS path
    cfg.MODEL.WEIGHTS = cfg.MODEL.WEIGHTS.replace("~", os.path.expanduser("~"))
    cfg.freeze()
    return cfg



############################################################################
### MAIN
############################################################################
def main():
    parser = argparse.ArgumentParser(description="Log Analyzer")
    parser.add_argument("config_file", type=str)
    parser.add_argument("log_file", type=str)
    args = parser.parse_args()

    # Load configs
    cfg = setup(args.config_file)
    valid_combos = get_valid_combos(cfg)
    print("Valid Combos: {} {}".format(valid_combos, len(valid_combos)))
    variant = args.config_file.split('_')[-1].split('.yaml')[0]
    
    # Load truth_log
    truth_log = torch.load(args.log_file, map_location='cpu')


    # Initialize counts tensors
    num_stride_dyn_blocks = sum([1 if x[0] == 1 else 0 for x in cfg.MODEL.ITD_BACKBONE.STRIDE_CONFIG])
    num_dilation_dyn_blocks = sum([1 if x[0] == 1 else 0 for x in cfg.MODEL.ITD_BACKBONE.DILATION_CONFIG])
    num_ksize_dyn_blocks = sum([1 if x[0] == 1 else 0 for x in cfg.MODEL.ITD_BACKBONE.KSIZE_CONFIG])
    num_stride_options = len(cfg.MODEL.ITD_BACKBONE.STRIDE_OPTIONS)
    num_dilation_options = len(cfg.MODEL.ITD_BACKBONE.DILATION_OPTIONS)
    num_ksize_options = len(cfg.MODEL.ITD_BACKBONE.KSIZE_OPTIONS)
    # BEST
    stride_choice_counts_best = torch.zeros(num_stride_dyn_blocks, num_stride_options, dtype=torch.int64)
    dilation_choice_counts_best = torch.zeros(num_dilation_dyn_blocks, num_dilation_options, dtype=torch.int64)
    ksize_choice_counts_best = torch.zeros(num_ksize_dyn_blocks, num_ksize_options, dtype=torch.int64)
    # MEDIAN
    stride_choice_counts_median = torch.zeros(num_stride_dyn_blocks, num_stride_options, dtype=torch.int64)
    dilation_choice_counts_median = torch.zeros(num_dilation_dyn_blocks, num_dilation_options, dtype=torch.int64)
    ksize_choice_counts_median = torch.zeros(num_ksize_dyn_blocks, num_ksize_options, dtype=torch.int64)
    # WORST
    stride_choice_counts_worst = torch.zeros(num_stride_dyn_blocks, num_stride_options, dtype=torch.int64)
    dilation_choice_counts_worst = torch.zeros(num_dilation_dyn_blocks, num_dilation_options, dtype=torch.int64)
    ksize_choice_counts_worst = torch.zeros(num_ksize_dyn_blocks, num_ksize_options, dtype=torch.int64)

    # Count selections over the log
    for curr_id, curr_summary in truth_log.items():
        # Put all APs into a list
        ap_summary = []
        for combo_idx in range(len(curr_summary)):
            ap_summary.append(curr_summary[combo_idx][0])

        # Convert ap_summary to tensor
        ap_summary = torch.tensor(ap_summary).view(1,-1)
        # Find indices for the combos that lead to highest, median, and lowest AP
        min_ap_val, min_ap_ind = torch.min(ap_summary, dim=1)
        median_ap_val, median_ap_ind = torch.median(ap_summary, dim=1)
        max_ap_val, max_ap_ind = torch.max(ap_summary, dim=1)

        # Record selections
        # WORST
        for db_i in range(len(valid_combos[min_ap_ind.item()][0])):
            stride_choice_counts_worst[db_i][valid_combos[min_ap_ind.item()][0][db_i]] += 1
        for db_i in range(len(valid_combos[min_ap_ind.item()][1])):
            dilation_choice_counts_worst[db_i][valid_combos[min_ap_ind.item()][1][db_i]] += 1
        for db_i in range(len(valid_combos[min_ap_ind.item()][2])):
            ksize_choice_counts_worst[db_i][valid_combos[min_ap_ind.item()][2][db_i]] += 1
        # MEDIAN
        for db_i in range(len(valid_combos[median_ap_ind.item()][0])):
            stride_choice_counts_median[db_i][valid_combos[median_ap_ind.item()][0][db_i]] += 1
        for db_i in range(len(valid_combos[median_ap_ind.item()][1])):
            dilation_choice_counts_median[db_i][valid_combos[median_ap_ind.item()][1][db_i]] += 1
        for db_i in range(len(valid_combos[median_ap_ind.item()][2])):
            ksize_choice_counts_median[db_i][valid_combos[median_ap_ind.item()][2][db_i]] += 1
        # BEST
        for db_i in range(len(valid_combos[max_ap_ind.item()][0])):
            stride_choice_counts_best[db_i][valid_combos[max_ap_ind.item()][0][db_i]] += 1
        for db_i in range(len(valid_combos[max_ap_ind.item()][1])):
            dilation_choice_counts_best[db_i][valid_combos[max_ap_ind.item()][1][db_i]] += 1
        for db_i in range(len(valid_combos[max_ap_ind.item()][2])):
            ksize_choice_counts_best[db_i][valid_combos[max_ap_ind.item()][2][db_i]] += 1

    # Display selection counts and build output_lists
    output_lists = []
    print("\n\nChoice Counts (WORST):")
    output_lists.append(["WORST"])
    for i in range(stride_choice_counts_worst.shape[0]):
        print("Stride Block:   ", i, stride_choice_counts_worst[i])
        output_lists.append(stride_choice_counts_worst[i].tolist())
    for i in range(dilation_choice_counts_worst.shape[0]):
        print("Dilation Block: ", i, dilation_choice_counts_worst[i])
        output_lists.append(dilation_choice_counts_worst[i].tolist())
    for i in range(ksize_choice_counts_worst.shape[0]):
        print("Ksize Block:    ", i, ksize_choice_counts_worst[i])
        output_lists.append(ksize_choice_counts_worst[i].tolist())

    print("\n\nChoice Counts (MEDIAN):")
    output_lists.append(["MEDIAN"])
    for i in range(stride_choice_counts_median.shape[0]):
        print("Stride Block:   ", i, stride_choice_counts_median[i])
        output_lists.append(stride_choice_counts_median[i].tolist())
    for i in range(dilation_choice_counts_median.shape[0]):
        print("Dilation Block: ", i, dilation_choice_counts_median[i])
        output_lists.append(dilation_choice_counts_median[i].tolist())
    for i in range(ksize_choice_counts_median.shape[0]):
        print("Ksize Block:    ", i, ksize_choice_counts_median[i])
        output_lists.append(ksize_choice_counts_median[i].tolist())

    print("\n\nChoice Counts (BEST):")
    output_lists.append(["BEST"])
    for i in range(stride_choice_counts_best.shape[0]):
        print("Stride Block:   ", i, stride_choice_counts_best[i])
        output_lists.append(stride_choice_counts_best[i].tolist())
    for i in range(dilation_choice_counts_best.shape[0]):
        print("Dilation Block: ", i, dilation_choice_counts_best[i])
        output_lists.append(dilation_choice_counts_best[i].tolist())
    for i in range(ksize_choice_counts_best.shape[0]):
        print("Ksize Block:    ", i, ksize_choice_counts_best[i])
        output_lists.append(ksize_choice_counts_best[i].tolist())
    print("\n")
    print("DONE:", variant)
    output_lists.append([variant])
    output_lists.append([])
    output_lists.append([])
    print("\n\n")

    with open('tmp.csv', mode='a') as tmpfile:
        tmp_writer = csv.writer(tmpfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(output_lists)):
            print(output_lists[i])
            tmp_writer.writerow(output_lists[i])

if __name__ == "__main__":
    main()
