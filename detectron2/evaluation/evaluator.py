# Copyright (c) Facebook, Inc. and its affiliates.
import datetime
import os
import logging
import time
import math
import statistics
from collections import OrderedDict
from contextlib import contextmanager
import torch

import detectron2.utils.comm as comm
from detectron2.utils.comm import get_world_size, is_main_process
from detectron2.utils.logger import log_every_n_seconds


class DatasetEvaluator:
    """
    Base class for a dataset evaluator.

    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.

    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        pass

    def process(self, inputs, outputs):
        """
        Process the pair of inputs and outputs.
        If they contain batches, the pairs can be consumed one-by-one using `zip`:

        .. code-block:: python

            for input_, output in zip(inputs, outputs):
                # do evaluation on single input/output pair
                ...

        Args:
            inputs (list): the inputs that's used to call the model.
            outputs (list): the return value of `model(inputs)`
        """
        pass

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:

                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        pass


class DatasetEvaluators(DatasetEvaluator):
    """
    Wrapper class to combine multiple :class:`DatasetEvaluator` instances.

    This class dispatches every evaluation call to
    all of its :class:`DatasetEvaluator`.
    """

    def __init__(self, evaluators):
        """
        Args:
            evaluators (list): the evaluators to combine.
        """
        super().__init__()
        self._evaluators = evaluators

    def reset(self):
        for evaluator in self._evaluators:
            evaluator.reset()

    def process(self, inputs, outputs):
        for evaluator in self._evaluators:
            evaluator.process(inputs, outputs)

    def evaluate(self):
        results = OrderedDict()
        for evaluator in self._evaluators:
            result = evaluator.evaluate()
            if is_main_process() and result is not None:
                for k, v in result.items():
                    assert (
                        k not in results
                    ), "Different evaluators produce results with the same key {}".format(k)
                    results[k] = v
        return results


def inference_on_dataset(model, data_loader, evaluator):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.forward` accurately.
    The model will be used in eval mode.

    Args:
        model (nn.Module): a module which accepts an object from
            `data_loader` and returns some outputs. It will be temporarily set to `eval` mode.

            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator (DatasetEvaluator): the evaluator to run. Use `None` if you only want
            to benchmark, but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} images".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_compute_time = 0
    with inference_context(model), torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0

            start_compute_time = time.perf_counter()
            outputs = model(inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            evaluator.process(inputs, outputs)

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Inference done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results



def itd_inference_on_dataset(cfg, model, data_loader, evaluator_tmp, evaluator_worst, evaluator_median, evaluator_best, valid_combos):
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} images".format(len(data_loader)))
    total = len(data_loader)

    # Check if evaluators are None
    if evaluator_tmp is None:
        # create a no-op evaluator
        evaluator_tmp = DatasetEvaluators([])
    evaluator_tmp.reset()
    if evaluator_worst is None:
        # create a no-op evaluator
        evaluator_worst = DatasetEvaluators([])
    evaluator_worst.reset()
    if evaluator_median is None:
        # create a no-op evaluator
        evaluator_median = DatasetEvaluators([])
    evaluator_median.reset()
    if evaluator_best is None:
        # create a no-op evaluator
        evaluator_best = DatasetEvaluators([])
    evaluator_best.reset()

    # Initialize counts tensors
    num_stride_dyn_blocks = sum([1 if x[0] == 1 else 0 for x in cfg.MODEL.ITD_BACKBONE.STRIDE_CONFIG])
    num_dilation_dyn_blocks = sum([1 if x[0] == 1 else 0 for x in cfg.MODEL.ITD_BACKBONE.DILATION_CONFIG])
    num_ksize_dyn_blocks = sum([1 if x[0] == 1 else 0 for x in cfg.MODEL.ITD_BACKBONE.KSIZE_CONFIG])
    num_stride_options = len(cfg.MODEL.ITD_BACKBONE.STRIDE_OPTIONS)
    num_dilation_options = len(cfg.MODEL.ITD_BACKBONE.DILATION_OPTIONS)
    num_ksize_options = len(cfg.MODEL.ITD_BACKBONE.KSIZE_OPTIONS)
    # BEST
    stride_choice_counts_best = torch.zeros(num_stride_dyn_blocks, num_stride_options, dtype=torch.int64, device='cuda')
    dilation_choice_counts_best = torch.zeros(num_dilation_dyn_blocks, num_dilation_options, dtype=torch.int64, device='cuda')
    ksize_choice_counts_best = torch.zeros(num_ksize_dyn_blocks, num_ksize_options, dtype=torch.int64, device='cuda')
    # MEDIAN
    stride_choice_counts_median = torch.zeros(num_stride_dyn_blocks, num_stride_options, dtype=torch.int64, device='cuda')
    dilation_choice_counts_median = torch.zeros(num_dilation_dyn_blocks, num_dilation_options, dtype=torch.int64, device='cuda')
    ksize_choice_counts_median = torch.zeros(num_ksize_dyn_blocks, num_ksize_options, dtype=torch.int64, device='cuda')
    # WORST
    stride_choice_counts_worst = torch.zeros(num_stride_dyn_blocks, num_stride_options, dtype=torch.int64, device='cuda')
    dilation_choice_counts_worst = torch.zeros(num_dilation_dyn_blocks, num_dilation_options, dtype=torch.int64, device='cuda')
    ksize_choice_counts_worst = torch.zeros(num_ksize_dyn_blocks, num_ksize_options, dtype=torch.int64, device='cuda')

    # Initialize truth log
    truth_log = {}

    with inference_context(model), torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            # Extract img_ids
            img_ids = [input_dict["image_id"] for input_dict in inputs]
            # Initialize truth log entry
            truth_log[img_ids[0]] = []
            #print("inputs:", inputs)
            #print("img_ids:", img_ids)
            # Initialize empty summaries
            ap_summary = []
            results_summary = []
            # Loop over all valid combos
            for config_combo in valid_combos:
                # Forward w/ current config_combo
                processed_results, loss_dict = model(inputs, config_combo)
                # Record current results to results_summary
                results_summary.append(processed_results)
                #print("\nconfig_combo:", config_combo)
                #print("processed_results:", processed_results)
                # Synchronize all workers after forward
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                # Process the results in the tmp evaluator
                evaluator_tmp.process(inputs, processed_results)
                # Call evaluate (quiet)
                results_tmp = evaluator_tmp.evaluate(img_ids=img_ids, quiet=True)
                #print("\nresults_tmp:", results_tmp)
                curr_ap = results_tmp['bbox']['AP']
                # Check for / correct NaN values
                if math.isnan(curr_ap):
                    curr_ap = 0
                #print("curr_ap:", curr_ap)
                # Add AP to summary
                ap_summary.append(curr_ap)
                # Append to truth log
                curr_losses = {k: l.item() for k, l in loss_dict.items()}
                truth_log[img_ids[0]].append([curr_ap, curr_losses])
                # Reset tmp evaluator
                evaluator_tmp.reset()

            # Convert ap_summary to tensor
            ap_summary = torch.tensor(ap_summary, device='cuda').view(1,-1)
            # Find indices for the combos that lead to highest, median, and lowest AP
            min_ap_val, min_ap_ind = torch.min(ap_summary, dim=1)
            median_ap_val, median_ap_ind = torch.median(ap_summary, dim=1)
            max_ap_val, max_ap_ind = torch.max(ap_summary, dim=1)
            #print("\n\nap_summary:", ap_summary)
            #print("min_ind:", min_ap_ind)#, results_summary[min_loss_ind.item()])
            #print("median_ind:", median_ap_ind)#, results_summary[median_loss_ind.item()])
            #print("max_ind:", max_ap_ind)#, results_summary[max_loss_ind.item()])

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

            # Synchronize all workers after forward
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            # Process the results in the evaluators
            evaluator_worst.process(inputs, results_summary[min_ap_ind.item()])
            evaluator_median.process(inputs, results_summary[median_ap_ind.item()])
            evaluator_best.process(inputs, results_summary[max_ap_ind.item()])
            
            # Occasionally log
            if idx % 10 == 0:
                logger.info("Exhaustive ITD inference in progress: [{} / {}]".format(idx+1, total))


    # Evaluate worst
    logger.info("\n\n*** WORST ***")
    results_worst = evaluator_worst.evaluate()
    # Evaluate median
    logger.info("\n\n*** MEDIAN ***")
    results_median = evaluator_median.evaluate()
    # Evaluate best
    logger.info("\n\n*** BEST ***")
    results_best = evaluator_best.evaluate()


    # If distributed, accumulate counts from all devices
    if num_devices > 1:
        # Synchronize with a barrier
        comm.synchronize()
        # Gather counts tensors from each device into lists
        all_stride_choice_counts_worst = comm.gather(stride_choice_counts_worst, dst=0)
        all_stride_choice_counts_median = comm.gather(stride_choice_counts_median, dst=0)
        all_stride_choice_counts_best = comm.gather(stride_choice_counts_best, dst=0)
        all_dilation_choice_counts_worst = comm.gather(dilation_choice_counts_worst, dst=0)
        all_dilation_choice_counts_median = comm.gather(dilation_choice_counts_median, dst=0)
        all_dilation_choice_counts_best = comm.gather(dilation_choice_counts_best, dst=0)
        all_ksize_choice_counts_worst = comm.gather(ksize_choice_counts_worst, dst=0)
        all_ksize_choice_counts_median = comm.gather(ksize_choice_counts_median, dst=0)
        all_ksize_choice_counts_best = comm.gather(ksize_choice_counts_best, dst=0)
        # Gather truth_log
        all_truth_logs = comm.gather(truth_log, dst=0)

        if comm.is_main_process():
            # Move all tensors onto cpu
            all_stride_choice_counts_worst = [t.to('cpu') for t in all_stride_choice_counts_worst]
            all_stride_choice_counts_median = [t.to('cpu') for t in all_stride_choice_counts_median]
            all_stride_choice_counts_best = [t.to('cpu') for t in all_stride_choice_counts_best]
            all_dilation_choice_counts_worst = [t.to('cpu') for t in all_dilation_choice_counts_worst]
            all_dilation_choice_counts_median = [t.to('cpu') for t in all_dilation_choice_counts_median]
            all_dilation_choice_counts_best = [t.to('cpu') for t in all_dilation_choice_counts_best]
            all_ksize_choice_counts_worst = [t.to('cpu') for t in all_ksize_choice_counts_worst]
            all_ksize_choice_counts_median = [t.to('cpu') for t in all_ksize_choice_counts_median]
            all_ksize_choice_counts_best = [t.to('cpu') for t in all_ksize_choice_counts_best]
            # Sum counts from each device
            total_stride_choice_counts_worst = torch.sum(torch.cat(all_stride_choice_counts_worst, dim=0), dim=0, keepdim=True)
            total_stride_choice_counts_median = torch.sum(torch.cat(all_stride_choice_counts_median, dim=0), dim=0, keepdim=True)
            total_stride_choice_counts_best = torch.sum(torch.cat(all_stride_choice_counts_best, dim=0), dim=0, keepdim=True)
            total_dilation_choice_counts_worst = torch.sum(torch.cat(all_dilation_choice_counts_worst, dim=0), dim=0, keepdim=True)
            total_dilation_choice_counts_median = torch.sum(torch.cat(all_dilation_choice_counts_median, dim=0), dim=0, keepdim=True)
            total_dilation_choice_counts_best = torch.sum(torch.cat(all_dilation_choice_counts_best, dim=0), dim=0, keepdim=True)
            total_ksize_choice_counts_worst = torch.sum(torch.cat(all_ksize_choice_counts_worst, dim=0), dim=0, keepdim=True)
            total_ksize_choice_counts_median = torch.sum(torch.cat(all_ksize_choice_counts_median, dim=0), dim=0, keepdim=True)
            total_ksize_choice_counts_best = torch.sum(torch.cat(all_ksize_choice_counts_best, dim=0), dim=0, keepdim=True)
            # Combine truth_logs
            total_truth_log = {}
            for tl in all_truth_logs:
                total_truth_log.update(tl)

    # If not distributed, just rename to make consistent
    else:
        total_stride_choice_counts_worst = stride_choice_counts_worst.to('cpu')
        total_stride_choice_counts_median = stride_choice_counts_median.to('cpu')
        total_stride_choice_counts_best = stride_choice_counts_best.to('cpu')
        total_dilation_choice_counts_worst = dilation_choice_counts_worst.to('cpu')
        total_dilation_choice_counts_median = dilation_choice_counts_median.to('cpu')
        total_dilation_choice_counts_best = dilation_choice_counts_best.to('cpu')
        total_ksize_choice_counts_worst = ksize_choice_counts_worst.to('cpu')
        total_ksize_choice_counts_median = ksize_choice_counts_median.to('cpu')
        total_ksize_choice_counts_best = ksize_choice_counts_best.to('cpu')
        total_truth_log = truth_log

    # Display total counts
    if num_devices == 1 or comm.is_main_process():
        print("\n\nChoice Counts (WORST):")
        for i in range(total_stride_choice_counts_worst.shape[0]):
            print("Stride Block:   ", i, total_stride_choice_counts_worst[i])
        for i in range(total_dilation_choice_counts_worst.shape[0]):
            print("Dilation Block: ", i, total_dilation_choice_counts_worst[i])
        for i in range(total_ksize_choice_counts_worst.shape[0]):
            print("Ksize Block:    ", i, total_ksize_choice_counts_worst[i])

        print("\n\nChoice Counts (MEDIAN):")
        for i in range(total_stride_choice_counts_median.shape[0]):
            print("Stride Block:   ", i, total_stride_choice_counts_median[i])
        for i in range(total_dilation_choice_counts_median.shape[0]):
            print("Dilation Block: ", i, total_dilation_choice_counts_median[i])
        for i in range(total_ksize_choice_counts_median.shape[0]):
            print("Ksize Block:    ", i, total_ksize_choice_counts_median[i])

        print("\n\nChoice Counts (BEST):")
        for i in range(total_stride_choice_counts_best.shape[0]):
            print("Stride Block:   ", i, total_stride_choice_counts_best[i])
        for i in range(total_dilation_choice_counts_best.shape[0]):
            print("Dilation Block: ", i, total_dilation_choice_counts_best[i])
        for i in range(total_ksize_choice_counts_best.shape[0]):
            print("Ksize Block:    ", i, total_ksize_choice_counts_best[i])
        print("\n\n")

        # Save truth log to file
        truth_log_filepath = os.path.join(cfg.OUTPUT_DIR, "truth_log.pth")
        print("Saving truth_log to: {} ".format(truth_log_filepath))
        torch.save(total_truth_log, truth_log_filepath)


    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results_worst is None:
        results_worst = {}
    if results_median is None:
        results_median = {}
    if results_best is None:
        results_best = {}
    return results_worst, results_median, results_best



@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)
