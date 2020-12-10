# Copyright (c) Facebook, Inc. and its affiliates.
import datetime
import logging
import time
from collections import OrderedDict
from contextlib import contextmanager
import torch

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



def itd_inference_on_dataset(model, data_loader, evaluator_worst, evaluator_median, evaluator_best, valid_combos):
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} images".format(len(data_loader)))
    total = len(data_loader)

    # Check if evaluators are None
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

    with inference_context(model), torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            #print("inputs:", inputs)
            # Initialize empty summaries
            loss_summary = []
            results_summary = []
            # Loop over all valid combos
            for config_combo in valid_combos:
                # Forward w/ current config_combo
                processed_results, loss_dict = model(inputs, config_combo)
                # Compute scalar loss value
                losses = sum(loss_dict.values())
                # Record current loss to loss_summary
                loss_summary.append(losses.view(1,1))
                # Record current results to results_summary
                results_summary.append(processed_results)

            # Convert loss_summary to a tensor
            loss_summary = torch.cat(loss_summary, dim=1)
            # Find indices for the combos that lead to highest, median, and lowest loss
            min_loss_val, min_loss_ind = torch.min(loss_summary, dim=1)
            median_loss_val, median_loss_ind = torch.median(loss_summary, dim=1)
            max_loss_val, max_loss_ind = torch.max(loss_summary, dim=1)
            #print("loss_summary:", loss_summary)
            #print("min_ind:", min_loss_ind)
            #print("median_ind:", median_loss_ind)
            #print("max_ind:", max_loss_ind)


            # Synchronize all workers after forward
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            # Process the results in the evaluators
            evaluator_worst.process(inputs, results_summary[max_loss_ind.item()])
            evaluator_median.process(inputs, results_summary[median_loss_ind.item()])
            evaluator_best.process(inputs, results_summary[min_loss_ind.item()])
            
            # Occasionally log
            if idx % 10 == 0:
                logger.info("Exhaustive ITD inference in progress: [{} / {}]".format(idx+1, total))

    # Call evaluate
    logger.info("\n\n*** WORST ***")
    results_worst = evaluator_worst.evaluate()
    logger.info("\n\n*** MEDIAN ***")
    results_median = evaluator_median.evaluate()
    logger.info("\n\n*** BEST ***")
    results_best = evaluator_best.evaluate()
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
