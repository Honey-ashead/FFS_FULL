#!/usr/bin python
# Copyright (c) Facebook, Inc. and its affiliates.

import os
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, train_argument_parser, default_setup, launch
from detectron2.evaluation import (
    COCOEvaluator,
    DatasetEvaluators,
    verify_results,
)
from detectron2.config import CfgNode as CN
from detectron2.utils.logger import setup_logger
import torch, numpy, random


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluators = [COCOEvaluator(dataset_name, cfg, True, output_folder)]
        return DatasetEvaluators(evaluators)

    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.

        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.

        Args:
            resume (bool): whether to do resume or not
        """
        self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)
        if resume and self.checkpointer.has_checkpoint():
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration
            self.start_iter = self.iter + 1


def add_probabilistic_config(cfg):
    """
        Add configuration elements specific to probabilistic detectron.

    Args:
        cfg (CfgNode): detectron2 configuration node.

    """
    _C = cfg
    _C.FFS = CN()
    # Probabilistic Modeling Setup
    _C.MODEL.PROBABILISTIC_MODELING = CN()
    _C.MODEL.PROBABILISTIC_MODELING.MC_DROPOUT = CN()
    _C.MODEL.PROBABILISTIC_MODELING.CLS_VAR_LOSS = CN()
    _C.MODEL.PROBABILISTIC_MODELING.BBOX_COV_LOSS = CN()

    # Annealing step for losses that require some form of annealing
    _C.MODEL.PROBABILISTIC_MODELING.ANNEALING_STEP = 0

    # Monte-Carlo Dropout Settings
    _C.MODEL.PROBABILISTIC_MODELING.DROPOUT_RATE = 0.0

    # Loss configs
    _C.MODEL.PROBABILISTIC_MODELING.CLS_VAR_LOSS.NAME = 'none'
    _C.MODEL.PROBABILISTIC_MODELING.CLS_VAR_LOSS.NUM_SAMPLES = 3

    _C.MODEL.PROBABILISTIC_MODELING.BBOX_COV_LOSS.NAME = 'none'
    _C.MODEL.PROBABILISTIC_MODELING.BBOX_COV_LOSS.COVARIANCE_TYPE = 'diagonal'
    _C.MODEL.PROBABILISTIC_MODELING.BBOX_COV_LOSS.NUM_SAMPLES = 1000

    # Probabilistic Inference Setup
    _C.PROBABILISTIC_INFERENCE = CN()
    _C.PROBABILISTIC_INFERENCE.MC_DROPOUT = CN()
    _C.PROBABILISTIC_INFERENCE.BAYES_OD = CN()
    _C.PROBABILISTIC_INFERENCE.ENSEMBLES_DROPOUT = CN()
    _C.PROBABILISTIC_INFERENCE.ENSEMBLES = CN()

    # General Inference Configs
    _C.PROBABILISTIC_INFERENCE.INFERENCE_MODE = 'standard_nms'
    _C.PROBABILISTIC_INFERENCE.MC_DROPOUT.ENABLE = False
    _C.PROBABILISTIC_INFERENCE.MC_DROPOUT.NUM_RUNS = 1
    _C.PROBABILISTIC_INFERENCE.AFFINITY_THRESHOLD = 0.7

    # Bayes OD Configs
    _C.PROBABILISTIC_INFERENCE.BAYES_OD.BOX_MERGE_MODE = 'bayesian_inference'
    _C.PROBABILISTIC_INFERENCE.BAYES_OD.CLS_MERGE_MODE = 'bayesian_inference'
    _C.PROBABILISTIC_INFERENCE.BAYES_OD.DIRCH_PRIOR = 'uniform'

    # Ensembles Configs
    _C.PROBABILISTIC_INFERENCE.ENSEMBLES.BOX_MERGE_MODE = 'pre_nms'
    _C.PROBABILISTIC_INFERENCE.ENSEMBLES.RANDOM_SEED_NUMS = [
        0, 1000, 2000, 3000, 4000]
    # 'mixture_of_gaussian' or 'bayesian_inference'
    _C.PROBABILISTIC_INFERENCE.ENSEMBLES.BOX_FUSION_MODE = 'mixture_of_gaussians'

    # vos args.
    _C.FFS.SAMPLE_NUMBER = 1000
    _C.FFS.STARTING_ITER = 12000
    _C.FFS.BATCH_SIZE = 100
    _C.FFS.SAMPLE_FROM = 100

    _C.FFS.ENABLE_CLUSTERING = True
    _C.FFS.CLUSTERING_START_ITER = 10
    _C.FFS.CLUSTERING_UPDATE_MU_ITER = 30
    _C.FFS.CLUSTERING_MOMENTUM = 0.99
    _C.FFS.CLUSTERING_ITEMS_PER_CLASS = 20
    _C.FFS.MARGIN = 10.0


def setup(arguments, seed=None, is_testing=False, ood=False):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_probabilistic_config(cfg)

    if ood:
        cfg.DATASETS.OOD = tuple()
    cfg.merge_from_file(arguments.config_file)
    cfg.MODEL.ROI_BOX_HEAD.DROPOUT_RATE = cfg.MODEL.PROBABILISTIC_MODELING.DROPOUT_RATE
    cfg["OUTPUT_DIR"] = "./outputs/"

    if is_testing:
        assert os.path.isdir(cfg["OUTPUT_DIR"]), "Checkpoint directory {} does not exists".format(cfg["OUTPUT_DIR"])

    # os.mkdirs(cfg["OUTPUT_DIR"], exist_ok=True)
    cfg.merge_from_list(arguments.opts)
    cfg['SEED'] = 0
    cfg.freeze()
    default_setup(cfg, arguments)
    setup_logger(cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="Probabilistic Detection")

    if seed is not None:
        torch.manual_seed(seed)
        numpy.random.seed(seed)
        random.seed(seed)

    return cfg


def main(argus):
    cfg = setup(argus, argus.seed)

    if argus.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=argus.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=argus.resume)
    return trainer.train()


if __name__ == "__main__":
    args = train_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
