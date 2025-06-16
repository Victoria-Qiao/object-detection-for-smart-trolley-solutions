import logging
import os
from collections import OrderedDict

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
import argparse

'''
parser = argparse.ArgumentParser()
parser.add_argument('--train_folder', default=None)
parser.add_argument('--test_folder', default=None)
parser.add_argument('--output_dir', default=None)
parser.add_argument('--step1', default=None)
parser.add_argument('--step2', default=None)
parser.add_argument('--max_iter', default=None)

#parser.add_argument('--fil', default=None)

args = parser.parse_args()
'''

#train_folder = '../../../v3_v4_v5_sampled/imagr_instore_cam0_sampled/train' 
#test_folder = '../../../v3_v4_v5_sampled/imagr_instore_cam0_sampled/val'

cam = 2
train_folder = f'../../../imagr_instore_v3_v4_OD_dataset_jpg_unrotated_10k_groomed_260524/cam{cam}'
test_folder = f'../../../imagr_instore_v5_OD_dataset_jpg_unrotated_5k_groomed_260524/cam{cam}'
output_dir = f'output/from_scratch_cam{cam}_unrotate_groom'
bs = 32
step1 = 33979 #26550
step2 = 47570 #37170
max_iter = 67957 #53100

register_coco_instances('self_coco_train', {},
                        f'{train_folder}/annotations/v3_v4_OD_retinanet_jpg_unrotated_200524_cam{cam}.json', f'{train_folder}/images')
register_coco_instances('self_coco_val', {},
                        f'{test_folder}/annotations/OD_retinanet_jpg_unrotated_200524_cam{cam}.json',
                       f'{test_folder}/images')
coco_val_metadata = MetadataCatalog.get('self_coco_val')
dataset_dicts_val = DatasetCatalog.get('self_coco_val')
# print(coco_val_metadata)

coco_train_metadata = MetadataCatalog.get('self_coco_train')
dataset_dicts_train = DatasetCatalog.get('self_coco_train')



def build_evaluator(cfg, dataset_name, output_folder=None):
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        return CityscapesSemSegEvaluator(dataset_name)
    elif evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    elif evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, output_dir=output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    elif len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


class Trainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_evaluator(cfg, dataset_name, output_folder)

    # @classmethod
    # def test_with_TTA(cls, cfg, model):
    #     logger = logging.getLogger("detectron2.trainer")
    #     # In the end of training, run an evaluation with TTA
    #     # Only support some R-CNN models.
    #     logger.info("Running inference with test-time augmentation ...")
    #     model = GeneralizedRCNNWithTTA(cfg, model)
    #     evaluators = [
    #         cls.build_evaluator(
    #             cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
    #         )
    #         for name in cfg.DATASETS.TEST
    #     ]
    #     res = cls.test(cfg, model, evaluators)
    #     res = OrderedDict({k + "_TTA": v for k, v in res.items()})
    #     return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    #cfg = get_cfg()
    #cfg.merge_from_file(args.config_file)
    #cfg.merge_from_list(args.opts)
    #cfg.freeze()
    #default_setup(cfg, args)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
    cfg.merge_from_list(args.opts)
    cfg.DATASETS.TRAIN = ("self_coco_train",)
    cfg.DATASETS.TEST = ('self_coco_val',)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
    #cfg.MODEL.WEIGHTS = ('output/task_new_office_new_onboard/finetune_2211_ob_data_sample/model_0031999.pth') 
    #cfg.MODEL.WEIGHTS = ('output/ob_data_sample/from_scratch/model_0089999.pth')
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = bs
    #cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST=0.5  # This is the real "batch size" commonly known to deep learning people
    cfg.INPUT.MIN_SIZE_TRAIN = (800,)
    cfg.SOLVER.BASE_LR = 0.00025
    #cfg.SOLVER.BASE_LR = 0.02 * bs / 16  # pick a good LR
    cfg.SOLVER.STEPS = (step1, step2)
    cfg.SOLVER.MAX_ITER = max_iter  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.CHECKPOINT_PERIOD = 2500
    cfg.TEST.EVAL_PERIOD = 1000
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST=0.75
    cfg.SOLVER.STEPS = []  # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
    
    cfg.OUTPUT_DIR = output_dir#'output/task_finetune_feb_27/finetune_2211'
    #cfg.OUTPUT_DIR = 'output/ob_data_resample_98/from_scratch'
    cfg.SOLVER.AMP.ENABLED = False
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    if args.eval_only:
        cfg.MODEL.WEIGHTS = ('output/ob_data_sample/finetune_2211_0229999/model_0083999.pth')
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            print('is main process')
            verify_results(cfg, res)
        return res
    
    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    # trainer.resume_or_load(resume=args.resume)
    trainer.resume_or_load(resume=False)
    # if cfg.TEST.AUG.ENABLED:
    #     trainer.register_hooks(
    #         [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
    #     )
    return trainer.train()


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
