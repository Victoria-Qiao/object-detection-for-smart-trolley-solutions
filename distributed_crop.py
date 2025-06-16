import glob
import multiprocessing as mp
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import os
import cv2
import numpy as np
from tqdm import tqdm
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--infer_mode', default='from_scratch')
parser.add_argument('--file_name', default=None)
parser.add_argument('--img_dir', default=None)
#parser.add_argument('--fil', default=None)

args = parser.parse_args()

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
#cfg.OUTPUT_DIR = 'output/ob_data_resample_98/from_scratch'
#cfg.OUTPUT_DIR = 'output/ob_data_resample_98/finetune_2211_229999'
if args.infer_mode == 'from_scratch':
    cfg.OUTPUT_DIR = 'output/task_finetune_feb_27/from_scratch'#finetune_2211_0229999'
    which_epoch = '0019999'
else:
    cfg.OUTPUT_DIR = 'output/task_finetune_feb_27/finetune_2211'
    which_epoch = '0023999'

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST=0.8
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
#which_epoch = '0089999'
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, f"model_{which_epoch}.pth")


class MultiProcessingCropObject():
    def __init__(self, img_dir):
        self.img_dir = img_dir
        #self.device_ids = [0,1,2,3,4,5,6,7]
        self.device_ids = [0,1,2,3,4,5,6,7]
    def chunkify(self, lst, chunk_size):
        return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

    def crop(self, barcode_list, predictor):
        epoch = '19999' if args.infer_mode=='from_scratch' else '23999'
        root = f'../../../imagr_instore_v5/crops'
        #root = f'../../nas_cv/faster-rcnn-ob-from-scratch-{epoch}-infer'
        for event in barcode_list:
            cam_list = os.listdir(self.img_dir + event)
            for cam in cam_list:
                imgs = os.listdir(self.img_dir + event + '/' + cam)
                crop_path = f'{root}/' + event + '/' + cam
                #infer_path = f'{root}/' + event + '/' + cam
                if not os.path.exists(crop_path):
                    os.makedirs(crop_path)
                #if not os.path.exists(infer_path):
                #    os.makedirs(infer_path)
                for img_path in tqdm(imgs):
                    #img_basename = os.path.basename(img_path)
                    path = self.img_dir + event + '/' + cam + '/' + img_path
                    input = cv2.imread(path)
                    output = predictor(input)
                    
                    #v = Visualizer(input[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1)
                    #out = v.draw_instance_predictions(output["instances"].to("cpu"))
                    #cv2.imwrite(f'{root}/infer_imgs_epoch_{epoch}/{event}/{cam}/{img_path}', out.get_image()[:,:,::-1], [cv2.IMWRITE_JPEG_QUALITY, 100])
                    
                    boxes = output['instances'].get('pred_boxes')
                    if not boxes:
                        continue
                    for idx, pred_box in enumerate(boxes[0]):
                        x_top_left = int(pred_box[0])
                        y_top_left = int(pred_box[1])
                        x_bottom_right = int(pred_box[2])
                        y_bottom_right = int(pred_box[3])
                        out = input[y_top_left:y_bottom_right, x_top_left:x_bottom_right, ::-1]
                        cv2.imwrite(f'{root}/{event}/{cam}/{img_path}', out[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 100])

    def parallel_crop(self, barcode_list, cfg):
        predictor = DefaultPredictor(cfg)
        self.crop(barcode_list, predictor)
        return


    def multiprocess_crop(self):
        barcode_list = os.listdir(self.img_dir)
        crop_list = os.listdir('../../../imagr_instore_v5/crops')
        barcode_list = [i for i in barcode_list if i not in crop_list]
        print(len(barcode_list))
        #barcode_list = [i for i in barcode_list if 'nothing' not in i]
        #barcode_list = random.choices(barcode_list,k=5)
        processes = []
        num_gpus = len(self.device_ids)

        #chunk_size = len(barcode_list) // num_gpus
        #chunks = self.chunkify(barcode_list, chunk_size)
        #print('number of chunks: ', len(chunks))

        chunks = np.array_split(np.array(barcode_list), num_gpus)
        print('number of chunks:', len(chunks))

        for idx in range(num_gpus):
            cfg.MODEL.DEVICE = f'cuda:{self.device_ids[idx]}'
            p = mp.Process(target=self.parallel_crop, args=(chunks[idx], cfg))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()


#img_dir = '../../big_daddy/walter_stuff/onboard/onboard_jpg/'
#img_dir = '../../big_daddy/nigel/hm01b0_data/OB_OD_testset_0224/'
#img_dir = '../../big_daddy/gp2211_qun_new/'
img_dir = args.img_dir
crop_object = MultiProcessingCropObject(img_dir)
crop_object.multiprocess_crop()
