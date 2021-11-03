import copy
import os.path as osp

import mmcv
import numpy as np
import json

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset

@DATASETS.register_module()
class TBX11KDataset2(CustomDataset):

    CLASSES = ('ActiveTuberculosis', 'ObsoletePulmonaryTuberculosis')#, 'PulmonaryTuberculosis')
    ## "ActiveTuberculosis":1; ObsoletePulmonaryTuberculosis:2
    def load_annotations(self, train_path):
        cat2label = {i+1: i for i, k in enumerate(self.CLASSES)}
        #print("cat2labelcat2labelcat2labelcat2label",cat2label)
        with open(train_path,'r') as f:
          data = json.load(f)
        
        data_infos = []

        for item in data['images']:
          image_id = item['id']
          fName = item['file_name']
          file_name = f'{self.img_prefix}{fName}'
          value = filter(lambda item1: item1['image_id'] == image_id,data['annotations'])

          image = mmcv.imread(file_name)
          height, width = image.shape[:2]
    
          data_info = dict(filename=f'{fName}', width=width, height=height)

          gt_bboxes = []
          gt_labels = []
          gt_bboxes_ignore = []
          gt_labels_ignore = []

          #print(item)
          for item2 in value:
            #print("item2item2item2item2",item2)
            category_id = item2['category_id']
            bbox = item2['bbox']
            #print(item2)

            gt_labels.append(cat2label[category_id])
            gt_bboxes.append(bbox)

          data_anno = dict(
              bboxes=np.array(gt_bboxes, dtype=np.float32).reshape(-1, 4),
              labels=np.array(gt_labels, dtype=np.long),
              bboxes_ignore=np.array(gt_bboxes_ignore,
                                     dtype=np.float32).reshape(-1, 4),
              labels_ignore=np.array(gt_labels_ignore, dtype=np.long))

          data_info.update(ann=data_anno)
          data_infos.append(data_info)
        #print("data_infosdata_infosdata_infos",data_infos)
        return data_infos
