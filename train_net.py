    #!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
import os

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import (default_argument_parser, default_setup, hooks,
                               launch)
from detectron2.evaluation import verify_results
from detectron2.utils.logger import setup_logger
from models import OpenDetTrainer, add_opendet_config, builtin
import argparse



from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json

from detectron2.data.datasets.pascal_voc import load_voc_instances
from detectron2.structures import BoxMode
import math
import json

import numpy as np
import os
import xml.etree.ElementTree as ET
from typing import List, Tuple, Union
from detectron2.utils.file_io import PathManager

import random
import numpy as np
import torch


# 注册数据集
class Register:
    
    """register my dataset"""
    #CLASS_NAMES = ['__background__', '1', '2', '3', '4', '5']  # 保留 background 类

    CLASS_NAMES = [
    # ShipRSImageNet_V1_CLASS_NAMES = [
    "Submarine", "Nimitz", "Midway", "Ticonderoga", "Atago DD", 
    "Hatsuyuki DD", "Hyuga DD", "Asagiri DD", "Perry FF", "Patrol",
    "YuTing LL", "YuDeng LL", "YuDao LL", "Austin LL", "Osumi LL",
    "LSD_41 LL", "LHA LL", "Commander", "Medical Ship",
    "Test Ship", "Training Ship", "Masyuu AS", "Sanantonio AS",
    "RoRo", "Cargo", "Barge", "Tugboat", "Ferry", "Yacht", "Wasp LL", "YuZhao LL",
    "Sailboat", "Fishing Vessel", "Oil Tanker", "Hovercraft", "Motorboat","Dock",
    # T2_CLASS_NAMES = [
    "EPF", "AOE", "Enterprise", "Container Ship", "Arleigh Burke DD",
    # UNK_CLASS = ["
    "unknown",
    ]

    ROOT = 'datasets/ShipRSImageNet_V1'  # 数据集路径

    def __init__(self,):
        self.CLASS_NAMES = Register.CLASS_NAMES #or ['__background__', ]
        # 数据集路径
        self.DATASET_ROOT = Register.ROOT
        self.ANN_ROOT = self.DATASET_ROOT

        # 声明数据集的子集
        self.PREDEFINED_SPLITS_DATASET = {
            # key               :  (dirname, split, year)
            "coco_my_train1234": (self.DATASET_ROOT,'train','voc2007'),#训练集
            "coco_my_val1234": (self.DATASET_ROOT,'test','voc2007'),#验证集
        }

    # 基于四个顶点坐标，获取旋转框的旋转角度
    def get_abbox_angle(self, annotation):
        centerx = (annotation[1] + annotation[3] + annotation[5] + annotation[7]) / 4
        centery = (annotation[2] + annotation[4] + annotation[6] + annotation[8]) / 4
        h = math.sqrt(math.pow((annotation[1] - annotation[3]), 2) + math.pow(
            (annotation[2] - annotation[4]), 2))
        w = math.sqrt(math.pow((annotation[1] - annotation[7]), 2) + math.pow(
            (annotation[2] - annotation[8]), 2))
        a = - math.degrees(math.atan2((annotation[8] - annotation[2]), (annotation[7] - annotation[1])))
        return a

    # 核心数据读取函数
    def get_dicts(self, dirname, split, class_names):
        """
        purpose: 用于定义自己的数据集格式，返回指定对象的数据
        :param ids: 需要返回数据的名称 (id) 号
        :return: 指定格式的数据字典
        """
        import os.path as osp
        
        with PathManager.open(os.path.join(dirname, "ImageSets", "Main", split + ".txt")) as f:
            fileids = np.loadtxt(f, dtype=str)
        
        print(f">>> Loading dataset: split={split}, found {len(fileids)} images")
        print(f">>> First 5 fileids: {fileids[:5]}")

        # Needs to read many small annotation files. Makes sense at local
        annotation_dirname = PathManager.get_local_path(os.path.join(dirname, "Annotations/"))
        dicts = []
        
        for fileid in fileids:
            # 去除可能存在的后缀名
            fileid_no_ext = osp.splitext(fileid)[0]
            
            anno_file = os.path.join(annotation_dirname, fileid_no_ext + ".xml")
            jpeg_file = os.path.join(dirname, "JPEGImages", fileid)
            
            # 检查文件是否存在
            if not os.path.exists(anno_file):
                print(f">>> Warning: Annotation file not found: {anno_file}")
                continue
            
            if not os.path.exists(jpeg_file):
                print(f">>> Warning: Image file not found: {jpeg_file}")
                continue

            with PathManager.open(anno_file) as f:
                tree = ET.parse(f)

            r = {
                "file_name": jpeg_file,
                "image_id": fileid_no_ext,
                "height": int(tree.findall("./size/height")[0].text),
                "width": int(tree.findall("./size/width")[0].text),
            }
            instances = []

            for obj in tree.findall("object"):
                cls = obj.find("name").text
                
                # 根据论文，将 8 个通用类别视为未知类别
                unknown_classes = [
                    "Ship", "Warship", "Merchant", "Destroyer", 
                    "Frigate", "Landing", "Auxiliary Ships", "Aircraft carrier"
                ]
                
                # 统一大小写不一致的类别名称
                class_normalization = {
                    "Training ship": "Training Ship",
                    "Test ship": "Test Ship", 
                    "Medical ship": "Medical Ship",
                }
                
                # 应用标准化
                cls = class_normalization.get(cls, cls)
                
                # 如果是未知类别，映射到 "unknown"
                if cls in unknown_classes:
                    cls = "unknown"
                
                # 如果仍然不在已知类别列表中，跳过并警告
                if cls not in class_names:
                    print(f">>> Warning: Unknown class '{cls}' found in {anno_file}")
                    print(f">>> Skipping this annotation...")
                    continue

                use_origin=True
                if use_origin:
                    bbox = obj.find("rotated_box")
                    angle = math.degrees(float(bbox.find('rot').text))

                else:
                    bbox = obj.find("rotated_box")
                    polygon = obj.find("polygon")
                    x1, y1 = float(polygon.find("x1").text), float(polygon.find("y1").text)
                    x2, y2 = float(polygon.find("x2").text), float(polygon.find("y2").text)
                    x3, y3 = float(polygon.find("x3").text), float(polygon.find("y3").text)
                    x4, y4 = float(polygon.find("x4").text), float(polygon.find("y4").text)
                    array = [0,x2,y2,x1,y1,x4,y4,x3,y3]
                    angle = self.get_abbox_angle(array)

                bbox_values = [
                        float(bbox.find(x).text) for x in ['cx', 'cy', 'width', 'height']
                ] + [angle]

                instances.append(
                    {"category_id": class_names.index(cls), "bbox": bbox_values, "bbox_mode": BoxMode.XYWHA_ABS}
                )
            r["annotations"] = instances
            dicts.append(r)
        
        print(f">>> Successfully loaded {len(dicts)} images")
        return dicts

    def register_dataset(self,):
        """
        purpose: register all splits of datasets with PREDEFINED_SPLITS_DATASET
        注册数据集（这一步就是将自定义数据集注册进Detectron2）
        """
        for key, (dirname, split, year) in self.PREDEFINED_SPLITS_DATASET.items():

            self.register_dataset_instances(name=key,dirname=dirname,split=split,year=year,class_names=self.CLASS_NAMES)

    # @staticmethod
    def register_dataset_instances(self, name, dirname, split, year, class_names):
        """
        purpose: register datasets to DatasetCatalog,
                 register metadata to MetadataCatalog and set attribute
        注册数据集实例，加载数据集中的对象实例
        """
        print(10*'>','register_dataset_instances')
        # print(name,image_root,json_file)
        
        # 预先加载并缓存数据集，避免每次重复解析 XML
        dataset_dicts = self.get_dicts(dirname, split, class_names)
        
        # 向 Detectron2 注册自定义数据集（使用已缓存的数据）
        DatasetCatalog.register(name, lambda: dataset_dicts)
        # DatasetCatalog.register(name, lambda: load_coco_json(json_file, image_root, name))
        # MetadataCatalog.get(name).set(json_file=json_file,
        #                               image_root=image_root,
        #                               evaluator_type="coco")

        # 为已注册的数据集添加描述信息
        MetadataCatalog.get(name).set(
            thing_classes=list(class_names), dirname=dirname, year=year, split=split, evaluator_type='pascal_voc'
        )


def setup(args):
    """
    Create configs and perform basic setups.
    """

    #创建一个空的配置对象，将所有模型、数据集、训练超参数等组织在一个属性结构中
    cfg = get_cfg()

    Register().register_dataset()  # register my dataset

    # add opendet config
    # 为配置对象cfg添加开集测试所需的自定义配置项
    add_opendet_config(cfg)
    # 从指定taml配置文件中读取配置，并合并到当前的cfg对象中
    cfg.merge_from_file(args.config_file)
    # 从命令行参数列表中合并配置。
    cfg.merge_from_list(args.opts)
    # Note: we use the key ROI_HEAD.NUM_KNOWN_CLASSES
    # for open-set data processing and evaluation.
    # 兼容性处理，确保MODEL.ROI_HEADS.NUM_KNOWN_CLASSES与模型实际已知类的数量一致
    if 'RetinaNet' in cfg.MODEL.META_ARCHITECTURE:
        cfg.MODEL.ROI_HEADS.NUM_KNOWN_CLASSES = cfg.MODEL.RETINANET.NUM_KNOWN_CLASSES
    # add output dir if not exist
    # 创建输出目录，保存日志
    if cfg.OUTPUT_DIR == "./output":
        config_name = os.path.basename(args.config_file).split(".yaml")[0]
        cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, config_name)
    # 将配置对象冻结，使其变为只读状态
    cfg.freeze()
    # 执行一系列标准的初始化操作，包括设置日志记录器、设置随机
    default_setup(cfg, args)
    # 设置名为"DUONet"的日志记录器
    setup_logger(output=cfg.OUTPUT_DIR,
                 distributed_rank=comm.get_rank(), name="DUONet")
    return cfg


def main(args):
    # 调用 setup 函数初始化配置
    cfg = setup(args)
    # 打印模型权重文件路径
    print(cfg.MODEL.WEIGHTS)
    # 如果是评估模式，则加载模型权重，执行测试并返回结果。默认为false
    if args.eval_only:
        # 使用 build_model 方法构建模型
        model = OpenDetTrainer.build_model(cfg)
        # 加载模型权重。若resume=True，则从cfg.OUTPUT_DIR中加载权重文件，否则从cfg.MODEL.WEIGHTS中加载权重文件。
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        # 在指定的测试数据集上评估模型，返回评估结果
        res = OpenDetTrainer.test(cfg, model)
        # cfg.TEST.AUG.ENABLED：配置项，控制是否启用测试时数据增强
        if cfg.TEST.AUG.ENABLED:
            # 使用测试时数据增强进行评估
            res.update(OpenDetTrainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    #  否则创建训练器，回复或重新开始训练
    trainer = OpenDetTrainer(cfg)
    # 从检查点恢复训练或加载预训练权重
    trainer.resume_or_load(resume=args.resume)
    # 启用测试时数据增强
    if cfg.TEST.AUG.ENABLED:
        # 注册一个评估钩子，在训练过程中执行测试时数据增强
        trainer.register_hooks(
            [hooks.EvalHook(
                0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    #调用分布式训练启动函数
    launch(
        main,#要运行的主程序
        args.num_gpus,#GPU数量
        num_machines=args.num_machines,#机器数量
        machine_rank=args.machine_rank,#当前机器的排名
        dist_url=args.dist_url,#指定分布式训练的 URL
        args=(args,),#传递给 main 函数的参数
    )
