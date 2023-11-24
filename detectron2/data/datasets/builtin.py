# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.


"""
This file registers pre-defined datasets at hard-coded paths, and their metadata.

We hard-code metadata for common datasets. This will enable:
1. Consistency check when loading the datasets
2. Use models on these standard datasets directly and run demos,
   without having to download the dataset annotations

We hard-code some paths to the dataset that's assumed to
exist in "./datasets/".

Users SHOULD NOT use this file to create new dataset / metadata for new dataset.
To add new dataset, refer to the tutorial "docs/DATASETS.md".
"""

import os
from detectron2.data import MetadataCatalog
from .builtin_meta import _get_builtin_metadata
from .coco import register_coco_instances
from collections import ChainMap

VOC_THING_CLASSES = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 'airplane', 'bicycle', 'boat', 'bus',
                     'car', 'motorcycle', 'train', 'bottle', 'chair', 'dining table', 'potted plant', 'couch', 'tv',
                     ]
VOC_ID_THING_CLASSES = [
    'person', 'dog', 'horse', 'sheep', 'motorcycle', 'train', 'dining table', 'potted plant', 'couch', 'tv'
]
VOC_OOD_THING_CLASSES = [
    'bird', 'cat', 'cow', 'airplane', 'bicycle', 'boat', 'bus', 'car', 'bottle', 'chair'
]


def setup_coco_dataset(root):
    root = os.path.join(root, 'coco')
    train_image_dir = os.path.join(root, 'train2017')
    test_image_dir = os.path.join(root, 'val2017')

    train_json_annotations = os.path.join(
        root, 'annotations', 'instances_train2017.json')
    test_json_annotations = os.path.join(
        root, 'annotations', 'instances_val2017.json')

    register_coco_instances(
        "coco_2017_custom_train",
        {},
        train_json_annotations,
        train_image_dir)
    MetadataCatalog.get(
        "coco_2017_custom_train").thing_classes = _get_builtin_metadata("coco")["thing_classes"]
    MetadataCatalog.get(
        "coco_2017_custom_train").thing_dataset_id_to_contiguous_id = (
        _get_builtin_metadata("coco"))["thing_dataset_id_to_contiguous_id"]

    register_coco_instances(
        "coco_2017_custom_val",
        {},
        test_json_annotations,
        test_image_dir)
    MetadataCatalog.get(
        "coco_2017_custom_val").thing_classes = _get_builtin_metadata("coco")["thing_classes"]
    MetadataCatalog.get(
        "coco_2017_custom_val").thing_dataset_id_to_contiguous_id = (
        _get_builtin_metadata("coco"))["thing_dataset_id_to_contiguous_id"]


def setup_openimages_dataset(root):
    root = os.path.join(root, "OpenImages")
    test_image_dir = os.path.join(root, 'images')
    test_json_annotations = os.path.join(
        root, 'COCO-Format', 'val_coco_format.json')

    register_coco_instances(
        "openimages_val",
        {},
        test_json_annotations,
        test_image_dir)
    MetadataCatalog.get(
        "openimages_val").thing_classes = _get_builtin_metadata("coco")["thing_classes"]
    MetadataCatalog.get(
        "openimages_val").thing_dataset_id_to_contiguous_id = dict(
        ChainMap(*[{i + 1: i} for i in range(80)]))


def setup_openimages_ood_dataset(root):
    root = os.path.join(root, "OpenImages")
    test_image_dir = os.path.join(root + 'ood_classes_rm_overlap', 'images')
    test_json_annotations = os.path.join(
        root + 'ood_classes_rm_overlap', 'COCO-Format', 'val_coco_format.json')

    register_coco_instances(
        "openimages_ood_val",
        {},
        test_json_annotations,
        test_image_dir)
    MetadataCatalog.get(
        "openimages_ood_val").thing_classes = _get_builtin_metadata("coco")["thing_classes"]
    MetadataCatalog.get(
        "openimages_ood_val").thing_dataset_id_to_contiguous_id = dict(
        ChainMap(*[{i + 1: i} for i in range(80)]))


def setup_voc_id_dataset(root):
    root = os.path.join(root, "VOC0712")
    train_image_dir = os.path.join(root, 'JPEGImages')
    test_image_dir = os.path.join(root, 'JPEGImages')

    train_json_annotations = os.path.join(
        root, 'voc0712_train_all.json')
    test_json_annotations = os.path.join(
        root, 'val_coco_format.json')

    register_coco_instances(
        "voc_custom_train_id",
        {},
        train_json_annotations,
        train_image_dir)
    MetadataCatalog.get(
        "voc_custom_train_id").thing_classes = VOC_ID_THING_CLASSES
    MetadataCatalog.get(
        "voc_custom_train_id").thing_dataset_id_to_contiguous_id = dict(
        ChainMap(*[{i + 1: i} for i in range(10)]))

    register_coco_instances(
        "voc_custom_val_id",
        {},
        test_json_annotations,
        test_image_dir)
    MetadataCatalog.get(
        "voc_custom_val_id").thing_classes = VOC_ID_THING_CLASSES
    MetadataCatalog.get(
        "voc_custom_val_id").thing_dataset_id_to_contiguous_id = dict(
        ChainMap(*[{i + 1: i} for i in range(10)]))


def setup_voc_dataset(root):
    root = os.path.join(root, "VOC0712")
    train_image_dir = os.path.join(root, 'JPEGImages')
    test_image_dir = os.path.join(root, 'JPEGImages')

    train_json_annotations = os.path.join(
        root, 'voc0712_train_all.json')
    test_json_annotations = os.path.join(
        root, 'val_coco_format.json')

    register_coco_instances(
        "voc_custom_train",
        {},
        train_json_annotations,
        train_image_dir)
    MetadataCatalog.get(
        "voc_custom_train").thing_classes = VOC_THING_CLASSES
    MetadataCatalog.get(
        "voc_custom_train").thing_dataset_id_to_contiguous_id = dict(
        ChainMap(*[{i + 1: i} for i in range(20)]))

    register_coco_instances(
        "voc_custom_val",
        {},
        test_json_annotations,
        test_image_dir)
    MetadataCatalog.get(
        "voc_custom_val").thing_classes = VOC_THING_CLASSES
    MetadataCatalog.get(
        "voc_custom_val").thing_dataset_id_to_contiguous_id = dict(
        ChainMap(*[{i + 1: i} for i in range(20)]))


def setup_voc_ood_dataset(root):
    root = os.path.join(root, "VOC0712")
    test_image_dir = os.path.join(root, 'JPEGImages')
    test_json_annotations = os.path.join(
        root, 'val_coco_format.json')

    register_coco_instances(
        "voc_ood_val",
        {},
        test_json_annotations,
        test_image_dir)
    MetadataCatalog.get(
        "voc_ood_val").thing_classes = VOC_OOD_THING_CLASSES
    MetadataCatalog.get(
        "voc_ood_val").thing_dataset_id_to_contiguous_id = dict(
        ChainMap(*[{i + 1: i} for i in range(10)]))


def setup_coco_ood_dataset(root):
    root = os.path.join(root, "coco")
    test_image_dir = os.path.join(root, 'val2017')
    test_json_annotations = os.path.join(
        root, 'annotations', 'instances_val2017_ood_rm_overlap.json')

    register_coco_instances(
        "coco_ood_val",
        {},
        test_json_annotations,
        test_image_dir)
    MetadataCatalog.get(
        "coco_ood_val").thing_classes = _get_builtin_metadata("coco")["thing_classes"]
    MetadataCatalog.get(
        "coco_ood_val").thing_dataset_id_to_contiguous_id = (
        _get_builtin_metadata("coco"))["thing_dataset_id_to_contiguous_id"]


def setup_coco_ood_train_dataset(root):
    root = os.path.join(root, "coco")
    test_image_dir = os.path.join(root, 'train2017')
    test_json_annotations = os.path.join(
        root, 'annotations', 'instances_train2017_ood.json')

    register_coco_instances(
        "coco_ood_train",
        {},
        test_json_annotations,
        test_image_dir)
    MetadataCatalog.get(
        "coco_ood_train").thing_classes = _get_builtin_metadata("coco")["thing_classes"]
    MetadataCatalog.get(
        "coco_ood_train").thing_dataset_id_to_contiguous_id = (
        _get_builtin_metadata("coco"))["thing_dataset_id_to_contiguous_id"]


def register_vis_dataset(root):
    root = os.path.join(root, "VIS")
    thing_classes = ['airplane', 'bear', 'bird', 'boat', 'car', 'cat', 'cow', 'deer', 'dog', 'duck',
                     'earless_seal', 'elephant', 'fish', 'flying_disc', 'fox', 'frog', 'giant_panda',
                     'giraffe', 'horse', 'leopard', 'lizard', 'monkey', 'motorbike', 'mouse', 'parrot',
                     'person', 'rabbit', 'shark', 'skateboard', 'snake', 'snowboard', 'squirrel', 'surfboard',
                     'tennis_racket', 'tiger', 'train', 'truck', 'turtle', 'whale', 'zebra']
    VIS_THING_DATASET_ID_TO_CONTIGUOUS_ID = dict(ChainMap(*[{i + 1: i} for i in range(40)]))
    metadata = {"thing_classes": thing_classes}

    register_coco_instances(
        'vis21_val',
        metadata,
        os.path.join(root, 'instances_val.json'),
        os.path.join(root, "JPEGImages/"),
    )
    register_coco_instances(
        'vis21_train',
        metadata,
        os.path.join(root, "instances_train.json"),
        os.path.join(root, "JPEGImages/"),
    )
    MetadataCatalog.get(
        "vis21_train").thing_dataset_id_to_contiguous_id = VIS_THING_DATASET_ID_TO_CONTIGUOUS_ID
    MetadataCatalog.get(
        "vis21_val").thing_dataset_id_to_contiguous_id = VIS_THING_DATASET_ID_TO_CONTIGUOUS_ID


if __name__.endswith(".builtin"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.path.expanduser(os.getenv("DETECTRON2_DATASETS", "/datasets"))
    setup_coco_ood_train_dataset(_root)
    setup_coco_dataset(_root)
    setup_openimages_dataset(_root)
    setup_openimages_ood_dataset(_root)
    setup_voc_id_dataset(_root)
    setup_voc_dataset(_root)
    setup_voc_ood_dataset(_root)
    setup_coco_ood_dataset(_root)
    register_vis_dataset(_root)
