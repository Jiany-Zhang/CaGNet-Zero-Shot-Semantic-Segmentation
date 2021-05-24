import os
import pathlib

import numpy as np
import scipy.io
from PIL import Image
from torchvision import transforms
import torch

from dataset import custom_transforms as tr
from .base import BaseDataset, load_obj, lbl_contains_unseen

SBD_DIR = pathlib.Path("./data/VOC2012/benchmark_RELEASE")


class SBDSegmentation(BaseDataset):
    NUM_CLASSES = 21

    def __init__(
        self,
        args,
        base_dir=SBD_DIR,
        split="train",
        load_embedding=None,
        w2c_size=300,
        weak_label=False,
        unseen_classes_idx_weak=[],
        transform=True,
    ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        if isinstance(split, str):
            split = [split]
        split.sort()
        super().__init__(
            args,
            base_dir,
            split,
            load_embedding,
            w2c_size,
            weak_label,
            unseen_classes_idx_weak,
            transform,
        )
        self._dataset_dir = self._base_dir / "dataset"
        self._image_dir = self._dataset_dir / "img"
        self._cat_dir = self._dataset_dir / "cls"

        # Get list of all images from the split and check that the files exist
        self.im_ids = []
        self.categories = []
        for splt in self.split:
            lines = (self._dataset_dir / f"{splt}.txt").read_text().splitlines()

            for line in lines:
                _image = self._image_dir / f"{line}.jpg"
                _categ = self._cat_dir / f"{line}.mat"
                assert _image.is_file()
                assert _categ.is_file()

                # if unseen classes
                if len(args.unseen_classes_idx) > 0:
                    _target = Image.fromarray(
                        scipy.io.loadmat(_categ)["GTcls"][0]["Segmentation"][0]
                    )
                    _target = np.array(_target, dtype=np.uint8)
                    if lbl_contains_unseen(_target, args.unseen_classes_idx):
                        continue

                self.im_ids.append(line)
                self.images.append(_image)
                self.categories.append(_categ)

        assert len(self.images) == len(self.categories)

        # Display stats
        print(f"(sbd) Number of images: {len(self.images):d}")

    def init_embeddings(self):
        if self.load_embedding == "attributes":
            embed_arr = np.load("embeddings/pascal/pascalvoc_class_attributes.npy")
        elif self.load_embedding == "w2c":
            embed_arr = load_obj(
                "embeddings/pascal/w2c/norm_embed_arr_" + str(self.w2c_size)
            )
        elif self.load_embedding == "w2c_bg":
            embed_arr = np.load("embeddings/pascal/pascalvoc_class_w2c_bg.npy")
        elif self.load_embedding == "my_w2c":
            embed_arr = np.load("embeddings/pascal/pascalvoc_class_w2c.npy")
        elif self.load_embedding == "fusion":
            attributes = np.load("embeddings/pascal/pascalvoc_class_attributes.npy")
            w2c = np.load("embeddings/pascal/pascalvoc_class_w2c.npy")
            embed_arr = np.concatenate((attributes, w2c), axis=1)
        else:
            raise KeyError(self.load_embedding)
        self.make_embeddings(embed_arr)

    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)

        if self.weak_label:
            unique_class = np.sort(np.unique(np.array(_target))).tolist()
            if 255 in unique_class:
                unique_class.remove(255)
            _one_hot = -1 * torch.ones(self.NUM_CLASSES)
            for i, j in enumerate(unique_class):
                _one_hot[i] = j
            '''
            _one_hot = torch.zeros(self.NUM_CLASSES)
            
            _one_hot[unique_class.tolist()] = 1
            '''
        sample = {"image": _img, "label": _target,}

        if self.transform:
            sample = self.transform_s(sample)
        else:
            sample = self.transform_weak(sample)
        sample["image_name"] = str(self.images[index])
        sample["weak_label"] = _one_hot
        return sample

    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert("RGB")
        _target = Image.fromarray(
            scipy.io.loadmat(self.categories[index])["GTcls"][0]["Segmentation"][0]
        )

        return _img, _target

    def transform_s(self, sample):
        composed_transforms = transforms.Compose(
            [
                tr.RandomHorizontalFlip(),
                tr.RandomScaleCrop(
                    base_size=self.args.base_size, crop_size=self.args.crop_size
                ),
                tr.RandomGaussianBlur(),
                tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.ToTensor(),
            ]
        )

        return composed_transforms(sample)

    def transform_weak(self, sample):

        composed_transforms = transforms.Compose(
            [
                tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.ToTensor(),
            ]
        )

        return composed_transforms(sample)

    def __str__(self):
        return f"SBDSegmentation(split={self.split})"
