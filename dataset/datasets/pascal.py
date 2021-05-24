import pathlib

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from dataset import custom_transforms as tr
from .base import BaseDataset, load_obj, lbl_contains_unseen


PASCAL_DIR = pathlib.Path("./data/VOC2012")


class VOCSegmentation(BaseDataset):
    """
    PascalVoc dataset
    """
    NUM_CLASSES = 21

    def __init__(
        self,
        args,
        base_dir=PASCAL_DIR,
        split="train",
        load_embedding=None,
        w2c_size=300,
        weak_label=True,
        unseen_classes_idx_weak=[],
        transform=True,
    ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
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
        self._image_dir = self._base_dir / "JPEGImages"
        self._cat_dir = self._base_dir / "SegmentationClass"

        self.unseen_classes_idx_weak = set(args.unseen_classes_idx)

        _splits_dir = self._base_dir / "ImageSets" / "Segmentation"

        self.im_ids = []
        self.categories = []

        lines = (_splits_dir / f"{self.split}.txt").read_text().splitlines()

        for ii, line in enumerate(lines):
            _image = self._image_dir / f"{line}.jpg"
            _cat = self._cat_dir / f"{line}.png"
            assert _image.is_file(), _image
            assert _cat.is_file(), _cat

            # if unseen classes and training split
            if len(args.unseen_classes_idx) > 0 and self.split == "train":
                cat = Image.open(_cat)
                cat = np.array(cat, dtype=np.uint8)
                if lbl_contains_unseen(cat, args.unseen_classes_idx):
                    continue

            self.im_ids.append(line)
            self.images.append(_image)
            self.categories.append(_cat)

        assert len(self.images) == len(self.categories)

        # Display stats
        print(f"(pascal) Number of images in {split}: {len(self.images):d}")

    def init_embeddings(self):
        embed_arr = load_obj("embeddings/pascal/w2c/norm_embed_arr_" + str(self.w2c_size))
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
            if self.split == "train":
                sample = self.transform_tr(sample)
            elif self.split == "val":
                sample = self.transform_val(sample)
        else:
            sample = self.transform_weak(sample)

        sample["image_name"] = str(self.images[index])
        sample["weak_label"] = _one_hot
        return sample

    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert("RGB")
        _target = Image.open(self.categories[index])
        return _img, _target

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose(
            [
                tr.RandomHorizontalFlip(),
                tr.RandomScaleCrop(
                    base_size=self.args.base_size,
                    crop_size=self.args.crop_size,
                    fill=255,
                ),
                tr.RandomGaussianBlur(),
                tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.ToTensor(),
            ]
        )
        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose(
            [
                tr.FixScale(crop_size=self.args.crop_size),
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
        return f"VOC2012(split={self.split})"
