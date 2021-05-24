import os
import os.path as osp
import pathlib

import numpy as np
import scipy
from PIL import Image
from torchvision import transforms
import torch
from dataset import custom_transforms as tr
from .base import BaseDataset, load_obj, lbl_contains_unseen


CONTEXT_DIR = pathlib.Path("./data/context/")


class ContextSegmentation(BaseDataset):
    """
    PascalVoc dataset
    """

    NUM_CLASSES = 60

    def __init__(
        self,
        args,
        base_dir=CONTEXT_DIR,
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

        self._image_dir = self._base_dir / "pascal/VOCdevkit/VOC2012/JPEGImages"
        self._cat_dir = self._base_dir / "full_annotations/trainval"

        self.unseen_classes_idx_weak = set(args.unseen_classes_idx)


        self.im_ids = []
        self.categories = []

        self.labels_459 = [
            label.decode().replace(" ", "")
            for idx, label in np.genfromtxt(
                osp.join(self._base_dir, "full_annotations/labels.txt"),
                delimiter=":",
                dtype=None,
            )
        ]
        self.labels_59 = [
            label.decode().replace(" ", "")
            for idx, label in np.genfromtxt(
                osp.join(self._base_dir, "classes-59.txt"), delimiter=":", dtype=None
            )
        ]
        """
        for main_label, task_label in zip(
            ("table", "bedclothes", "cloth"), ("diningtable", "bedcloth", "clothes")
        ):
            self.labels_59[self.labels_59.index(task_label)] = main_label
        """


        self.idx_59_to_idx_469 = {}
        for idx, l in enumerate(self.labels_59):
            if idx > 0:
                self.idx_59_to_idx_469[idx] = self.labels_459.index(l) + 1

        lines = (self._base_dir / f"{self.split}.txt").read_text().splitlines()

        for ii, line in enumerate(lines):
            _image = self._image_dir / f'{line}.jpg'
            _cat = self._cat_dir / f"{line}.mat"
            assert _image.is_file(), _image
            assert _cat.is_file(), _cat

            # if unseen classes and training split
            if len(args.unseen_classes_idx) > 0:
                cat = self.load_label(_cat)
                if lbl_contains_unseen(cat, args.unseen_classes_idx) and self.split == "train":
                    continue

            self.im_ids.append(line)
            self.images.append(_image)
            self.categories.append(_cat)

        assert len(self.images) == len(self.categories)

        # Display stats
        print(
            "(context) Number of images in {}: {:d}, {:d} deleted".format(
                split, len(self.images), len(lines) - len(self.images)
            )
        )

    def load_label(self, file_path):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        The full 459 labels are translated to the 59 class task labels.
        """
        label_459 = scipy.io.loadmat(file_path)["LabelMap"]
        label = np.zeros_like(label_459, dtype=np.uint8)
        for idx, l in enumerate(self.labels_59):
            if idx > 0:
                label[label_459 == self.idx_59_to_idx_469[idx]] = idx
        return label

    def init_embeddings(self):
        if self.load_embedding == "my_w2c":
            embed_arr = np.load("embeddings/context/pascalcontext_class_w2c.npy")
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

        sample = {"image": _img, "label": _target}

        if self.transform:
            if self.split == "train":
                sample = self.transform_tr(sample)
            elif self.split == "val":
                sample = self.transform_val(sample)
        else:
            sample = self.transform_weak(sample)

        sample["image_name"] = str(self.images[index])
        sample["weak_label"] = _one_hot
        sample["class_embedding"] = self.embeddings.weight.data.detach()
        return sample["image"], sample["label"]

    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert("RGB")
        _target = self.load_label(self.categories[index])
        _target = Image.fromarray(_target)
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
