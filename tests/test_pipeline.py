"""Unit tests on synthetic data (no DENTEX images, GPU or nnU-Net needed).

Run with:  python -m pytest tests   (or: python -m unittest discover tests)
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dentex.data.coco import XRay, Annotation, instance_labels, instance_masks, load_dentex, semantic_mask
from dentex.data.dataset import SegmentationDataset
from dentex.data.refine import BORDER, TOOTH, add_border_class, remove_overlaps, separate_teeth
from dentex.export import to_records
from dentex.labels import decode_enumeration, encode, fdi, hflip_lookup
from dentex.losses import DiceCELoss
from dentex.metrics import SegmentationMeter, foreground_scores
from dentex.models.unet import UNet
from dentex.nnsam import build_raw_dataset
from dentex.postprocess.boxes import Box, boxes_from_labelmap, dbscan_1d
from dentex.postprocess.instances import count_teeth, instances_from_border_prediction


def square(x0, y0, x1, y1):
    return [x0, y0, x1, y0, x1, y1, x0, y1]


def toy_xray():
    return XRay(1, "toy.png", 40, 20, [
        Annotation([square(2, 2, 18, 18)], (2, 2, 16, 16), quadrant=0, tooth=0, disease=1),
        Annotation([square(16, 2, 32, 18)], (16, 2, 16, 16), quadrant=1, tooth=3),
    ])


class TestLabels(unittest.TestCase):
    def test_enumeration_roundtrip_and_fdi(self):
        for q in range(4):
            for t in range(8):
                self.assertEqual(decode_enumeration(encode("enumeration", q, t)), (q, t))
        self.assertEqual(fdi(0, 0), 11)
        self.assertEqual(fdi(3, 7), 48)

    def test_hflip_swaps_left_and_right_quadrants(self):
        table = hflip_lookup("quadrant")
        self.assertEqual(table[1:], [2, 1, 4, 3])
        enum = hflip_lookup("enumeration")
        self.assertEqual(decode_enumeration(enum[encode("enumeration", 0, 5)]), (1, 5))
        self.assertEqual(hflip_lookup("disease"), list(range(5)))


class TestMasks(unittest.TestCase):
    def test_semantic_mask_labels(self):
        xray = toy_xray()
        quad = semantic_mask(xray, "quadrant")
        self.assertEqual(set(np.unique(quad)), {0, 1, 2})
        disease = semantic_mask(xray, "disease")
        self.assertEqual(set(np.unique(disease)), {0, 2})        # only the first tooth is diseased

    def test_overlap_removal_and_border(self):
        inst = instance_masks(toy_xray())
        overlap = inst[0] & inst[1]
        self.assertTrue(overlap.any())
        self.assertTrue((remove_overlaps(inst)[overlap] == 0).all())
        three = add_border_class(remove_overlaps(inst), border_size=1)
        self.assertEqual(set(np.unique(three)), {0, TOOTH, BORDER})

    def test_separate_teeth_leaves_a_gap_wider_than_the_overlap(self):
        inst = instance_masks(toy_xray())
        separated = separate_teeth(inst, [1, 2], gap=2)
        overlap = inst[0] & inst[1]
        self.assertTrue((separated[overlap] == 0).all())
        cols_with_both = [c for c in range(separated.shape[1]) if (separated[:, c] == 0).all()]
        self.assertGreater(len(cols_with_both), overlap.any(axis=0).sum())   # gap wider than the raw overlap
        self.assertEqual(set(np.unique(separated)) - {0}, {1, 2})

    def test_instance_labels_follow_task(self):
        self.assertEqual(instance_labels(toy_xray(), "quadrant"), [1, 2])
        self.assertEqual(instance_labels(toy_xray(), "disease"), [2, 0])

    def test_load_dentex_json(self):
        data = {"images": [{"id": 7, "file_name": "a.png", "width": 10, "height": 10}],
                "annotations": [{"image_id": 7, "bbox": [1, 1, 3, 3], "segmentation": [square(1, 1, 4, 4)],
                                 "category_id_1": 2, "category_id_2": 4, "category_id_3": 0}]}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "train.json"
            path.write_text(json.dumps(data))
            xrays = load_dentex(path)
        self.assertEqual(xrays[0].annotations[0].quadrant, 2)
        self.assertEqual(semantic_mask(xrays[0], "enumeration").max(), encode("enumeration", 2, 4))


class TestPostprocess(unittest.TestCase):
    def test_touching_teeth_are_separated(self):
        pred = np.zeros((20, 40), dtype=np.uint8)
        pred[2:18, 2:38] = TOOTH
        pred[2:18, 19:21] = BORDER                                   # predicted border between two teeth
        inst = instances_from_border_prediction(pred, min_core_area=10)
        self.assertEqual(count_teeth(inst), 2)

    def test_dbscan_1d(self):
        labels = dbscan_1d(np.array([100, 105, 3, 98, 4]), eps=10)
        self.assertEqual(len(set(labels[[0, 1, 3]])), 1)
        self.assertNotEqual(labels[0], labels[2])

    def test_box_ignores_small_fragment(self):
        lm = np.zeros((50, 50), dtype=np.uint8)
        lm[10:30, 10:25] = 3          # the tooth
        lm[45:47, 45:47] = 3          # stray fragment
        (box,) = boxes_from_labelmap(lm)
        self.assertEqual((box.x, box.y, box.w, box.h), (10.0, 10.0, 15.0, 20.0))

    def test_lesions_sharing_a_label_get_separate_boxes(self):
        lm = np.zeros((60, 60), dtype=np.uint8)
        lm[5:15, 5:15] = 2
        lm[40:52, 40:50] = 2
        self.assertEqual(len(boxes_from_labelmap(lm, per_component=True)), 2)
        self.assertEqual(len(boxes_from_labelmap(lm)), 1)

    def test_export_scales_and_links_disease_to_tooth(self):
        tooth = Box(label=encode("enumeration", 2, 6), x=100, y=100, w=50, h=80, score=0.9)
        lesion = Box(label=2, x=110, y=120, w=20, h=20, score=0.7)
        recs = to_records(5, (1024, 1024), enumeration=[tooth], disease=[lesion], input_size=512)
        self.assertEqual(recs[0]["bbox"], [200.0, 200.0, 100.0, 160.0])
        self.assertEqual((recs[1]["category_id_1"], recs[1]["category_id_2"], recs[1]["category_id_3"]), (2, 6, 1))


class TestModelAndMetrics(unittest.TestCase):
    def test_unet_shapes_and_loss(self):
        model = UNet(1, 5, base=8).eval()
        x = torch.rand(2, 1, 64, 64)
        logits = model(x)
        self.assertEqual(tuple(logits.shape), (2, 5, 64, 64))
        loss = DiceCELoss(5)(logits, torch.randint(0, 5, (2, 64, 64)))["loss"]
        self.assertTrue(torch.isfinite(loss))

    def test_perfect_prediction_scores(self):
        target = np.random.randint(0, 4, (32, 32))
        meter = SegmentationMeter(4)
        meter.update(target, target)
        self.assertAlmostEqual(meter.summary()["dice"], 1.0, places=6)
        self.assertAlmostEqual(foreground_scores(target, target)["iou"], 1.0, places=6)


class TestData(unittest.TestCase):
    def test_flip_augmentation_remaps_quadrants(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "images").mkdir()
            (Path(d) / "masks").mkdir()
            mask = np.zeros((16, 16), np.uint8)
            mask[:, :8] = 1                                      # quadrant 1 on the image's left
            Image.fromarray(np.zeros((16, 16), np.uint8)).save(Path(d) / "images" / "x.png")
            Image.fromarray(mask).save(Path(d) / "masks" / "x.png")
            ds = SegmentationDataset(d, "quadrant", augment=True, hflip_p=1.0, noise_p=0.0)
            flipped = ds[0]["mask"].numpy()
        self.assertTrue((flipped[:, 8:] == 2).all())             # moved right *and* relabelled as quadrant 2

    def test_binarize_maps_all_labels_to_tooth(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "images").mkdir()
            (Path(d) / "masks").mkdir()
            Image.fromarray(np.zeros((8, 8), np.uint8)).save(Path(d) / "images" / "x.png")
            Image.fromarray(np.array([[0, 7, 31, 2]] * 8, np.uint8).repeat(2, axis=1)).save(Path(d) / "masks" / "x.png")
            mask = SegmentationDataset(d, None, binarize=True)[0]["mask"].numpy()
        self.assertEqual(set(np.unique(mask)), {0, 1})

    def test_nnunet_raw_layout(self):
        with tempfile.TemporaryDirectory() as d:
            for sub in ("images", "masks"):
                (Path(d) / sub).mkdir()
                Image.fromarray(np.zeros((8, 8), np.uint8)).save(Path(d) / sub / "train_1.png")
            root = build_raw_dataset("disease", Path(d) / "images", Path(d) / "masks", Path(d) / "raw")
            spec = json.loads((root / "dataset.json").read_text())
            self.assertTrue((root / "imagesTr" / "DENTEX_0001_0000.png").exists())
            self.assertTrue((root / "labelsTr" / "DENTEX_0001.png").exists())
        self.assertEqual(root.name, "Dataset045_DENTEX")
        self.assertEqual(spec["labels"]["deep_caries"], 4)


if __name__ == "__main__":
    unittest.main()
