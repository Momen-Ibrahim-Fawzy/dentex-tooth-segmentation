# Project timeline (weekly progress reviews, Nov 2023 – Jan 2024)

| Week of | Work |
|---|---|
| 20 Nov 2023 | Downloaded DENTEX (≈13.7 GB) and analysed the three annotation levels; studied the HierarchicalDet baseline (diffusion-based hierarchical detection) |
| 27 Nov | U-Net tooth-segmentation baseline (Helli & Hamamci) reproduced on its original dataset (with Menna ElKammah) and applied to DENTEX |
| 4 Dec | Built DENTEX segmentation masks from the polygon annotations, resized to 512×512; measured the baseline on DENTEX without fine-tuning |
| 11 Dec | Trained one U-Net per task (quadrant, enumeration, disease) on the new masks |
| 18 Dec | Removed overlapping regions between neighbouring teeth and retrained; reviewed nnU-Net, SAM/AutoSAM and nnSAM |
| 25 Dec | Added a border class for instance separation; planned two routes to nnSAM (official code vs SAM embeddings in our U-Net) |
| 1 Jan 2024 | Multi-class masks with borders; set up nnSAM on DENTEX (fingerprinting, planning); wrote the converter to the challenge's prediction format |
| 8 Jan | Fixed mask errors; nnSAM training started |
| 15 Jan | nnSAM 5-fold training for all three tasks; box extraction with DBSCAN |
| 22 Jan | All folds complete; Dice results and qualitative evaluation on the 70 held-out X-rays; paper draft |
