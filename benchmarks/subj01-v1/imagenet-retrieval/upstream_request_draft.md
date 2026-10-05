# Draft only — not sent

Subject: Minimal provenance needed for the released Subject1 ImageNet retrieval baseline

Thank you for releasing NeuroAdapter and whole_brain_encoder. We are reproducing NeuroAdapter §6.1 using measured fMRI queries, the full 1,281,167-image ILSVRC2012 training set, and the released Subject1 ensemble (layers 1/3/5/7 × runs 1/2 × LH/RH).

We have verified all 16 checkpoint files and 16 validation-correlation files against the official Hugging Face LFS hashes at revision d8a978abb212eb2965b5d01673f96536b77e2ea0. All members load strictly and pass batched-versus-single forward checks using the public Schaefer assets and frozen DINOv2 ViT-B/14. We do not need new model training or replacement weights.

Could you please provide or confirm these three small provenance items for these particular weights?

1. The actual train, validation, and held-out test NSD global image IDs, including the IDs used to compute each val_corr_nonavg.npy. If all 16 models share the same split, a single set of arrays is enough. Please specify 0/1-based indexing and how image repetitions map to the beta rows. metadata_sub-01.npy or the script that generated these fields would also work. We cannot infer the IDs from the 515 rows shown in imagenet_test_sim.ipynb, and will intersect the confirmed held-out IDs with our existing 982-image benchmark before evaluation.

2. The precise beta preprocessing used to train these checkpoints: NSD derivative and surface space, the axes and sessions used for centering/scaling, ddof/epsilon/nonfinite or constant-vertex handling, and whether repetitions are averaged before or after normalization. We have the original Subject1 fsaverage GLMdenoise_RR betas, so a formula or preprocessing script may be sufficient; if training-derived statistics are needed, please include those or enough information to reproduce them.

3. The vertex mask definition in the producer of ground_truth_sims/labeled_area/corr read by NeuroAdapter/notebooks/imagenet_test_sim.ipynb. Is it the union of the 24 canonical pRF/fLoc ROIs per hemisphere, or does it additionally include the seven streams labels? The public fetch helper ORs all ROI dictionary entries, whereas another labeled-area helper explicitly excludes streams. We reconstructed the public parcels/nsd_labels/labels_s01.pt masks and verified all 48 individually against the official NSD native-to-fsaverage mapping. The canonical union has 16,236 LH and 17,380 RH vertices. A mask, vertex counts/hash, or the producer code would resolve the ambiguity.

For reference, our source is whole_brain_encoder@767f25afc2240f568f4db8c0ce09604e6f83aa72, and we also examined NeuroAdapter's pinned encoder commit 7c7cd1336fe6b68860c412a14a7324a332cdad7d. One checkpoint example is enc_1/run_1/lh/checkpoint_nonavg.pth, SHA256 4f3b41d591a68038797897a77d1e571a16d749f5c361c06f57ae42a5eca5347e.

We will disclose remaining differences from the internal experiment, preserve the published ensemble weighting, and use the same verified evaluation pool for all methods. We are not requesting the internal experiment's private output files or claiming byte-identical reproduction. Thank you for any pointer to an existing public preparation script or asset we may have missed.
