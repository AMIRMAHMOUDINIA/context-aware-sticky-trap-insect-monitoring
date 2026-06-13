# Experiment Plan

## Objective

This plan is complementary to Ong and Høye (2025), who evaluated trap-color, device, architecture, transfer-learning, and Grad-CAM effects on the same image resource. The present experiments focus on explicit metadata fusion, calibration, domain holdouts, annotation efficiency, and reproducible leakage controls.


The project asks a practical question: can image-acquisition context improve insect classification without making the model dependent on a particular camera or sticky-trap background? The experiments below compare image-only and context-aware models under both familiar and unseen acquisition conditions.

## Data source

Primary dataset: Ong and Høye (2024), Figshare record `10.6084/m9.figshare.23617383.v2`.

The images represent two insect species, three acquisition devices, and four sticky-trap colours. The dataset was created under controlled acquisition conditions. It is useful for controlled domain-shift experiments but is not equivalent to operational field monitoring.

## Data validation before modelling

1. Download and retain the original folder structure.
2. Generate `data/metadata.csv` with `scripts/build_manifest.py`.
3. Review every row marked `parse_status=review`.
4. Verify class, device, colour, and split counts against the source documentation.
5. Detect unreadable files and duplicate paths with `scripts/audit_manifest.py`.
6. Review exact duplicate files and determine whether multiple images originate from the same physical insect or sticky trap. Populate `group_id` whenever related images must remain in one split.
7. Record the manifest checksum, dataset version, and saved split assignment in the model card.

## Experiment 1 — Image-only baseline

**Model:** pretrained ResNet-18 with a small classification head.

**Comparison metric:** macro-F1 on the held-out test set.

**Secondary metrics:** balanced accuracy, per-class recall, log loss, expected calibration error.

**Purpose:** establish the performance attributable to visual information alone.

## Experiment 2 — Context-aware classifier

**Model:** same visual backbone plus trainable embeddings for device and trap colour.

**Hypothesis:** context can improve classification and calibration when the same domains occur in training and testing.

**Critical interpretation:** improved random-split performance may indicate useful context, shortcut learning, or both. Domain-shift experiments are required.

## Experiment 3 — Leave-one-device-out

For each device:

1. hold out all images from that device as the test set;
2. train on the remaining devices;
3. form a validation set without mixing related groups;
4. compare image-only and context-aware models;
5. report macro-F1, class recall, confidence, and calibration.

**Question:** can the model generalise to a camera not represented during training?

## Experiment 4 — Leave-one-trap-colour-out

Repeat the same procedure for blue, yellow, white, and transparent backgrounds.

**Question:** does the model learn insect morphology or depend on background colour?

## Experiment 5 — Active-learning simulation

1. Reserve fixed validation and test sets.
2. Begin with a small stratified labelled subset from the training pool.
3. Train the model.
4. Query new images either randomly or by predictive entropy. Use the same model seed for both strategies at each paired repeat and round.
5. Reveal their existing labels and retrain.
6. Repeat over equal annotation budgets and multiple random seeds.

**Primary result:** macro-F1 as a function of labelled-data fraction.

**Limitation:** this is a retrospective simulation, not a prospective expert-labelling study.

## Experiment 6 — Interpretation and error analysis

Generate Grad-CAM overlays for:

- correct high-confidence predictions;
- incorrect high-confidence predictions;
- uncertain images;
- each device and trap colour;
- each species.

Manually classify attention patterns as:

- insect morphology;
- sticky-trap background;
- image border or acquisition artefact;
- mixed or uninterpretable.

## Statistical reporting

Run each main comparison over at least three seeds. Report mean and standard deviation. Where models are evaluated on the same test images, compare per-image correctness or loss using paired procedures rather than treating runs as independent biological replicates.

Avoid presenting small numerical differences as meaningful without uncertainty and error inspection.

## Minimum figures

1. Class/device/colour distribution.
2. Image-only versus context-aware test performance.
3. Leave-one-device-out macro-F1.
4. Leave-one-colour-out macro-F1.
5. Active-learning curve.
6. Representative Grad-CAM and error-analysis panel.
