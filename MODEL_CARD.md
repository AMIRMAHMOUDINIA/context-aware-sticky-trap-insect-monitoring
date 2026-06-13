# Pre-Experiment Model Card

- Software version: `0.4.0`

## Status

**Implementation ready; real-data evaluation pending.**

The repository can train and evaluate the models described below, but no production or biological performance claim should be made until the public dataset has been downloaded, audited, and used in the documented experiments.

## Model family

Two main model variants are supported:

1. **Image-only classifier** — a pretrained visual backbone followed by a small classification head.
2. **Context-aware classifier** — the same visual backbone combined with trainable categorical embeddings for acquisition device and sticky-trap color.

Supported backbones include ResNet-18, MobileNetV3-Small, and EfficientNet-B0. A small custom CNN is included only for smoke tests and CPU debugging. Missing or unseen context categories map to a fixed zero embedding so that a held-out domain does not inject random untrained context into the classifier.

## Intended use

The models are intended for research on:

- insect image classification;
- acquisition-context effects;
- generalization to unseen cameras or trap backgrounds;
- calibration and confidence analysis;
- retrospective active-learning simulation;
- qualitative error inspection with Grad-CAM.

## Out-of-scope use

The models are not intended for:

- autonomous pesticide or crop-protection decisions;
- definitive taxonomic identification without expert review;
- direct transfer to new crop systems without validation;
- abundance estimation from full field sticky traps;
- open-set recognition of unseen species;
- operational deployment without drift monitoring and retraining procedures.

## Training data

Planned source dataset:

- Ong and Høye (2024), Figshare DOI `10.6084/m9.figshare.23617383.v2`;
- two stored-product pest species;
- three acquisition devices;
- four sticky-trap colors;
- controlled image-acquisition conditions.

The third-party images are not included in this repository.

## Split and leakage requirements

Before training:

- verify species, device, trap-color, and split labels;
- identify repeated photographs of the same specimen or physical trap;
- assign a shared `group_id` where related images must remain together;
- preserve authoritative source splits when available;
- record the dataset version and manifest checksum;
- save the exact split assignment and reject non-empty group identifiers that cross splits unless the study design explicitly justifies that pairing.

## Planned evaluation

Primary metric:

- macro-F1.

Secondary metrics:

- balanced accuracy;
- per-class precision and recall;
- log loss;
- expected calibration error;
- confusion matrix;
- reliability diagram.

Robustness evaluation:

- leave-one-device-out;
- leave-one-trap-color-out;
- repeated runs with documented random seeds.

Data-efficiency evaluation:

- random sampling versus predictive-entropy sampling at equal annotation budgets.

## Known risks and limitations

- Acquisition metadata may improve prediction while also encouraging shortcut learning.
- Controlled images may be substantially cleaner than operational field-trap images.
- A two-species classifier does not address non-target arthropods or unknown classes.
- Confidence estimates can become unreliable under domain shift.
- Grad-CAM is a qualitative diagnostic and should not be treated as proof of biological reasoning.
- Retrospective active learning does not reproduce all costs and decisions of a real expert-labeling workflow.

## Human oversight

Predictions should support, not replace, trained entomologists and integrated pest-management professionals. Any operational use would require domain-specific validation, false-negative analysis, drift monitoring, expert review rules, and a documented retraining process.

## Information to complete after experiments

- model architecture and backbone;
- pretraining weights;
- training date and code commit;
- dataset version and manifest checksum;
- train/validation/test sizes;
- held-out domains;
- final metrics with uncertainty;
- subgroup and error analyses;
- released checkpoint location.
