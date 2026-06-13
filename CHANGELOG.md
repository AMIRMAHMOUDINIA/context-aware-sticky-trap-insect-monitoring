# Changelog

## 0.4.0 — 2026-06-13

- blocked absolute and parent-directory image paths from escaping the configured data root;
- added unsafe-path reporting to the dataset audit;
- added strict configuration validation for split fractions, augmentation probabilities, and active-learning settings;
- added class-coverage checks for validation and test splits;
- made macro-F1 use the full declared class set, including classes absent from a subgroup;
- improved Figshare download safety by verifying existing files and cleaning failed partial downloads;
- added Grad-CAM target-class bounds checking;
- expanded the automated suite from 11 to 19 tests;
- updated documentation and package metadata for the clean replacement upload.

## 0.3.0 — 2026-06-13

- clarified the relationship to the published dataset study and the distinct research contribution;
- made unseen or missing context embeddings neutral rather than randomly initialized;
- paired active-learning training seeds across strategies at each repeat and round;
- added group-overlap enforcement for repeated specimens or traps;
- added exact-file duplicate detection to the data audit;
- saved split assignments and manifest checksums with each training run;
- added device- and trap-color subgroup metrics and a training-history figure;
- expanded automated tests to cover Grad-CAM, group leakage checks, neutral context, and active learning;
- aligned package version metadata and improved installation instructions.

## 0.2.0 — 2026-06-13

- humanized the research rationale and README;
- added dataset and model documentation;
- added configurable baseline, context-aware, domain-shift, active-learning, and Grad-CAM workflows.
