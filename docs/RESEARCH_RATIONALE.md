# Research Rationale

## Starting point

Insect-recognition models are often evaluated under random train/test splits in which the same cameras, backgrounds, and acquisition conditions appear on both sides of the split. This can produce optimistic results when a model learns stable visual cues that are specific to the dataset rather than to insect morphology.

This project treats acquisition context as both a possible source of useful information and a possible source of bias.

## Central question

Can device and sticky-trap color improve insect classification without making the model less robust when those conditions change?

## Working hypotheses

1. Adding context may improve classification and calibration when training and testing contain the same domains.
2. The same context may reduce robustness when a new camera or trap color is encountered.
3. Leave-one-domain-out evaluation will expose weaknesses that are hidden by ordinary random splits.
4. Entropy sampling may achieve similar performance with fewer labeled images than random selection, although uncertainty itself may become unreliable under domain shift.
5. Grad-CAM and prediction-level error review may reveal background or device shortcuts that are not obvious from aggregate metrics.

## What would count as a useful result?

A useful result does not have to show that context-aware modeling always performs better. The project is informative if it identifies:

- when context improves prediction;
- when context creates shortcut learning;
- which devices or trap colors cause the largest performance loss;
- whether model confidence remains calibrated;
- whether uncertainty sampling is consistently more efficient than random selection;
- which image characteristics dominate high-confidence errors.

## What should not be inferred

The planned experiments do not establish operational performance in crop fields. The source data contain two stored-product pests under controlled imaging conditions. Any transfer to open-field or greenhouse monitoring would require new data, additional species, field-level validation, and integration with ecological and crop-management context.
