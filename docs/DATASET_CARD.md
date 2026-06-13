# Dataset Card

## Dataset identity

- Title: *Annotated images of pests on different coloured sticky traps with different imaging devices*
- Authors: Song-Quan Ong and Toke Thomas Høye
- Dataset DOI: `10.6084/m9.figshare.23617383.v2`
- Related article DOI: `10.1016/j.dib.2024.110741`
- Repository data included here: **No**

## Intended use in this repository

The dataset is used to study image classification under acquisition-context variation. The target is insect species, while device and sticky-trap colour are treated as context and domain variables.

## Expected variables

- `image_path`
- `species`
- `device`
- `trap_color`
- `split`
- `group_id`

## Known scope

The source documentation describes two stored-product insect pests photographed with three acquisition devices on four sticky-trap colours. Images were prepared for deep-learning experiments under controlled acquisition conditions.

## Risks and limitations

- Controlled images may be cleaner than operational sticky-trap images.
- The number of species is small.
- Device and background can become shortcut features.
- Existing source splits may not control for repeated images of the same specimen unless explicitly documented.
- The dataset does not represent Belgian-endive or carrot pest communities.
- The related publication also constructed mixed-color subsets; the downloaded folder structure must be checked before deciding whether these are treated as an additional domain or excluded from four-color holdout experiments.
- Non-target arthropods and open-set recognition are not addressed.

## Required validation

Before training, verify the downloaded folder structure and compare label counts against the source documentation. Populate `group_id` if related images exist. Review exact-file duplicate groups from `scripts/audit_manifest.py`, and save the audit output with each research release.

## Licence and citation

Consult the Figshare record for current licence terms. Cite both the dataset and related article in any report or derived publication.
