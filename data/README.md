# Data directory

Third-party images are not included in this repository.

Recommended layout:

```text
data/
├── raw/
│   └── figshare/        # downloaded/extracted source files
├── metadata.csv         # generated manifest
└── smoke/               # optional synthetic smoke-test data
```

The expected manifest columns are documented in the root README and in `docs/DATASET_CARD.md`.
