#!/usr/bin/env bash
# Downloads the NIH malaria cell dataset from Kaggle into data/.
# Needs a Kaggle API token at ~/.kaggle/kaggle.json (see README).
set -euo pipefail
mkdir -p data
kaggle datasets download -d iarunava/cell-images-for-detecting-malaria -p data --unzip
echo "Done. Images should be in data/cell_images/{Parasitized,Uninfected}/"
