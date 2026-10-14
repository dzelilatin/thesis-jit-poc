#!/bin/bash
# Download ApacheJIT dataset from Zenodo
set -e
cd "$(dirname "$0")/.."
mkdir -p data
curl -L -o data/apachejit.zip \
  "https://zenodo.org/records/5907847/files/apachejit_dataset_replication.zip?download=1"
unzip -o data/apachejit.zip -d data/
echo "Done. CSV at data/apachejit/dataset/apachejit_total.csv"
