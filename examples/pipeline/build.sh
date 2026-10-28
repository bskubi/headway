#!/bin/sh
# The headway of the pipeline: its steps' code is in Nextflow, Snakemake, bash, R and Rust.
set -e
cd "$(dirname "$0")"
${PYTHON:-python} -m headway.cli build spec -o pipeline.html
