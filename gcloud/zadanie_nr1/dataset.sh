#!/bin/bash

gsutil -m cp -r gs://snr/dataset.tar /
tar -xf /dataset.tar
