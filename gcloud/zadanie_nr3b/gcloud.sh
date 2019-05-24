#!/usr/bin bash

gcloud ai-platform jobs submit training \
    zalando_v100_1 \
    --module-name=zalando_ds.zalando_ds \
    --package-path=zalando_ds \
    --job-dir=gs://snr/zalando_v100_1 \
    --config zalando_ds/config.json

