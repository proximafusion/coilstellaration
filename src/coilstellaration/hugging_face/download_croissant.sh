#!/bin/bash

 curl https://huggingface.co/api/datasets/proxima-fusion/coilstellaration/croissant \
        -s \
        -X GET \
        -H "Authorization: Bearer ${HF_API_TOKEN}" | tee coilstellaration.croissant.json
