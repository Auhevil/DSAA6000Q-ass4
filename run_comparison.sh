#!/bin/bash

echo 'Running original model...'
python src/compare_models.py original

echo 'Running DPO model...'
python src/compare_models.py dpo

echo 'Combining results...'
python src/compare_models.py combine
