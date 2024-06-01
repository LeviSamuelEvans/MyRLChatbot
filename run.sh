#!/bin/bash

# a very very basic run script to run the code
export PYTHONPATH=$(pwd)/src
python src/data_prep.py
python src/train.py
