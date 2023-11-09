#!/usr/bin/env bash

CONFIG=$1

python  basicsr/train.py -opt $CONFIG 
