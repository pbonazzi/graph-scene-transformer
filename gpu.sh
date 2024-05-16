#!/bin/bash
gpustat
echo Which GPU device should I use?
read id
export CUDA_VISIBLE_DEVICES=$id
git pull

