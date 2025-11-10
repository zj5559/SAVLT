#!/bin/bash


script='savlt'
config='SAVLT-B'
python -m torch.distributed.launch --nproc_per_node 2 lib/train/run_training.py --script ${script} --config ${config} --save_dir YOUR/PATH



