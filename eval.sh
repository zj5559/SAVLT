#!/bin/bash

#####################################

script='savlt'
config='SAVLT-B'

python -m torch.distributed.launch --nproc_per_node 2 --script ${script} --config ${config} --save_dir YOUR_SAVE_PATH


python tracking/test.py ${script} ${config} --dataset lasot_ext_lang --threads 2
python tracking/test.py ${script} ${config} --dataset otb99_lang --threads 2
python tracking/test.py ${script} ${config} --dataset lasot_lang --threads 2
python tracking/test.py ${script} ${config} --dataset tnl2k --threads 2


