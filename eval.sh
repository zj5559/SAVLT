#!/bin/bash

#####################################

script='savlt'
config='SAVLT-B'
python tracking/test.py ${script} ${config} --dataset lasot_ext_lang --threads 2
python tracking/test.py ${script} ${config} --dataset otb99_lang --threads 2
python tracking/test.py ${script} ${config} --dataset lasot_lang --threads 2
python tracking/test.py ${script} ${config} --dataset tnl2k --threads 2


