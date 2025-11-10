#!/bin/bash

script='savlt'
config='SAVLT-B-NL'
python tracking/test.py ${script} ${config} --dataset lasot_ext_lang_onlynl --threads 2
python tracking/test.py ${script} ${config} --dataset otb99_lang_onlynl --threads 2
python tracking/test.py ${script} ${config} --dataset tnl2k_onlynl --threads 2
python tracking/test.py ${script} ${config} --dataset lasot_lang_onlynl --threads 2
