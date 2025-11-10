import os
import sys
env_path = os.path.join(os.path.dirname(__file__), '../../..')
if env_path not in sys.path:
    sys.path.append(env_path)
from lib.test.vot.vegeta_class import run_vot_exp
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

run_vot_exp('vegeta_s2', 'vegeta_s2_b224_nlp', vis=False, out_conf=True, channel_type='rgbd')
# run_vot_exp('seqtrackv2_peft', 'seqtrackv2_b256_stt444_m', vis=False, out_conf=True, channel_type='rgb')
