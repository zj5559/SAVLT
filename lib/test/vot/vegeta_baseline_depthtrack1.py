import os
import sys
env_path = os.path.join(os.path.dirname(__file__), '../../..')
if env_path not in sys.path:
    sys.path.append(env_path)
from lib.test.vot.vegeta_class import run_vot_exp
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

run_vot_exp('vegeta_s2', 'vegeta_s2_b224_nlptnl2k_mul', vis=False, out_conf=True, channel_type='rgbd')
# run_vot_exp('seqtrack', 'seqtrack_b256', vis=False, out_conf=True, channel_type='rgb')
