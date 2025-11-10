import os

import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text, load_str


class TNL2kDataset(BaseDataset):
    """
    TNL2k test set
    """
    def __init__(self,onlyNLP=False):
        super().__init__()
        self.base_path = self.env_settings.tnl2k_path
        self.sequence_list = self._get_sequence_list()
        self.onlyNLP = onlyNLP
        self.grounding_path = '/media/zj/T9/models/grounding_results/jointnlt_grounding /tnl2k'

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        # class_name = sequence_name.split('-')[0]
        anno_path = '{}/{}/groundtruth.txt'.format(self.base_path, sequence_name)

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)
        if self.onlyNLP:
            grounding_rect=load_text(os.path.join(self.grounding_path,sequence_name,'init_grounding.txt'),
                                     delimiter=',',dtype=np.float64)
            ground_truth_rect[0,:]=grounding_rect

        text_dsp_path = '{}/{}/language.txt'.format(self.base_path, sequence_name)
        text_dsp = load_str(text_dsp_path)

        frames_path = '{}/{}/imgs'.format(self.base_path, sequence_name)
        frames_list = [f for f in os.listdir(frames_path)]
        frames_list = sorted(frames_list)
        frames_list = ['{}/{}'.format(frames_path, frame_i) for frame_i in frames_list]

        if self.onlyNLP:
            dataset_name = 'tnl2k_onlynl'
        else:
            dataset_name = 'tnl2k'
        # target_class = class_name
        # return Sequence(sequence_name, frames_list, 'tnl2k', ground_truth_rect.reshape(-1, 4), text_dsp=text_dsp)
        return Sequence(sequence_name, frames_list, dataset_name, ground_truth_rect.reshape(-1, 4),
                        language_query=text_dsp)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        sequence_list = []
        for seq in os.listdir(self.base_path):
            if os.path.isdir(os.path.join(self.base_path, seq)):
                sequence_list.append(seq)

        return sequence_list
