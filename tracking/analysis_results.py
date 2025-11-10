import _init_paths
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist

trackers = []

dataset_name = 'lasot_ext_lang'
dataset_name_result = dataset_name

# choosen from 'lasot_ext_lang', 'lasot_lang', 'otb99_lang', 'tnl2k'

#NL+BBOX
trackers.extend(trackerlist(name='savlt', parameter_name='SAVLT-B', dataset_name=dataset_name_result,
                            run_ids=None, display_name='SAVLT-B'))
trackers.extend(trackerlist(name='savlt', parameter_name='SAVLT-L', dataset_name=dataset_name_result,
                            run_ids=None, display_name='SAVLT-L'))

#NL
trackers.extend(trackerlist(name='savlt', parameter_name='SAVLT-B-BBOX', dataset_name=dataset_name_result,
                            run_ids=None, display_name='SAVLT-B-BBOX'))
trackers.extend(trackerlist(name='savlt', parameter_name='SAVLT-L-BBOX', dataset_name=dataset_name_result,
                            run_ids=None, display_name='SAVLT-L-BBOX'))


#_onlynl
# dataset_name_result = dataset_name+'_onlynl'
# trackers.extend(trackerlist(name='savlt', parameter_name='SAVLT-B-NL', dataset_name=dataset_name_result,
#                             run_ids=None, display_name='SAVLT-B-NL'))
# trackers.extend(trackerlist(name='savlt', parameter_name='SAVLT-L-NL', dataset_name=dataset_name_result,
#                             run_ids=None, display_name='SAVLT-L-NL'))

dataset = get_dataset(dataset_name)

print_results(trackers, dataset, dataset_name_result, merge_results=True,seq_eval=False, plot_types=('success', 'prec', 'norm_prec'),
              force_evaluation=True)

