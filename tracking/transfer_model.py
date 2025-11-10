import torch
import collections
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

path1 = '/home/cx/cx1/GOKU_repo/Goku/checkpoints/train/gohan/gohan_b224_vast3/GOHAN_ep0500.pth.tar'
model1 = torch.load(path1)
constructor1 = model1['constructor']
net1 = model1['net']
model2 = {}
model2['net'] = net1
model2['constructor'] = constructor1
torch.save(model2, '/home/cx/cx1/GOKU_repo/Goku4/checkpoints/train/gohan/gohan_b224_vast3/GOHAN_ep0500.pth.tar')
#

# # names = ['seqtrack_b256', 'seqtrack_b256_got', 'seqtrack_b384', 'seqtrack_b384_got', 'seqtrack_l256', 'seqtrack_l256_got', 'seqtrack_l384', 'seqtrack_l384_got']
# names = ['seqtrack_l256_got', 'seqtrack_l384', 'seqtrack_l384_got']
#
# for name in names:
#     path1 = '/home/cx/cx1/MSRA/CLOUD/MyExperiments/SeqTrack/checkpoints/train/seqtrack_old/' + name + '/SEQTRACK_ep0005.pth.tar'
#     model1 = torch.load(path1)
#     # path2 = '/home/cx/cx1/MSRA/CLOUD/MyExperiments/PlayGround/checkpoints/train/vt/v_l_16_384_fb_bs16/VT_ep0020.pth.tar'
#     # model2 = torch.load(path2)
#     net1 = model1['net']
#     a=1
#
#     net1_new = collections.OrderedDict()
#     net1_keys = net1.keys()
#     for key in net1_keys:
#         if key.split('.')[0] == 'backbone':
#             key_new = key.replace('backbone', 'encoder', 1)
#         elif key.split('.')[0] == 'transformer':
#             key_new = key.replace('decoder', 'body', 1)
#             key_new = key_new.replace('transformer', 'decoder', 1)
#         else:
#             key_new = key
#         net1_new[key_new] = net1[key]
#
#     model1['net'] = net1_new
#     model1['epoch'] = 500
#     model1['actor_type'] = 'SeqTrackActor'
#     model1['net_type'] = 'SEQTRACK'
#     torch.save(model1, '/home/cx/cx1/MSRA/CLOUD/MyExperiments/SeqTrack/checkpoints/train/seqtrack_new/' + name + '/SEQTRACK_ep0005.pth.tar')



# torch.save(net1, '/home/cx/cx1/MSRA/CLOUD/MyExperiments/PlayGround/checkpoints/train/pstmt/v_l_16_256_tr4_1s2t_x_e5_6m_bs8_all/PSTMT_ep0005.pth')

# net2 = model2['net']
# path3 = '/home/cx/cx1/MSRA/CLOUD/MyExperiments/PlayGround/debug/x.pth'
# input1 = torch.load(path3)
# path4 = '/home/cx/cx1/MSRA/CLOUD/MyExperiments/PlayGround/debug/x_3090.pth'
# input2 = torch.load(path4)
# for key in net2.keys():
#     a = ((net1[key] == net2[key]) + 0).min()
#     if a == 0:
#         print(key)

# path3 = '/home/cx/cx1/MSRA/CLOUD/MyExperiments/PlayGround/checkpoints/train/vtmt2/v_l_16_256/VTMT_ep0050.pth.tar'
# model3 = torch.load(path3)
# model3 = {'net':collections.OrderedDict()}
# model3['net'].update(model1['net'])
# model3['net'].update(model2['net'])
# torch.save(model3, '/home/cx/cx1/MSRA/CLOUD/MyExperiments/PlayGround/checkpoints/train/vtmt2/v_l_16_256_bs16/VTMT_ep0050.pth.tar')

# constructor1 = model1['constructor']
# model2 = {}
# model2['net'] = net1
# model2['constructor'] = constructor1
# torch.save(model2, '/home/cx/cx1/MSRA/CLOUD/MyExperiments/PlayGround/checkpoints/train/vittrack_baseline/vit_l_16_256_bs16/model.pth')
#

