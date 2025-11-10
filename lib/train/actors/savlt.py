from . import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, box_xyxy_to_cxcywh, box_cxcywh_to_xyxy, box_iou
import torch
from lib.train.admin import multigpu
from lib.utils.heapmap_utils import generate_heatmap
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import numpy as np
def calculate_acc(pred_label,labels):
    acc = (pred_label == labels).sum().float() / len(labels)
    return acc.item()
def eval(pred_label,labels):
    if (pred_label.sum()==len(pred_label) and labels.sum()==len(pred_label)) or (pred_label.sum()==0 and labels.sum()==0):
        return 1.0, 1.0,1.0
    tnr = metrics.recall_score(labels, pred_label, pos_label=0)
    tpr = metrics.recall_score(labels, pred_label, pos_label=1)
    acc = metrics.accuracy_score(labels, pred_label)
    return acc,tpr,tnr
def eval1(pred_label,labels):
    # if (pred_label.sum()==len(pred_label) and labels.sum()==len(pred_label)) or (pred_label.sum()==0 and labels.sum()==0) \
    #         or (pred_label.sum() == 2*len(pred_label) and labels.sum() == 2*len(pred_label)):
    #     return 1.0, 1.0,1.0,1.0
    if 0 in labels:
        recall_invis = ((pred_label==labels) & (labels==0)).sum()/(labels==0).sum()
    else:
        recall_invis=1.0

    if 1 in labels:
        recall_vis = ((pred_label==labels) & (labels==1)).sum()/(labels==1).sum()
    else:
        recall_vis=1.0

    if 2 in labels:
        recall_distractor = ((pred_label==labels) & (labels==2)).sum()/(labels==2).sum()
    else:
        recall_distractor=1.0
    acc = metrics.accuracy_score(labels, pred_label)
    return acc,recall_invis,recall_vis,recall_distractor
class SAVLTActor(BaseActor):
    """ Actor for training the vegeta_s3"""
    def __init__(self, net, objective, loss_weight, settings, cfg):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg
        self.multi_modal_language = cfg.DATA.MULTI_MODAL_LANGUAGE

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'search_anno'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict = self.forward_pass(data)

        # compute losses
        loss, status = self.compute_losses(out_dict, data)

        return loss, status

    def forward_pass(self, data):
        b = data['search_images'].shape[1]   # n,b,c,h,w
        search_list = data['search_images'].view(-1, *data['search_images'].shape[2:]).split(b,dim=0)  # (n*b, c, h, w)
        template_list = data['template_images'].view(-1, *data['template_images'].shape[2:]).split(b,dim=0)
        template_anno_list = data['template_anno'].view(-1, *data['template_anno'].shape[2:]).split(b,dim=0)

        z,x=self.net(template_list=template_list,
                           search_list=search_list,
                           template_anno_list=template_anno_list,
                           mode='patch_embed') # encoder: patch_embedding

        if self.multi_modal_language:
            text = data['nlp_ids'].permute(1,0)
            text_len=data['num_nlp']
            text_src,text_mask = self.net(z=z,x=x,text_data=text,text_len=text_len,mode='text')

            # with torch.no_grad():
            #     text_src_local=self.net(text_data=data['nlp_ids_local'].permute(1,0), mode='text')
            #     text_src_curr = self.net(text_data=data['nlp_ids_curr'].permute(1, 0), mode='text')
            # if torch.isnan(text_src).any():
            #     raise ValueError("Network outputs is NAN! Stop Training")
        else:
            text_src = None
            text_mask=None

        enc_opt,enc_early = self.net(z=z,
                           x=x,
                           text_src=text_src,
                           text_mask=text_mask,
                           mode='encoder') # forward the encoder

        outputs,logit_scale= self.net(feature=enc_opt, mode="decoder")
        cls_index_batch = data['cls_label'].cuda().long()
        outputs['state_class_label'] = cls_index_batch  # [0:invisible 1:normal 2:distractor]
        outputs['logit_scale'] = logit_scale

        num_patch_x = np.power(self.cfg.DATA.SEARCH.SIZE // self.cfg.MODEL.ENCODER.STRIDE, 2)
        num_frames = self.cfg.DATA.SEARCH.NUMBER
        num_patch_z = np.power(self.cfg.DATA.TEMPLATE.SIZE // self.cfg.MODEL.ENCODER.STRIDE, 2)
        num_template = self.cfg.DATA.TEMPLATE.NUMBER
        if self.cfg.TRAIN.CLIP_FEAT_STAGE == 'self':
            feat_clip = enc_early
        elif self.cfg.TRAIN.CLIP_FEAT_STAGE == 'cross':
            feat_clip = enc_opt[0]
        if self.cfg.MODEL.ENCODER.CLASS_POS=='end':
            outputs['img_search_clip'] =feat_clip[:,0:num_patch_x*num_frames,:]
            outputs['img_template_clip']=feat_clip[:,num_patch_x*num_frames:
                                                     (num_patch_x*num_frames+num_patch_z),:]
            outputs['text_global_clip']=feat_clip[:,-2:-1,:]
            outputs['cls_feat'] = feat_clip[:, -1:, :]

        elif self.cfg.MODEL.ENCODER.CLASS_POS=='start':
            outputs['img_search_clip'] = feat_clip[:, 1:num_patch_x * num_frames+1, :]
            outputs['img_template_clip'] = feat_clip[:, num_patch_x * num_frames+1:
                                                        (num_patch_x * num_frames + num_patch_z )+1,:]
            outputs['text_global_clip'] = feat_clip[:, -1:,:]
            outputs['cls_feat'] = feat_clip[:, :1, :]

        return outputs

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        # task classification loss
        # task_cls_loss = self.objective['task_cls'](pred_dict['task_class'], pred_dict['task_class_label'])

        # gt gaussian map
        gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        gt_visible=(pred_dict['state_class_label']==1).squeeze()
        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.ENCODER.STRIDE) # list of torch.Size([b, H, W])
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1) # torch.Size([b, 1, H, W])

        # Get boxes
        pred_boxes = pred_dict['pred_boxes'] # torch.Size([b, 1, 4])
        # if torch.isnan(pred_boxes).any():
        #     raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                           max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec[gt_visible.bool()], gt_boxes_vec[gt_visible.bool()])  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # giou_loss[(1-gt_visible).bool()]=torch.tensor(0.0).cuda()
        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec[gt_visible.bool()], gt_boxes_vec[gt_visible.bool()])  # (BN,4) (BN,4)
        # l1_loss[(1 - gt_visible).bool()] = torch.tensor(0.0).cuda()
        # compute location loss
        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](pred_dict['score_map'][gt_visible.bool()], gt_gaussian_maps[gt_visible.bool()])
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)
        # location_loss[(1 - gt_visible).bool()] = torch.tensor(0.0).cuda()
        # weighted sum
        loss = (self.loss_weight['giou'] * giou_loss +
                self.loss_weight['l1'] * l1_loss +
                self.loss_weight['focal'] * location_loss)
        if self.loss_weight['clip']:
            if self.cfg.TRAIN.CLIP_WITH=='search':
                clip_loss = self.objective['clip'](pred_dict['img_search_clip'], pred_dict['text_global_clip'],
                                                pred_dict['logit_scale'])
            elif self.cfg.TRAIN.CLIP_WITH=='template':
                clip_loss = self.objective['clip'](pred_dict['img_template_clip'], pred_dict['text_global_clip'],
                                                pred_dict['logit_scale'])
            elif self.cfg.TRAIN.CLIP_WITH=='cls':
                clip_loss = self.objective['clip'](pred_dict['cls_feat'], pred_dict['text_global_clip'],
                                                pred_dict['logit_scale'])
        else:
            clip_loss = torch.tensor(0.0, device=l1_loss.device)
        all_loss=loss
        all_loss = all_loss + self.loss_weight['clip'] * clip_loss

        rate_invis = (pred_dict['state_class_label'] == 0).sum() / len(pred_dict['state_class_label'])
        rate_distractor = (pred_dict['state_class_label'] == 2).sum() / len(pred_dict['state_class_label'])
        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {"Loss/total": all_loss.item(),
                      "Loss/tracking": loss.item(),
                      "Loss/clip": clip_loss.item(),
                      "IoU": mean_iou.item(),
                      'clip_logit_scale': pred_dict['logit_scale'],
                      "rate_invis": rate_invis,
                      "rate_distractor": rate_distractor
                      }
            return all_loss, status
        else:
            return loss