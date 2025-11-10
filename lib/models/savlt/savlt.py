"""
Vegeta_s3 Model
"""
import torch
import math
import os
from torch import nn
import torch.nn.functional as F
from .encoder import build_encoder
from .clip import build_textencoder
from .decoder import build_decoder
from .task_decoder import build_task_decoder,build_text_decoder
from lib.utils.box_ops import box_xyxy_to_cxcywh
from lib.utils.pos_embed import get_sinusoid_encoding_table, get_2d_sincos_pos_embed
import numpy as np


class SAVLT(nn.Module):
    """ This is the base class for BASELINE_MODEL """
    def __init__(self, text_encoder, encoder, decoder,
                 num_frames=1, num_template=1,
                 decoder_type="CENTER", task_feature_type="average",clip_loss=False,temp_learn=False):
        """ Initializes the model.
        """
        super().__init__()
        self.encoder = encoder
        self.text_encoder = text_encoder
        self.decoder_type = decoder_type

        self.class_token = False if (encoder.body.cls_token is None) else True
        self.class_pos=encoder.body.cls_pos
        self.task_feature_type = task_feature_type

        self.num_patch_x = self.encoder.body.num_patches_search
        self.num_patch_z = self.encoder.body.num_patches_template
        self.fx_sz = int(math.sqrt(self.num_patch_x))
        self.fz_sz = int(math.sqrt(self.num_patch_z))

        self.decoder = decoder

        self.num_frames = num_frames
        self.num_template = num_template

        if clip_loss:
            init_logit_scale=np.log(1 / 0.07)
            if temp_learn:
                self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
            else:
                self.logit_scale = torch.ones([]) * init_logit_scale
        else:
            self.logit_scale=torch.ones([])

    def get_logit_scale(self):
        return self.logit_scale.exp()
    def forward(self, text_data=None,text_len=None,
                template_list=None, search_list=None, template_anno_list=None,z=None,x=None,
                text_src=None,text_mask=None,
                feature=None, mode="encoder"):
        """
        image_list: list of template and search images, template images should precede search images
        xz: feature from encoder
        seq: input sequence of the decoder
        mode: encoder or decoder.
        """
        if mode == "text":
            return self.forward_textencoder(z,x,text_data,text_len)
        elif mode =='patch_embed':
            return self.forward_encoder_early(template_list=template_list, search_list=search_list, template_anno_list=template_anno_list,mode='patch_embed')
        elif mode == "encoder":
            return self.forward_encoder(z=z,x=x, text_src=text_src,text_mask=text_mask,mode='encoder')
        elif mode == "decoder":
            return self.forward_decoder(feature),self.get_logit_scale()
        else:
            raise ValueError

    def forward_textencoder(self, z,x,text_data,text_len):
        # Forward the encoder
        text_src,text_mask = self.text_encoder(z,x,text_data,text_len)

        return text_src,text_mask

    def forward_encoder_early(self, template_list, search_list, template_anno_list,mode='patch_embed'):
        # Forward the encoder
        z,x = self.encoder(template_list=template_list, search_list=search_list, template_anno_list=template_anno_list, mode=mode)
        return z,x

    def forward_encoder(self, z,x, text_src,text_mask=None,mode='encoder'):
        # Forward the encoder
        xz,xz_ori = self.encoder(z=z,x=x,text_src=text_src,text_mask=text_mask,mode=mode)
        return xz,xz_ori

    def forward_decoder(self, feature, gt_score_map=None):

        feature = feature[0]
        if self.class_token and self.class_pos=='start':
            feature = feature[:,1:self.num_patch_x * self.num_frames+1]
        else:
            feature = feature[:,0:self.num_patch_x * self.num_frames] # (B, HW, C)

        bs, HW, C = feature.size()
        if self.decoder_type in ['CORNER', 'CENTER']:
            feature = feature.permute((0, 2, 1)).contiguous()
            feature = feature.view(bs, C, self.fx_sz, self.fx_sz)
        if self.decoder_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.decoder(feature, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.decoder_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.decoder(feature, gt_score_map)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        elif self.decoder_type == "MLP":
            # run the mlp head
            score_map, bbox, offset_map = self.decoder(feature, gt_score_map)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError

def build_savlt(cfg,training=True):


    encoder = build_encoder(cfg)
    if cfg.DATA.MULTI_MODAL_LANGUAGE:
        text_encoder = build_textencoder(cfg, encoder)
    else:
        text_encoder = None
    decoder = build_decoder(cfg, encoder)

    if cfg.TRAIN.CLIP_WEIGHT==0:
        clip_loss=False
    else:
        clip_loss=True
    model = SAVLT(
        text_encoder,
        encoder,
        decoder,
        num_frames = cfg.DATA.SEARCH.NUMBER,
        num_template = cfg.DATA.TEMPLATE.NUMBER,
        decoder_type=cfg.MODEL.DECODER.TYPE,
        task_feature_type=cfg.MODEL.TASK_DECODER.FEATURE_TYPE,
        clip_loss=clip_loss,
        temp_learn=cfg.TRAIN.CLIP_TEMP_LEARN
    )

    if cfg.MODEL.PRETRAIN_FILE is not None and training:
        current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
        pretrained_path = os.path.join(current_dir, '../../../pretrained')
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
        checkpoint = torch.load(pretrained, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)
        print('missing keys:',missing_keys)
    return model
