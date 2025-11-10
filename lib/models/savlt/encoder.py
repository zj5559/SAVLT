"""
Encoder modules: we use ITPN for the encoder.
"""

from torch import nn
from lib.utils.misc import is_main_process
from lib.models.savlt import vit as vit_module
import lib.models.savlt.dinov2.build_dinov2 as dinov2_module
from lib.models.savlt import fastitpn as fastitpn_module
from lib.models.savlt import itpn as oriitpn_module



class EncoderBase(nn.Module):

    def __init__(self, encoder: nn.Module, train_encoder: bool, open_layers: list, num_channels: int):
        super().__init__()
        open_blocks = open_layers[2:]
        open_items = open_layers[0:2]
        for name, parameter in encoder.named_parameters():

            if not train_encoder:
                freeze = True
                for open_block in open_blocks:
                    if open_block in name:
                        freeze = False
                if name in open_items:
                    freeze = False
                if freeze == True:
                    parameter.requires_grad_(False)  # here should allow users to specify which layers to freeze !

        self.body = encoder
        self.num_channels = num_channels

    def forward(self, template_list=None, search_list=None, template_anno_list=None, z=None,x=None,text_src=None,text_mask=None,mode='encoder'):
        xs = self.body(template_list=template_list, search_list=search_list, template_anno_list=template_anno_list, z=z,x=x,text_src=text_src,text_mask=text_mask,mode=mode)
        return xs


class Encoder(EncoderBase):
    """ViT encoder."""
    def __init__(self, name: str,
                 train_encoder: bool,
                 pretrain_type: str,
                 search_size: int,
                 search_number: int,
                 template_size: int,
                 template_number: int,
                 open_layers: list,
                 cfg=None):
        if "dinov2" in name.lower():
            encoder = getattr(dinov2_module, name)(pretrained=is_main_process(),
                                                   search_size=search_size, template_size=template_size,
                                                   search_number=search_number, template_number=template_number,
                                                   drop_path_rate=cfg.MODEL.ENCODER.DROP_PATH,
                                                   use_checkpoint=cfg.MODEL.ENCODER.USE_CHECKPOINT,
                                                   pos_type=cfg.MODEL.ENCODER.POS_TYPE,
                                                   token_type_indicate=cfg.MODEL.ENCODER.TOKEN_TYPE_INDICATE,
                                                   task_indicate=cfg.MODEL.ENCODER.TASK_INDICATE,
                                                   task_num=cfg.MODEL.TASK_NUM
                                                   )
            if "vits14" in name:
                num_channels = 384
            elif "vitb14" in name:
                num_channels = 768
            elif "vitl14" in name:
                num_channels = 1024
            elif "vitg14" in name:
                num_channels = 1536
            else:
                num_channels = 768
        elif "vit" in name.lower():
            encoder = getattr(vit_module, name)(pretrained=is_main_process(), pretrain_type=pretrain_type,
                                                search_size=search_size, template_size=template_size,
                                                search_number=search_number, template_number=template_number,
                                                drop_path_rate=cfg.MODEL.ENCODER.DROP_PATH,
                                                use_checkpoint=cfg.MODEL.ENCODER.USE_CHECKPOINT,
                                                pos_type=cfg.MODEL.ENCODER.POS_TYPE,
                                                task_indicate=cfg.MODEL.ENCODER.TASK_INDICATE,
                                                task_num=cfg.MODEL.TASK_NUM
                                                )
            if "_base_" in name:
                num_channels = 768
            elif "_large_" in name:
                num_channels = 1024
            elif "_huge_" in name:
                num_channels = 1280
            else:
                num_channels = 768
        elif "fastitpn" in name.lower():
            encoder = getattr(fastitpn_module, name)(
                pretrained=is_main_process(),
                search_size=search_size,
                template_size=template_size,
                drop_rate=0.0,
                drop_path_rate=0.1,
                attn_drop_rate=0.0,
                init_values=0.1,
                drop_block_rate=None,
                use_mean_pooling=True,
                grad_ckpt=False,
                cls_token=cfg.MODEL.ENCODER.CLASS_TOKEN,
                cls_pos=cfg.MODEL.ENCODER.CLASS_POS,
                pos_type=cfg.MODEL.ENCODER.POS_TYPE,
                token_type_indicate=cfg.MODEL.ENCODER.TOKEN_TYPE_INDICATE,
                task_indicate=cfg.MODEL.ENCODER.TASK_INDICATE,
                task_num=cfg.MODEL.CLS_NUM,
                pretrain_type = cfg.MODEL.ENCODER.PRETRAIN_TYPE,
                patchembed_init = cfg.MODEL.ENCODER.PATCHEMBED_INIT,
                text_input_mode = cfg.MODEL.ENCODER.TEXT_INPUT_MODE
            )
            if "itpnb" in name:
                num_channels = 512
            elif "itpnl" in name:
                num_channels = 768
            elif "itpnt" in name:
                num_channels = 384
            elif "itpns" in name:
                num_channels = 384
            else:
                num_channels = 512
        elif "oriitpn" in name.lower():
            encoder = getattr(oriitpn_module, name)(
                pretrained=is_main_process(),
                search_size=search_size,
                template_size=template_size,
                drop_path_rate=0.1,
                init_values=0.1,
                use_mean_pooling=True,
                ape=True,
                rpe=True,
                pos_type=cfg.MODEL.ENCODER.POS_TYPE,
                token_type_indicate=cfg.MODEL.ENCODER.TOKEN_TYPE_INDICATE,
                task_indicate=cfg.MODEL.ENCODER.TASK_INDICATE,
                task_num=cfg.MODEL.TASK_NUM,
                pretrain_type=cfg.MODEL.ENCODER.PRETRAIN_TYPE
            )
            if "itpnb" in name:
                num_channels = 512
            else:
                num_channels = 512
        else:
            raise ValueError()
        super().__init__(encoder, train_encoder, open_layers, num_channels)



def build_encoder(cfg):
    train_encoder = (cfg.TRAIN.ENCODER_MULTIPLIER > 0) and (cfg.TRAIN.FREEZE_ENCODER == False)
    encoder = Encoder(cfg.MODEL.ENCODER.TYPE, train_encoder,
                      cfg.MODEL.ENCODER.PRETRAIN_TYPE,
                      cfg.DATA.SEARCH.SIZE, cfg.DATA.SEARCH.NUMBER,
                      cfg.DATA.TEMPLATE.SIZE, cfg.DATA.TEMPLATE.NUMBER,
                      cfg.TRAIN.ENCODER_OPEN, cfg)
    return encoder
