import argparse
import torch
from thop import profile
from thop.utils import clever_format
import time
import importlib
from torch import nn
import numpy as np


def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Parse args for training')
    # for train
    parser.add_argument('--script', type=str, default='goten',
                        help='training script name')
    parser.add_argument('--config', type=str, default='goten_l384', help='yaml configure file name')
    args = parser.parse_args()

    return args


def get_complexity_MHA(m:nn.MultiheadAttention, x, y):
    """(L, B, D): sequence length, batch size, dimension"""
    d_mid = m.embed_dim
    query, key, value = x[0], x[1], x[2]
    Lq, batch, d_inp = query.size()
    Lk = key.size(0)
    """compute flops"""
    total_ops = 0
    # projection of Q, K, V
    total_ops += d_inp * d_mid * Lq * batch  # query
    total_ops += d_inp * d_mid * Lk * batch * 2  # key and value
    # compute attention
    total_ops += Lq * Lk * d_mid * 2
    m.total_ops += torch.DoubleTensor([int(total_ops)])


def evaluate(model, template_list, search_list, template_anno_list, z,x,language, text_src, text_mask,enc_opt,bs):
    """Compute FLOPs, Params, and Speed"""
    # custom_ops = {nn.MultiheadAttention: get_complexity_MHA}
    custom_ops={}
    print("<==== 1.image encoder ====>")
    macs1, params1 = profile(model, inputs=(None,None,template_list, search_list, template_anno_list, None,None,None,None,None,"patch_embed"),
                           custom_ops={}, verbose=False)
    macs1, params1 = clever_format([macs1, params1], "%.3f")
    print('==>Macs is ', macs1)
    print('==>Params is ', params1)

    print("<==== 2.text encoder ====>")
    macs2, params2 = profile(model, inputs=(
    language, text_len, None, None, None, z, x, None, None, None, "text"),
                             custom_ops={}, verbose=False)
    macs2, params2 = clever_format([macs2, params2], "%.3f")
    print('==>Macs is ', macs2)
    print('==>Params is ', params2)

    print("<==== 3.relation model ====>")
    macs3, params3 = profile(model, inputs=(
        None, None, None, None, None, z, x, text_src, text_mask, None, "encoder"),
                             custom_ops={}, verbose=False)
    macs3, params3 = clever_format([macs3, params3], "%.3f")
    print('==>Macs is ', macs3)
    print('==>Params is ', params3)

    print("<==== 4.tracking head ====>")
    macs4, params4 = profile(model, inputs=(
        None, None, None, None, None, None, None, None, None, enc_opt, "decoder"),
                             custom_ops={}, verbose=False)
    macs4, params4 = clever_format([macs4, params4], "%.3f")
    print('==>Macs is ', macs4)
    print('==>Params is ', params4)


    '''Speed Test'''
    T_w = 10
    T_t = 100
    print("testing speed ...")
    with torch.no_grad():
        # overall
        for i in range(T_w):
            _, _ = model(template_list=template_list,
                         search_list=search_list,
                         template_anno_list=template_anno_list,
                         mode='patch_embed')  # encoder: patch_embedding

            _, _ = model(z=z, x=x, text_data=language, text_len=text_len, mode='text')
            _, _ = model(z=z,
                                       x=x,
                                       text_src=text_src,
                                       text_mask=text_mask,
                                       mode='encoder')  # forward the encoder

            _, _ = model(feature=enc_opt, mode="decoder")
        start = time.time()
        for i in range(T_t):
            _, _ = model(template_list=template_list,
                         search_list=search_list,
                         template_anno_list=template_anno_list,
                         mode='patch_embed')  # encoder: patch_embedding

            _, _ = model(z=z, x=x, text_data=language, text_len=text_len, mode='text')
            _, _ = model(z=z,
                         x=x,
                         text_src=text_src,
                         text_mask=text_mask,
                         mode='encoder')  # forward the encoder

            _, _ = model(feature=enc_opt, mode="decoder")
        end = time.time()
        avg_lat = (end - start) / (T_t * bs)
        fps=1/avg_lat
        print("The average overall latency is %.2f ms" % (avg_lat * 1000))
        print('fps is',fps)



def get_data(bs, sz):
    img_patch = torch.randn(bs, 3, sz, sz)
    return img_patch

if __name__ == "__main__":
    device = "cuda:1"
    # device = "cpu"
    torch.cuda.set_device(device)
    # Compute the Flops and Params of our model
    args = parse_args()
    args.script='savlt'
    args.config='cocoop_l384_8_search_v2'
    '''update cfg'''
    yaml_fname = '/home/zj/tracking/2024/language/Goku-main/experiments/%s/%s.yaml' % (args.script, args.config)
    config_module = importlib.import_module('lib.config.%s.config' % args.script)
    cfg = config_module.cfg
    config_module.update_config_from_file(yaml_fname)
    '''set some values'''
    bs = 1
    z_sz = cfg.TEST.TEMPLATE_SIZE
    x_sz = cfg.TEST.SEARCH_SIZE
    '''import network module'''
    model_module = importlib.import_module('lib.models.'+args.script)
    model_constructor = model_module.build_baseline_clip_cocoop
    model = model_constructor(cfg)
    # get the template and search
    template = get_data(bs, z_sz)
    search = get_data(bs, x_sz)
    language=torch.ones(bs, 77).long()
    text_len=torch.ones(bs).long()+10
    # transfer to device
    model = model.to(device)
    template = template.to(device)
    search = search.to(device)
    language=language.to(device)
    text_len=text_len.to(device)
    model.eval()
    # evaluate the model properties
    template_list = [template]
    search_list = [search]
    # template_anno_list = [torch.tensor([[0.2117, 0.2794, 0.5766, 0.4411]], device='cuda:1', dtype=torch.float64)]
    template_anno_list = [torch.tensor([[0.2117, 0.2794, 0.5766, 0.4411]], device=device, dtype=torch.float64)]

    z, x = model(template_list=template_list,
                    search_list=search_list,
                    template_anno_list=template_anno_list,
                    mode='patch_embed')  # encoder: patch_embedding

    text_src, text_mask = model(z=z, x=x, text_data=language, text_len=text_len, mode='text')
    enc_opt, enc_early = model(z=z,
                                  x=x,
                                  text_src=text_src,
                                  text_mask=text_mask,
                                  mode='encoder')  # forward the encoder

    outputs, logit_scale = model(feature=enc_opt, mode="decoder")

    evaluate(model, template_list, search_list, template_anno_list, z,x,language, text_src, text_mask,enc_opt,bs)

