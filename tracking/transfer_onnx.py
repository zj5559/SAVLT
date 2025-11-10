import argparse
import torch
import _init_paths
from lib.utils.box_ops import box_xyxy_to_cxcywh
import torch.nn as nn
import torch.nn.functional as F
# for onnx conversion and inference
import torch.onnx
import numpy as np
import onnx
import onnxruntime
import time
import os
from lib.test.evaluation.environment import env_settings
import importlib
import math


def parse_args():
    parser = argparse.ArgumentParser(description='Parse args for training')
    parser.add_argument('--script', type=str, default='goten', help='script name')
    parser.add_argument('--config', type=str, default='goten_s224', help='yaml configure file name')
    args = parser.parse_args()
    return args


def get_data(bs=1, sz=256):
    # img_patch = torch.randn(bs, 3, sz, sz, requires_grad=True)
    img_patch = torch.randn(bs, 3, sz, sz)
    return img_patch
def get_data_anno(bs=1):
    # anno = torch.randn(bs, 4)
    anno = torch.tensor([[0.2117, 0.2794, 0.5766, 0.4411]], device='cpu')
    return anno

class GOTEN(nn.Module):
    def __init__(self, encoder, decoder,
                 num_frames=1, num_template=1, decoder_type="CENTER"):
        super().__init__()
        self.encoder = encoder
        self.decoder_type = decoder_type

        self.num_patch_x = self.encoder.body.num_patches_search
        self.num_patch_z = self.encoder.body.num_patches_template
        self.fx_sz = int(math.sqrt(self.num_patch_x))
        self.fz_sz = int(math.sqrt(self.num_patch_z))

        self.decoder = decoder

        self.num_frames = num_frames
        self.num_template = num_template

    def forward(self, template, search, template_anno):
        # run the backbone
        template_list = [template]
        search_list = [search]
        template_anno_list = [template_anno]
        feature = self.encoder(template_list, search_list, template_anno_list)
        feature = feature[0]
        feature = feature[:, 0:self.num_patch_x * self.num_frames]
        bs, HW, C = feature.size()
        feature = feature.permute((0, 2, 1)).contiguous()
        feature = feature.view(bs, C, self.fx_sz, self.fx_sz)
        score_map_ctr, bbox, size_map, offset_map = self.decoder(feature, None)
        outputs_coord = bbox
        outputs_coord_new = outputs_coord.view(bs, 1, 4)
        return outputs_coord_new


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


if __name__ == "__main__":
    load_checkpoint = False
    # update cfg
    args = parse_args()
    yaml_fname = 'experiments/%s/%s.yaml' % (args.script, args.config)
    config_module = importlib.import_module('lib.config.%s.config' % args.script)
    cfg = config_module.cfg
    config_module.update_config_from_file(yaml_fname)
    save_name = "checkpoints/train/%s/%s/GOTEN_ep%04d.onnx" % (args.script, args.config, cfg.TEST.EPOCH)
    # build the model
    model_module = importlib.import_module('lib.models.goten')
    model_constructor = model_module.build_goten
    model = model_constructor(cfg)
    # load checkpoint
    if load_checkpoint:
        save_dir = env_settings().save_dir
        checkpoint_name = os.path.join(save_dir,
                                       "checkpoints/train/%s/%s/GOTEN_ep%04d.pth.tar"
                                       % (args.script, args.config, cfg.TEST.EPOCH))
        model.load_state_dict(torch.load(checkpoint_name, map_location='cpu')['net'], strict=True)
    # merge conv+bn for levit
    # transfer to test mode
    model.eval()
    """ rebuild the inference-time model """
    encoder = model.encoder
    decoder = model.decoder
    torch_model = GOTEN(encoder, decoder, 1, cfg.TEST.NUM_TEMPLATES, cfg.MODEL.DECODER.TYPE)
    torch_model.cuda()
    torch_model.eval()
    # print(torch_model)
    # torch.save(torch_model.state_dict(), "complete.pth")
    # get the network input
    bs = 1
    sz_x = cfg.TEST.SEARCH_SIZE
    sz_z = cfg.TEST.TEMPLATE_SIZE
    search = get_data(bs=bs, sz=sz_x)
    search_cuda = search.cuda()
    template = get_data(bs=bs, sz=sz_z)
    template_cuda = template.cuda()
    template_anno = get_data_anno(bs=bs)
    template_anno_cuda = template_anno.cuda()
    torch.onnx.export(torch_model,  # model being run
                      (template_cuda, search_cuda, template_anno_cuda),  # model input (a tuple for multiple inputs)
                      save_name,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=11,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['template', 'search', 'template_anno'],  # model's input names
                      output_names=['outputs_coord_new'],  # the model's output names
                      # dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                      #               'output': {0: 'batch_size'}}
                      )
    """########## inference with the pytorch model ##########"""
    # forward the template
    N = 1000
    # torch_model = torch_model.cuda()
    # torch_model.eval() # to move attention.ab to cuda for levit
    # torch_model.box_head.coord_x = torch_model.box_head.coord_x.cuda()
    # torch_model.box_head.coord_y = torch_model.box_head.coord_y.cuda()

    # """########## inference with the onnx model ##########"""
    onnx_model = onnx.load(save_name)
    onnx.checker.check_model(onnx_model)
    print("creating session...")
    ort_session = onnxruntime.InferenceSession(save_name, providers=["CUDAExecutionProvider"])
    # # ort_session.set_providers(["TensorrtExecutionProvider"],
    # #                   [{'device_id': '1', 'trt_max_workspace_size': '2147483648', 'trt_fp16_enable': 'True'}])
    print("execuation providers:")
    print(ort_session.get_providers())
    # # compute ONNX Runtime output prediction
    # generate data
    # search = get_data(bs=bs, sz=sz_x)
    # template = get_data(bs=bs, sz=sz_z)
    # pytorch inference
    # search_cuda, template_cuda = search.cuda(), template.cuda()
    ort_inputs = {'template': to_numpy(template).astype(np.float32),
                  'search': to_numpy(search).astype(np.float32),
                  'template_anno': to_numpy(template_anno).astype(np.float32)
                  }
    """warmup (the first one running latency is quite large for the onnx model)"""
    for i in range(50):
        # pytorch inference
        torch_outs = torch_model(template_cuda, search_cuda, template_anno_cuda)
        # onnx inference
        ort_outs = ort_session.run(None, ort_inputs)
    """begin the timing"""
    t_pyt = 0  # pytorch time
    t_ort = 0  # onnxruntime time
    s_pyt = time.time()
    for i in range(N):
        torch_outs = torch_model(template_cuda, search_cuda, template_anno_cuda)
    e_pyt = time.time()
    lat_pyt = e_pyt - s_pyt
    t_pyt += lat_pyt
    s_ort = time.time()
    for i in range(N):
        # ort_inputs = model(xz=model([search_cuda, template_cuda], mode="backbone"), mode="transformer")[0]
        ort_outs = ort_session.run(None, ort_inputs)
    e_ort = time.time()
    lat_ort = e_ort - s_ort
    t_ort += lat_ort
    print("pytorch model average latency", t_pyt/N*1000)
    print("onnx model average latency:", t_ort/N*1000)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_outs), ort_outs[0], rtol=1e-03, atol=1e-05)
    print("The deviation between the first output: {}".format(np.max(np.abs(to_numpy(torch_outs[0]) - ort_outs[0]))))
    #
    # print("Exported model has been tested with ONNXRuntime, and the result looks good!")



