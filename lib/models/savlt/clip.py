import torch
import clip
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from timm.models.layers import trunc_normal_
from collections import OrderedDict
import numpy as np

class TextEncoder(nn.Module):
    def __init__(self, type, out_channel,feat_type='last',max_len=40,use_coop=True,n_ctx=8,cocoop_type='search'):
        super().__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, self.preprocess = clip.load(type, device=device)
        clip_model.float()
        self.clip_embed_dim = clip_model.text_projection.size(1)

        self.clip_transformer = clip_model.transformer
        self.clip_positional_embedding = clip_model.positional_embedding
        self.clip_ln_final = clip_model.ln_final
        self.clip_text_projection = clip_model.text_projection
        self.clip_context_length=clip_model.context_length
        self.clip_token_embedding=clip_model.token_embedding


        self.text_proj = nn.Linear(self.clip_embed_dim, out_channel)
        self.feat_type=feat_type
        self.max_len=max_len

        self.use_coop=use_coop
        if use_coop:
            self.cocoop_type=cocoop_type
            self.n_ctx = n_ctx
            ctx_vectors = torch.empty(n_ctx, self.clip_embed_dim)
            nn.init.normal_(ctx_vectors, std=0.02)
            self.ctx = nn.Parameter(ctx_vectors)
            print(f"Number of context words (tokens): {n_ctx}")

            vis_dim=out_channel
            if 'search' in self.cocoop_type:
                if out_channel==512:
                    self.ctx_meta_net = nn.Sequential(OrderedDict([
                        ("conv",nn.Conv2d(vis_dim, vis_dim//4, kernel_size=5,
                              padding=0, stride=2,bias=True)),
                        # ("bn",nn.BatchNorm2d(vis_dim//4)),
                        ("relu", nn.ReLU(inplace=True)),
                        ("avgpool",nn.AvgPool2d(5)),
                        ("linear", nn.Linear(vis_dim // 4, self.clip_embed_dim))
                    ]))
                elif out_channel==768:
                    self.ctx_meta_net = nn.Sequential(OrderedDict([
                        ("conv1", nn.Conv2d(vis_dim, vis_dim // 4, kernel_size=5,
                                           padding=0, stride=2, bias=True)),
                        # ("bn",nn.BatchNorm2d(vis_dim//4)),
                        ("relu1", nn.ReLU(inplace=True)),
                        ("conv2", nn.Conv2d(vis_dim//4, vis_dim // 4, kernel_size=5,
                                           padding=0, stride=2, bias=True)),
                        # ("bn",nn.BatchNorm2d(vis_dim//4)),
                        ("relu2", nn.ReLU(inplace=True)),
                        ("avgpool", nn.AvgPool2d(3)),
                        ("linear", nn.Linear(vis_dim // 4, self.clip_embed_dim))
                    ]))
            elif 'template' in self.cocoop_type:
                self.ctx_meta_net = nn.Sequential(OrderedDict([
                    ("conv", nn.Conv2d(vis_dim, vis_dim // 4, kernel_size=3,
                                       padding=0, stride=1, bias=True)),
                    # ("bn",nn.BatchNorm2d(vis_dim//4)),
                    ("relu", nn.ReLU(inplace=True)),
                    ("avgpool", nn.AvgPool2d(5)),
                    ("linear", nn.Linear(vis_dim // 4, self.clip_embed_dim))
                ]))
            for n,p in self.ctx_meta_net.named_parameters():
                if p.dim()>1:
                    # print('normal',n,p.shape)
                    trunc_normal_(p, std=.02)
                else:
                    # print('const', n,p.shape)
                    nn.init.constant_(p, 0)

        trunc_normal_(self.text_proj.weight, std=.02)
        nn.init.constant_(self.text_proj.bias, 0)
    @property
    def dtype(self):
        return self.text_proj.weight.dtype
    def forward(self, z,x,text_data,text_len):

        embedding = self.clip_token_embedding(text_data)
        bs = text_data.shape[0]
        add_idx = torch.zeros(bs).to(text_len.device)
        if self.use_coop:
            #todo preprocess im_feat
            if 'search' in self.cocoop_type:
                im_feat=x
            elif 'template' in self.cocoop_type:
                im_feat=z

            im_feat = im_feat.permute((0, 2, 1)).contiguous()
            fx_sz=int(np.sqrt(im_feat.shape[2]))
            im_feat = im_feat.view(bs, im_feat.shape[1], fx_sz, fx_sz)

            #debug
            for idx in range(len(self.ctx_meta_net)-1):
                im_feat=self.ctx_meta_net[idx](im_feat)
            im_feat=im_feat.squeeze()
            bias=self.ctx_meta_net[-1](im_feat)# (batch, ctx_dim)
            if bs==1:
                bias=bias.unsqueeze(0)
            bias = bias.unsqueeze(1).expand(-1, self.n_ctx, -1)  # (batch, n_ctx, ctx_dim)

            ctx = self.ctx
            if ctx.dim() == 2:
                ctx = ctx.unsqueeze(0).expand(bs, -1, -1)
            ctx_shifted = ctx + bias  # (batch, n_ctx, ctx_dim)
            prompts=[]
            for idx in range(bs):
                if text_len[idx]==0:
                    prompts.append(embedding[idx].unsqueeze(0))
                else:
                    prompts.append(torch.cat(
                        [
                            embedding[idx, :text_len[idx], :],  # (bs, len(SOS+LANG), dim)
                            ctx_shifted[idx],  # (bs, n_ctx, dim)
                            embedding[idx, text_len[idx]:, :],  # (n_cls, 1, dim) EOS
                        ],
                        dim=0,
                    )[:self.clip_context_length,:].unsqueeze(0))
                    add_idx[idx]+=self.n_ctx
            x=torch.cat(prompts,dim=0)
        else:
            x=embedding

        ###check
        # a = text_data.argmax(dim=-1) + add_idx.type(text_data.dtype)
        # print('===========')
        # for idx in range(bs):
        #     print(x[idx, a[idx].item()].mean())

        x = x + self.clip_positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip_ln_final(x)



        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        eot_indices = text_data.argmax(dim=-1) + add_idx.type(text_data.dtype)
        if self.feat_type=='last':
            text_src = x[torch.arange(x.shape[0]), eot_indices] @ self.clip_text_projection
            mask=None
            # mask = torch.zeros(bs, 1, dtype=int).to(x.device)
        elif self.feat_type=='prompt':
            seq=[]
            for idx in range(bs):
                if text_len[idx]==0:
                    seq.append(x[idx,text_len[idx]:self.n_ctx].unsqueeze(0))
                else:
                    seq.append(x[idx,text_len[idx]:text_len[idx]+self.n_ctx].unsqueeze(0))
            seq=torch.cat(seq,dim=0)
            text_src=seq @ self.clip_text_projection
            mask=None

        elif self.feat_type=='all':
            text_src = x @ self.clip_text_projection
            temp = torch.zeros(self.max_len, text_src.shape[-1]).to(x.device)

            seq = []
            for i in range(bs):
                s_idx = 1
                e_idx = eot_indices[i]
                # new，好像有句子超过40了
                if e_idx - s_idx > self.max_len:
                    print('====',e_idx)
                if e_idx==0:
                    seq.append(x[i, 0:1])
                    eot_indices[i]=2
                else:
                    seq.append(x[i, s_idx:e_idx][:self.max_len])
            seq.append(temp)
            padded_seq = pad_sequence(seq).permute(1, 0, 2)
            text_src = padded_seq[:-1]
            no_mask_sum = eot_indices - 1
            mask = torch.ones(bs, self.max_len, dtype=int).to(x.device)
            for i in range(bs):
                mask[i, :no_mask_sum[i]] = mask[i, :no_mask_sum[i]] - 1

        text_src = self.text_proj(text_src)
        if len(text_src.shape)==2:
            text_src = text_src.unsqueeze(1)
        return text_src,mask

def build_textencoder(cfg, encoder):
    num_channels_enc = encoder.num_channels
    model = TextEncoder(cfg.MODEL.TEXT_ENCODER.TYPE, num_channels_enc,feat_type=cfg.MODEL.TEXT_ENCODER.FEAT_TYPE,
                        max_len=cfg.MODEL.TEXT_ENCODER.MAX_LEN,
                        use_coop=cfg.MODEL.TEXT_ENCODER.USECOOP,n_ctx=cfg.MODEL.TEXT_ENCODER.NUM_CTX,
                        cocoop_type=cfg.MODEL.TEXT_ENCODER.COCOOP_TYPE)
    return model
