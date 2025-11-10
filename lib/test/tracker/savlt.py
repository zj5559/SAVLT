from lib.test.tracker.basetracker import BaseTracker
import torch
from lib.test.tracker.utils import sample_target, transform_image_to_crop
import cv2
from lib.utils.box_ops import box_xywh_to_xyxy, box_xyxy_to_cxcywh,box_iou
from lib.test.utils.hann import hann2d
from lib.models.savlt import build_savlt
from lib.test.tracker.utils import Preprocessor
from lib.utils.box_ops import clip_box
import clip
import numpy as np
import torch.nn.functional as F


class SAVLT(BaseTracker):
    def __init__(self, params, dataset_name):
        super(SAVLT, self).__init__(params)
        network = build_savlt(params.cfg,training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.fx_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.ENCODER.STRIDE
        if self.cfg.TEST.WINDOW == True: # for window penalty
            self.output_window = hann2d(torch.tensor([self.fx_sz, self.fx_sz]).long(), centered=True).cuda()

        self.num_template = self.cfg.TEST.NUM_TEMPLATES

        self.debug = params.debug
        self.frame_id = 0

        # online update settings
        DATASET_NAME = dataset_name.upper()
        # if hasattr(self.cfg.TEST.UPDATE_INTERVALS, DATASET_NAME):
        #     self.update_intervals = self.cfg.TEST.UPDATE_INTERVALS[DATASET_NAME]
        # else:
        #     self.update_intervals = self.cfg.TEST.UPDATE_INTERVALS.DEFAULT
        # print("Update interval is: ", self.update_intervals)

        self.update_template=self.cfg.TEST.UPDATE_TEMPLATE
        self.update_interval=self.cfg.TEST.UPDATE_INTERVAL
        self.update_thres = self.cfg.TEST.UPDATE_THRESHOLD
        # if hasattr(self.cfg.TEST.UPDATE_THRESHOLD, DATASET_NAME):
        #     self.update_threshold = self.cfg.TEST.UPDATE_THRESHOLD[DATASET_NAME]
        # else:
        #     self.update_threshold = self.cfg.TEST.UPDATE_THRESHOLD.DEFAULT
        # print("Update threshold is: ", self.update_threshold)

        # mapping similar datasets
        if 'GOT10K' in DATASET_NAME:
            DATASET_NAME = 'GOT10K'
        if 'LASOT' in DATASET_NAME:
            DATASET_NAME = 'LASOT'
        if 'OTB' in DATASET_NAME:
            DATASET_NAME = 'TNL2K'

        #multi modal language
        if hasattr(self.cfg.TEST.MULTI_MODAL_LANGUAGE, DATASET_NAME):
            self.multi_modal_language = self.cfg.TEST.MULTI_MODAL_LANGUAGE[DATASET_NAME]
        else:
            self.multi_modal_language = self.cfg.TEST.MULTI_MODAL_LANGUAGE.DEFAULT
        print("MULTI_MODAL_LANGUAGE is: ", self.multi_modal_language)

        #using nlp information
        if hasattr(self.cfg.TEST.USE_NLP, DATASET_NAME):
            self.use_nlp = self.cfg.TEST.USE_NLP[DATASET_NAME]
        else:
            self.use_nlp = self.cfg.TEST.USE_NLP.DEFAULT
        print("USE_NLP is: ", self.use_nlp)



    def initialize(self, image, info: dict):
        #in case the grounding bbox is too small
        if info['init_bbox'][2]<3:
            info['init_bbox'][2]=3
        if info['init_bbox'][3] < 3:
            info['init_bbox'][3] = 3
        # get the initial templates
        z_patch_arr, resize_factor = sample_target(image, info['init_bbox'], self.params.template_factor,
                                       output_sz=self.params.template_size)
        z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr)

        self.template_list = [template] * self.num_template

        self.state = info['init_bbox']
        prev_box_crop = transform_image_to_crop(torch.tensor(info['init_bbox']),
                                                torch.tensor(info['init_bbox']),
                                                resize_factor,
                                                torch.Tensor([self.params.template_size, self.params.template_size]),
                                                normalize=True)
        self.template_anno_list = [prev_box_crop.to(template.device).unsqueeze(0)]
        self.frame_id = 0

        # language information
        if self.multi_modal_language:
            if self.use_nlp:
                init_nlp = info.get("init_nlp")
            else:
                init_nlp = None
            text_data, _,text_len = self.extract_token_from_nlp_clip(init_nlp)
            text_data = text_data.unsqueeze(0).to(template.device)
            text_len=torch.tensor(text_len).unsqueeze(0).to(template.device)
            self.text_data=text_data
            self.text_len=text_len
            # with torch.no_grad():
            #     self.text_src,self.text_mask = self.network.forward_textencoder(text_data=text_data,text_len=text_len)
            # self.text_src_local=self.text_src.clone()
        else:
            self.text_data =None
            self.text_len = None

        # self.show(self.template_list, 0, self.template_anno_list)

    # def show(self, template_list, i, template_anno_list):
    #     image = template_list[i][0]
    #     _, H, W = image.shape
    #     import cv2
    #     x1, y1, w, h = template_anno_list[i][0]
    #     x1, y1, w, h = int(x1*W), int(y1*H), int(w*W), int(h*H)
    #     image_show = image.permute(1,2,0).cpu().numpy()
    #     max = image_show.max()
    #     min = image_show.min()
    #     image_show = (image_show-min) * 255 / (max-min)
    #     image_show = np.ascontiguousarray(image_show.astype('uint8'))
    #     cv2.rectangle(image_show, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 0, 255), thickness=2)
    #     cv2.imshow(str(i), image_show)
    #     if cv2.waitKey() & 0xFF == ord('q'):
    #         pass

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor = sample_target(image, self.state, self.params.search_factor,
                                                   output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr)

        search_list = [search]

        # run the encoder
        with torch.no_grad():
            z, x = self.network.forward_encoder_early(template_list=self.template_list,
                                                 search_list=search_list, template_anno_list=self.template_anno_list,
                                                 mode='patch_embed')
            text_src, text_mask = self.network.forward_textencoder(z,x,self.text_data,self.text_len)
            enc_opt,_ = self.network.forward_encoder(z=z,x=x, text_src=text_src,text_mask=text_mask,mode='encoder')

        # run the decoder
        with torch.no_grad():
            out_dict = self.network.forward_decoder(feature=enc_opt)

        # add hann windows
        pred_score_map = out_dict['score_map']
        if self.cfg.TEST.WINDOW == True: # for window penalty
            response = self.output_window * pred_score_map
        else:
            response = pred_score_map
        if 'size_map' in out_dict.keys():
            pred_boxes, conf_score = self.network.decoder.cal_bbox(response, out_dict['size_map'],
                                                                   out_dict['offset_map'], return_score=True)
        else:
            pred_boxes, conf_score = self.network.decoder.cal_bbox(response,
                                                                   out_dict['offset_map'],
                                                                   return_score=True)
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        box1=torch.Tensor(self.state).unsqueeze(0)
        box1[:,2:]+=box1[:,:2]
        box2 = torch.Tensor(info['gt_bbox']).unsqueeze(0)
        box2[:, 2:] += box2[:, :2]
        if info['gt_bbox'][2]<=1 or info['gt_bbox'][3]<=1:
            iou=torch.Tensor([-1])
        else:
            iou,_=box_iou(box1,box2)
            iou=iou[0]

        # update the template
        visible=conf_score>0.5
        if self.update_template and self.frame_id % self.update_interval==0:
            if conf_score>self.update_thres:
                # print('update:',self.frame_id,iou)
                z_patch_arr, resize_factor = sample_target(image, self.state, self.params.template_factor,
                                                           output_sz=self.params.template_size)
                template = self.preprocessor.process(z_patch_arr)
                self.template_list[-1]=template

                prev_box_crop = transform_image_to_crop(torch.tensor(self.state),
                                                        torch.tensor(self.state),
                                                        resize_factor,
                                                        torch.Tensor(
                                                            [self.params.template_size, self.params.template_size]),
                                                        normalize=True)
                self.template_anno_list[-1]=(prev_box_crop.to(template.device).unsqueeze(0))

        # for debug
        if image.shape[-1] == 6:
            image_show = image[:,:,:3]
        else:
            image_show = image
        if self.debug == 1:
            x1, y1, w, h = self.state
            image_BGR = cv2.cvtColor(image_show, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
            cv2.imshow('vis', image_BGR)
            cv2.waitKey(1)

        return {"target_bbox": self.state,
                "best_score": conf_score.squeeze().item(),
                "pred_visible":visible.item(),
                "iou":iou.item()}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def extract_token_from_nlp_clip(self, nlp):
        if nlp is None:
            nlp_ids = torch.zeros(77, dtype=torch.long)
            nlp_masks = torch.zeros(77, dtype=torch.long)
            num_nlp=0
        else:
            # print('nlp',nlp)
            nlp = nlp.replace("_", " ")
            nlp_ids = clip.tokenize(nlp, truncate=True).squeeze(0)
            nlp_masks = (nlp_ids == 0).long()
            num_nlp=nlp_ids.argmax().item()

        return nlp_ids, nlp_masks,num_nlp


def get_tracker_class():
    return SAVLT
