import os
from torchvision.ops.boxes import box_area
import numpy as np
import torch

def calculate_statistics(numbers):
    if not numbers:
        print("列表为空")
        return

    # 计算均值
    mean = sum(numbers) / len(numbers)

    # 计算最小值和最大值
    min_value = min(numbers)
    max_value = max(numbers)

    # 打印结果
    print("总数量:", len(numbers))
    print("均值:{:.2f}".format(mean))
    print("最小值:{:.2f}".format(min_value))
    print("最大值:{:.2f}".format(max_value))

def box_iou(boxes1, boxes2):
    boxes1[2] = boxes1[0]+boxes1[2]
    boxes1[3] = boxes1[1]+boxes1[3]
    boxes2[2] = boxes2[0]+boxes2[2]
    boxes2[3] = boxes2[1]+boxes2[3]
    boxes1 = boxes1.unsqueeze(0)
    boxes2 = boxes2.unsqueeze(0)
    area1 = box_area(boxes1) # (N,)
    area2 = box_area(boxes2) # (N,)

    lt = torch.max(boxes1[:, :2], boxes2[:, :2])  # (N,2)
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # (N,2)

    wh = (rb - lt).clamp(min=0)  # (N,2)
    inter = wh[:, 0] * wh[:, 1]  # (N,)

    union = area1 + area2 - inter

    iou = inter / union
    iou = iou.mean()
    return iou


hit_results_path = '/home/cx/cx1/201proj/save/uav_hit201'
hit_annos_path = '/home/cx/cx1/GOKU_repo/Goku/data/UAV123/anno/UAV123'

anno_names_list = os.listdir(hit_annos_path)
anno_names_list = [anno_name for anno_name in anno_names_list if anno_name[-3:]=='txt']

# anno_names_list = ['uav_'+result_name for result_name in result_names_list]
success_iou_threshold = 0.5

success_ratio_dict = {}
for anno_name in anno_names_list:
    success_ratio_dict[anno_name[0:-4]] = {}
    anno_path = os.path.join(hit_annos_path, anno_name)
    result_path = os.path.join(hit_results_path, 'uav_'+anno_name)
    with open(anno_path) as f_a:
        with open(result_path) as f_r:
            result_bboxes = f_r.readlines()
            anno_bboxes = f_a.readlines()
            assert len(result_bboxes) == len(anno_bboxes)
            total_num = 0
            success_num = 0
            for i in range(len(result_bboxes)):
                box_r = result_bboxes[i].split()
                box_r = torch.tensor([float(item) for item in box_r])
                box_a = anno_bboxes[i].strip().split(',')
                box_a = torch.tensor([float(item) for item in box_a])
                if i == 0:
                    area = box_a[2] * box_a[3]
                iou = box_iou(box_a,box_r)
                total_num += 1
                if iou >= success_iou_threshold:
                    success_num += 1
            ratio = success_num / total_num
            success_ratio_dict[anno_name[0:-4]]['ratio'] = ratio
            success_ratio_dict[anno_name[0:-4]]['area'] = area

area_thresholds = [9, 25, 100, 225, 400, 625, 900, 1600, 2500, 3600, 4900, 6400, 8100, 10000]
for area_threshold in area_thresholds:
    print("\n")
    print("面积阈值:", area_threshold)
    selected_list = []
    for name in success_ratio_dict.keys():
        if success_ratio_dict[name]['area'] < area_threshold:
            selected_list.append(success_ratio_dict[name]['ratio'])
    calculate_statistics(selected_list)


a=1



