import torch
from opencood.models.fuse_modules.fusion_in_one import regroup

record_len = torch.tensor((2,3))
heter_feature_2d_gt = torch.ones((5,1,1,1))
heter_feature_2d = torch.zeros((5,1,1,1)) 
heter_feature_2d_gt_split = regroup(heter_feature_2d_gt, record_len)
shape_num = 0
print("record_len: ", record_len)
print("heter_feature_2d_gt_shape: ", heter_feature_2d_gt.shape)
print("heter_feature_2d_gt_split: ", len(heter_feature_2d_gt_split))
for index in range(len(heter_feature_2d_gt_split)):
    #print("heter_feature_2d_gt_split_shape: ", heter_feature_2d_gt_split[index].shape)
    #print(heter_feature_2d_gt_split[index].shape[0])
    #print(shape_num)
    heter_feature_2d[shape_num] = heter_feature_2d_gt_split[index][0]
    shape_num = shape_num + heter_feature_2d_gt_split[index].shape[0]

print(heter_feature_2d)