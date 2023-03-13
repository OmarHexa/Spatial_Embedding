"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import numpy as np

import torch
import torch.nn as nn
from criterions.lovasz_losses import lovasz_hinge


class SpatialEmbLoss(nn.Module):

    def __init__(self, to_center=True, n_sigma=1, class_weight=[1],num_class=1):
        super().__init__()

        print('Created spatial emb loss function with: to_center: {}, n_sigma: {}'.format(
            to_center, n_sigma))

        self.to_center = to_center
        self.n_sigma = n_sigma
        self.class_weight = class_weight
        self.num_class =num_class
        # coordinate map
        xm = torch.linspace(0, 1, 1024).view(1, 1, -1).expand(1, 1024, 1024)
        ym = torch.linspace(0, 1, 1024).view(1, -1, 1).expand(1, 1024, 1024)
        xym = torch.cat((xm, ym), 0)

        self.register_buffer("xym", xym)

    def forward(self, prediction, instances, labels, w_inst=1, w_var=10, w_seed=1, iou=False, iou_meter=None):

        batch_size, height, width = prediction.size(
            0), prediction.size(2), prediction.size(3)

        xym_s = self.xym[:, 0:height, 0:width].contiguous()  # 2 x h x w

        loss = 0

        for b in range(0, batch_size):

            spatial_emb = torch.tanh(prediction[b, 0:2]) + xym_s  # 2 x h x w
            sigma = prediction[b, 2:2+self.n_sigma]  # n_sigma x h x w
            seed_map = torch.sigmoid(prediction[b, 2+self.n_sigma:2+self.n_sigma + self.num_class])  # num_class x h x w
            # loss accumulators
            var_loss = 0
            instance_loss = 0
            seed_loss = 0
            obj_count = 0

            instance = instances[b].unsqueeze(0)  # 1 x h x w
            label = labels[b].unsqueeze(0)  # 1 x h x w
            
            # regress bg to zero
            
            
            # label_ids = label.unique() # mlclass
            # label_ids = label_ids[label_ids!=0] #mlclass
            #For every class find the instances and calculate loss
            #TODO check whether it should be equat to total class instead of class present in the image
            for cl in range(self.num_class):
                label_mask = label.eq(cl+1) # 1 x h x w
                class_seed = seed_map[cl].unsqueeze(0) # 1 x h x w
                bg_mask = label == 0
                #TODO depeding on the previous TODO this can change
                if bg_mask.sum() > 0:
                    seed_loss += torch.sum(torch.pow(class_seed[bg_mask] - 0, 2))
                #only instances of cl class 
                instance_per_cls = instance*label_mask

                instance_ids = instance_per_cls.unique()
                instance_ids = instance_ids[instance_ids != 0]

                for id in instance_ids:

                    in_mask = instance_per_cls.eq(id)   # 1 x h x w

                    # calculate center of attraction
                    if self.to_center:
                        xy_in = xym_s[in_mask.expand_as(xym_s)]
                        xy_in =xy_in.view(2,-1)
                        center = xy_in.mean(1).view(2, 1, 1)  # 2 x 1 x 1
                    else:
                        center = spatial_emb[in_mask.expand_as(spatial_emb)].view(
                                            2, -1).mean(1).view(2, 1, 1)  # 2 x 1 x 1

                    # calculate sigma
                    sigma_in = sigma[in_mask.expand_as(
                                    sigma)].view(self.n_sigma, -1)
                    # sigma_k for the instance
                    s = sigma_in.mean(1).view(
                            self.n_sigma, 1, 1)   # n_sigma x 1 x 1

                    # calculate var loss before exp
                    var_loss = var_loss + \
                            torch.mean(
                            torch.pow(sigma_in - s[..., 0].detach(), 2))

                    s = torch.exp(s*10)

                    # calculate gaussian
                    dist = torch.exp(-1*torch.sum(
                            torch.pow(spatial_emb - center, 2)*s, 0, keepdim=True))

                    # apply lovasz-hinge loss
                    instance_loss = instance_loss + \
                            lovasz_hinge(dist*2-1, in_mask)

                    # seed loss
                    seed_loss += self.class_weight[cl-1] * torch.sum(
                                torch.pow(class_seed[in_mask] - dist[in_mask].detach(), 2))

                    # calculate instance iou
                    if iou:
                        iou_meter.update(calculate_iou(dist > 0.5, in_mask))

                    obj_count += 1

            if obj_count > 0:
                instance_loss /= obj_count
                var_loss /= obj_count

            seed_loss = seed_loss / (height * width)

            loss += w_inst * instance_loss + w_var * var_loss + w_seed * seed_loss

        loss = loss / (b+1)
        return loss


def calculate_iou(pred, label):
    intersection = ((label == 1) & (pred == 1)).sum()
    union = ((label == 1) | (pred == 1)).sum()
    if not union:
        return 0
    else:
        iou = intersection.item() / union.item()
        return iou




def DiceBceLoss(inputs, targets, smooth=1):      
        
        #flatten label and prediction tensors
    inputs = inputs.view(-1)
    targets = targets.view(-1)
        
        
    intersection = (inputs * targets).sum()                            
    dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
    BCE = nn.functional.binary_cross_entropy(inputs, targets, reduction='mean')
    Dice_BCE = BCE + dice_loss
        
    return Dice_BCE

class DiceBceLossMulti(nn.Module):
    def __init__(self, num_class=5, class_weight=None):
        super().__init__()
        self.class_id = list(range(1,num_class+1))
        self.class_weight = class_weight
    def forward(self,prediction, labels, iou=False, iou_meter=None):
        batch_size = prediction.size(0)
        loss =0
        for b in range(batch_size):
            label = labels[b]
            for id, cl in enumerate(self.class_id):
                pred = prediction[b,id]
                gt = label.eq(cl).type(torch.float)  
                loss+= DiceBceLoss(pred,gt)
                # calculate instance iou
                if iou:
                    iou_meter.update(calculate_iou(pred > 0.5, gt))
        return loss


if __name__ == "__main__":
    input = torch.randn(1, 5, 416, 416)
    labels =torch.ones((1,1,416,416), dtype=torch.bool)
    loss = DiceBceLossMulti(5)
    l = loss(torch.sigmoid(input),labels)
    