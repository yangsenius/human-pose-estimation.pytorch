# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn

import logging
logger = logging.getLogger(__name__)

class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        #self.criterion = nn.MSELoss(size_average =True,reduce=False, reduction='sum') # reduce=false
        #self.critterion = self.heatmaploss()
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        #print(output.size())
        logger.info('=> output.size {}'.format(output.size()))
        logger.info('=> target.size {}'.format(target.size()))
        
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)# tuple
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1) # tuple
        
        logger.info('=> heatmaps_pred.size {}'.format(heatmaps_pred[1].size()))
        logger.info('=> heatmaps_gt.size {}'.format(heatmaps_gt[1].size()))

        logger.info('=> target_weight.size {}'.format(target_weight[1].size()))

        
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                 #print(heatmap_pred.size())
                 #"""=> output.size torch.Size([8, 17, 97, 73])
			    #=> target.size torch.Size([8, 17, 97, 73])
			    #=> tuple: heatmaps_pred[1].size  torch.Size([8, 1, 7081])
			    #=> tuple: heatmaps_gt[2].size torch.Size([8, 1, 7081])
			     #=> target_weight.size torch.Size([8, 17, 1])"""
			
                #print(heatmap_gt.size())
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)
        
        return loss/ num_joints

    def criterion(self,heatmap_pred, heatmap_gt):
        assert heatmap_pred.size() == heatmap_gt.size()
        loss = ((heatmap_pred - heatmap_gt)**2)
        loss = loss.mean(dim=1)
        return loss
