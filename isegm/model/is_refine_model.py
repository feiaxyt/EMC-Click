from tkinter import E
import torch.nn as nn
import torch
import torch.nn.functional as F
from isegm.utils.serialization import serialize
from .is_model import ISModel
from .modeling.hrnet_ocr import HighResolutionNet
from isegm.model.modifiers import LRMult
import torchvision.ops.roi_align as roi_align
from isegm.model.ops import DistMaps
import random
import numpy as np
class HRNetRefineModel(ISModel):
    @serialize
    def __init__(self, width=48, ocr_width=256, small=True, backbone_lr_mult=0.1,
                 norm_layer=nn.BatchNorm2d, refiner_conf=dict(), **kwargs):
        super().__init__(norm_layer=norm_layer, **kwargs)
        self.feature_extractor = HighResolutionNet(width=width, ocr_width=ocr_width, small=small,
                                                   num_classes=1, norm_layer=norm_layer)
        self.feature_extractor.apply(LRMult(backbone_lr_mult))
        if ocr_width > 0:
            self.feature_extractor.ocr_distri_head.apply(LRMult(1.0))
            self.feature_extractor.ocr_gather_head.apply(LRMult(1.0))
            self.feature_extractor.conv3x3_ocr.apply(LRMult(1.0))
        
        self.width=width
        base_radius = 5
        
        self.refiner = AttRefineAfterLayer(feature_dims=ocr_width * 2, spatial_scale=0.25, **refiner_conf)
        
        self.dist_maps_base = DistMaps(norm_radius=base_radius, spatial_scale=1.0,
                                      cpu_mode=False, use_disks=True)
        refiner_spatial_scale = 1.0
        self.dist_maps_refine = DistMaps(norm_radius=5, spatial_scale=refiner_spatial_scale,
                                        cpu_mode=False, use_disks=True)
                                    
    def get_coord_features(self, image, prev_mask, points, is_first_point=False):
        new_points = points.clone()
        if is_first_point:
            new_points[new_points[:, :, 2] > 0] = -1 #index = 0 is the first center click
        coord_features, coord_dist_map = self.dist_maps_base(image, new_points, out_dist_map=True)
        if prev_mask is not None:
            coord_features = torch.cat((prev_mask, coord_features), dim=1)
        return coord_features, coord_dist_map
    
    def backbone_forward(self, image, coord_features=None, points=None, coord_dist_map=False):
        mask, mask_aux, feature = self.feature_extractor(image, coord_features)
        return {'instances': mask, 'instances_aux':mask_aux, 'feature': feature}

    def refine(self, image, points, full_feature, full_logits):
        '''
        bboxes : [b,5]
        '''
        full_logits = F.interpolate(full_logits, image.shape[-2:], mode='bilinear', align_corners=True)
        click_map, coord_dist_map = self.dist_maps_refine(image, points, out_dist_map=True)
        refined_mask = self.refiner(image, click_map, full_feature, full_logits, points, coord_dist_map)
        return {'instances_refined': refined_mask}

    def forward(self, image, points, cached_outputs=None, cached_instances_lr=None):
        image, prev_mask = self.prepare_input(image)
        if cached_outputs is None:
            coord_features, coord_dist_map = self.get_coord_features(image, prev_mask, points, is_first_point=True)
            click_map = coord_features[:,1:,:,:]
            if self.only_first_click:
                coord_features = coord_features[:,[1],:,:]

            coord_features = self.maps_transform(coord_features)
            outputs = self.backbone_forward(image, coord_features, points, coord_dist_map)
            outputs['instances_lr'] = outputs['instances'].detach()
            outputs['click_map'] = click_map
            outputs['instances'] = nn.functional.interpolate(outputs['instances'], size=image.size()[2:],
                                                            mode='bilinear', align_corners=True)
            if self.with_aux_output:
                outputs['instances_aux'] = nn.functional.interpolate(outputs['instances_aux'], size=image.size()[2:],
                                                                mode='bilinear', align_corners=True)
        else:
            outputs = cached_outputs
        
        instances_lr = outputs['instances_lr'] if cached_instances_lr is None else cached_instances_lr
        refine_output = self.refine(image, points, outputs['feature'], instances_lr.detach().clone())
        outputs['instances_lr'] = refine_output['instances_refined'].detach()
        refine_output['instances_refined'] = F.interpolate(refine_output['instances_refined'], size=image.size()[2:],mode='bilinear',align_corners=True)
        return outputs, refine_output

class AttRefineAfterLayer(nn.Module):
    def __init__(self, input_dims=6, mid_dims=96, feature_dims=96, num_classes=1, spatial_scale=0.25, corr_channel=96, conv_layer='xconv2', cross_attention_head=4):
        super(AttRefineAfterLayer, self).__init__()
        self.num_classes = num_classes
        self.spatial_scale = spatial_scale
        self.mid_dims = mid_dims
        self.corr_channel = corr_channel
        if conv_layer == 'xconv':
            ConvLayer = XConvBnRelu
        elif conv_layer == 'xconv2':
            ConvLayer = XConvBnRelu2
        self.feature_convq = nn.Conv2d(mid_dims, mid_dims, 1)
        self.feature_convk = nn.Conv2d(mid_dims, mid_dims, 1)

        self.corr_conv = ConvLayer(self.corr_channel, self.corr_channel)
        self.corr_conv_pos = self.corr_conv
        self.corr_conv_neg = self.corr_conv
        
        self.corr_fusion = ConvLayer(self.corr_channel * 2, mid_dims)
        
        self.image_conv1 = nn.Sequential(nn.PixelUnshuffle(4), ConvLayer(input_dims*16, mid_dims))

        self.image_conv2 = ConvLayer(mid_dims, mid_dims)

        self.refine_fusion = ConvModule(
            in_channels= feature_dims,
            out_channels= mid_dims,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.cross_attention = nn.MultiheadAttention(mid_dims, cross_attention_head)

        fuse_in_ch = mid_dims * 2

        self.fusion1 = nn.Sequential(
            ConvLayer(fuse_in_ch, mid_dims),
            ConvLayer(mid_dims, mid_dims)
        )
        self.all_fusion = ConvLayer(mid_dims * 2, mid_dims)
        self.refine_fusion1 = ConvLayer( mid_dims, mid_dims)
        self.refine_fusion2 = ConvLayer( mid_dims, mid_dims)
        self.refine_pred = nn.Conv2d(mid_dims, num_classes,3,1,1)
        
    def forward(self, image, click_map, final_feature, cropped_logits, points, coord_dist_map):
        #points : b * 2n * 3
        #click_map: b * 2 * h * w
        #coord_dist_map: b * 2 * h * w
        points = points.clone()
        #process image and mask
        mask = cropped_logits
        if mask is not None:
            bin_mask = torch.sigmoid(mask) #> 0.49
        else:
            bin_mask = torch.zeros_like(image[:, :1], dtype=image.dtype, device=image.device)
        input_image = torch.cat([image,click_map,bin_mask], 1)
        image_feature = self.image_conv1(input_image)
        image_feature = self.image_conv2(image_feature)

        final_feature = self.refine_fusion(final_feature)

        if final_feature.shape[-1] != image_feature.shape[-1] or final_feature.shape[-2] != image_feature.shape[-2]:
            final_feature = F.interpolate(final_feature, size=image_feature.size()[2:],mode='bilinear',align_corners=True)
        
        fuse_feature = torch.cat([final_feature, image_feature], dim=1)
        fuse_feature = self.fusion1(fuse_feature)

        batch_size, c, h, w = fuse_feature.shape
        points[:, :, :2] = torch.round(points[:, :, :2] * self.spatial_scale)
        points[:, :, 0][points[:, :, 0] >= h] = h - 1 
        points[:, :, 1][points[:, :, 1] >= w] = w - 1
        points = points.long()
        num_points = points.shape[1] // 2
        
        coord_dist_map[coord_dist_map == 1e6] = 1
        coord_dist_map = torch.sqrt(coord_dist_map) / torch.sqrt(torch.tensor(coord_dist_map.shape[-2]**2 + coord_dist_map.shape[-1]**2, dtype=torch.float32, device=coord_dist_map.device))
        if coord_dist_map.shape[-2] != h or coord_dist_map.shape[-1] != w:
            coord_dist_map = F.interpolate(coord_dist_map, (h, w), mode='bilinear', align_corners=True)
        coord_dist_map = coord_dist_map.reshape(batch_size, 2, h*w)
        
        with torch.no_grad():
            features = fuse_feature.detach().clone()
            features = features.permute(0, 2, 3, 1).reshape(batch_size, h*w, c) #b * hw * c
            points_idx = points[:, :, 0] * w + points[:, :, 1] #b * 2n
            points_idx[points_idx < 0] = 0
            point_features = torch.gather(features, 1, points_idx.unsqueeze(-1).expand(batch_size, -1, c)) #b * 2n * c
            point_features = F.normalize(point_features, p=2.0, dim=-1)
            features = F.normalize(features, p=2.0, dim=-1)
            similarity = torch.einsum('bnc,bmc->bnm', point_features, features) # b * 2n * wh
            similarity[points[:,:,2] < 0] = 0
            positive_similarity = similarity[:, :num_points].max(dim=1)[0]
            negative_similarity = similarity[:, num_points:].max(dim=1)[0] # b * wh

            positive_scores = (positive_similarity + (1 - negative_similarity)) * (1 - coord_dist_map[:, 0])
            negative_scores = (negative_similarity + (1 - positive_similarity)) * (1 - coord_dist_map[:, 1])
            
            positive_value, positive_indices = torch.topk(positive_scores, self.corr_channel, dim=-1)
            negative_value, negative_indices = torch.topk(negative_scores, self.corr_channel, dim=-1)
        
        #correlation
        feature_q = self.feature_convq(fuse_feature)
        feature_k = self.feature_convk(fuse_feature)

        new_c = feature_q.shape[1]
        feature_q = feature_q.permute(0, 2, 3, 1).reshape(batch_size, h*w, new_c)
        selected_positive_features = torch.gather(feature_q, 1, positive_indices.unsqueeze(-1).expand(batch_size, -1, new_c)) # b * K * new_c
        selected_negative_features = torch.gather(feature_q, 1, negative_indices.unsqueeze(-1).expand(batch_size, -1, new_c)) # b * K * new_c

        positive_kernel = selected_positive_features.reshape(-1, new_c, 1, 1)
        feature_k = feature_k.reshape(1, -1, h, w)
        positive_corr = F.conv2d(feature_k, positive_kernel, groups=batch_size).reshape(batch_size, -1, h, w)
        positive_corr = self.corr_conv_pos(positive_corr)

        negative_kernel = selected_negative_features.reshape(-1, new_c, 1, 1)
        negative_corr = F.conv2d(feature_k, negative_kernel, groups=batch_size).reshape(batch_size, -1, h, w)
        negative_corr = self.corr_conv_neg(negative_corr)
        corr_res = torch.cat([positive_corr, negative_corr], dim=1)

        corr_res = self.corr_fusion(corr_res)

        #cross attention
        new_c = fuse_feature.shape[1]
        feature_reshape = fuse_feature.permute(0, 2, 3, 1).reshape(batch_size, -1, new_c)
        indices = positive_indices
        indices = torch.cat([positive_indices, negative_indices], dim=-1)
        kv = torch.gather(feature_reshape, 1, indices.unsqueeze(-1).expand(batch_size, -1, new_c)).permute(1, 0, 2) # b * K * new_c
        fuse_feature = fuse_feature + self.cross_attention(feature_reshape.permute(1, 0, 2), kv, kv)[0].permute(1, 2, 0).reshape(*fuse_feature.shape)
                        
        fuse_feature = torch.cat([fuse_feature, corr_res], dim=1)

        fuse_feature = self.all_fusion(fuse_feature)

        fuse_feature = self.refine_fusion1(fuse_feature)
        fuse_feature = self.refine_fusion2(fuse_feature)
        pred_full = self.refine_pred(fuse_feature)

        return pred_full

class ConvModule(nn.Module):
    def __init__(self, in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
            ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()
       
    def forward(self, x):
        return self.activation( self.norm( self.conv(x)  ) )

class XConvBnRelu(nn.Module):
    """
    Xception conv bn relu
    """
    def __init__(self, input_dims = 3, out_dims = 16, kernel_size=3, stride=1, padding=1):
        super(XConvBnRelu, self).__init__()
        self.conv3x3 = nn.Conv2d(input_dims,input_dims,kernel_size=kernel_size,stride=stride,padding=padding,groups=input_dims) 
        self.conv1x1 = nn.Conv2d(input_dims,out_dims,1,1,0)
        self.norm = nn.BatchNorm2d(out_dims)
        self.activation = nn.ReLU()
    def forward(self,x):
        x = self.conv3x3(x)
        x = self.conv1x1(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

class XConvBnRelu2(nn.Module):
    """
    Xception conv bn relu
    """
    def __init__(self, input_dims = 3, out_dims = 16,   **kwargs):
        super(XConvBnRelu2, self).__init__()
        self.conv3x3_1 = nn.Conv2d(input_dims,input_dims,3,1,1,groups=input_dims)
        self.norm1 = nn.BatchNorm2d(input_dims)
        self.conv3x3_2 = nn.Conv2d(input_dims,input_dims,3,1,1,groups=input_dims)
        self.conv1x1 = nn.Conv2d(input_dims,out_dims,1,1,0)
        self.norm2 = nn.BatchNorm2d(out_dims)
        self.activation = nn.ReLU()
    def forward(self,x):
        x = self.conv3x3_1(x)
        x = self.norm1(x)
        x = self.conv3x3_2(x)
        x = self.conv1x1(x)
        x = self.norm2(x)
        x = self.activation(x)
        return x

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           ):
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)
