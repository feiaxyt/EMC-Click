import torch.nn as nn
import torch
import torch.nn.functional as F
from isegm.utils.serialization import serialize
from .is_model import ISModel
from isegm.model.ops import DistMaps, ScaleLayer, BatchImageNormalize
from .modeling.hrnet_ocr import HighResolutionNet
from isegm.model.modifiers import LRMult
from .modeling.segformer.segformer_model import SegFormer
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
import torchvision.ops.roi_align as roi_align
from isegm.model.ops import DistMaps
from isegm.model.is_refine_model import AttRefineAfterLayer


class SegFormerModel(ISModel):
    @serialize
    def __init__(self, feature_stride = 4, backbone_lr_mult=0.1,
                 norm_layer=nn.BatchNorm2d, model_version = 'b0',
                 refiner_conf=dict(), **kwargs):
        super().__init__(norm_layer=norm_layer, **kwargs)

        self.model_version = model_version
        self.feature_extractor = SegFormer(self.model_version, side_dims=self.coord_feature_ch)
        self.feature_extractor.backbone.apply(LRMult(backbone_lr_mult))

        base_radius = 5
                
        if self.model_version == 'b0':
            feature_indim = 256
        else:
            feature_indim = 512
        
        self.refiner = AttRefineAfterLayer(feature_dims=feature_indim, spatial_scale=0.25, **refiner_conf)
        
        refiner_spatial_scale = 1.0
        self.dist_maps_base = DistMaps(norm_radius=base_radius, spatial_scale=1.0,
                                      cpu_mode=False, use_disks=True)

        self.dist_maps_refine = DistMaps(norm_radius=5, spatial_scale=refiner_spatial_scale,
                                        cpu_mode=False, use_disks=True)

        self.maps_transform = None

    def get_coord_features(self, image, prev_mask, points, is_first_point=False):
        new_points = points.clone()
        if is_first_point:
            new_points[new_points[:, :, 2] > 0] = -1 #index = 0 is the first center click
        coord_features, coord_dist_map = self.dist_maps_base(image, new_points, out_dist_map=True)
        if prev_mask is not None:
            coord_features = torch.cat((prev_mask, coord_features), dim=1)
        return coord_features, coord_dist_map

    def backbone_forward(self, image, coord_features=None, points=None, coord_dist_map=False):
        mask, feature = self.feature_extractor(image, coord_features)
        return {'instances': mask, 'instances_aux':mask, 'feature': feature}

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

            #small_coord_features = self.maps_transform(small_coord_features)
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

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           ):
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)
