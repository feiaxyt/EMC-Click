import torch
import torch.nn.functional as F
from torchvision import transforms
from isegm.inference.transforms import AddHorizontalFlip, SigmoidForPred, LimitLongestSide, ResizeTrans
from isegm.utils.crop_local import  map_point_in_bbox,get_focus_cropv1, get_focus_cropv2, get_object_crop, get_click_crop


class ClickAttPredictor(object):
    def __init__(self, model, device,
                 net_clicks_limit=None,
                 with_flip=False,
                 zoom_in=None,
                 max_size=None,
                 infer_size = 256,
                 focus_crop_r = 1.4,
                 **kwargs):
        self.with_flip = with_flip
        self.net_clicks_limit = net_clicks_limit
        self.original_image = None
        self.device = device
        self.zoom_in = zoom_in
        self.prev_prediction = None
        self.model_indx = 0
        self.click_models = None
        self.net_state_dict = None

        if isinstance(model, tuple):
            self.net, self.click_models = model
        else:
            self.net = model

        self.to_tensor = transforms.ToTensor()

        self.transforms = []
        if max_size is not None:
            self.transforms.append(LimitLongestSide(max_size=max_size))
        self.crop_l = infer_size
        self.focus_crop_r = focus_crop_r
        self.transforms.append(ResizeTrans(self.crop_l))
        self.transforms.append(SigmoidForPred())
        self.focus_roi = None 
        self.global_roi = None 

        if self.with_flip:
            self.transforms.append(AddHorizontalFlip())

    def set_input_image(self, image):
        image_nd = self.to_tensor(image)
        for transform in self.transforms:
            transform.reset()
        self.original_image = image_nd.to(self.device)
        if len(self.original_image.shape) == 3:
            self.original_image = self.original_image.unsqueeze(0)
        self.prev_prediction = torch.zeros_like(self.original_image[:, :1, :, :])
        self.prev_outputs = None

    def set_prev_mask(self, mask):
        self.prev_prediction = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(self.device).float()

    def get_prediction(self, clicker, prev_mask=None):
        clicks_list = clicker.get_clicks()
        click = clicks_list[-1]
        last_y,last_x = click.coords[0],click.coords[1]
        self.last_y = last_y
        self.last_x = last_x

        if self.click_models is not None:
            model_indx = min(clicker.click_indx_offset + len(clicks_list), len(self.click_models)) - 1
            if model_indx != self.model_indx:
                self.model_indx = model_indx
                self.net = self.click_models[model_indx]

        input_image = self.original_image
        if prev_mask is None:
            prev_mask = self.prev_prediction
        if hasattr(self.net, 'with_prev_mask') and self.net.with_prev_mask:
            input_image = torch.cat((input_image, prev_mask), dim=1)

        image_nd, clicks_lists, is_image_changed = self.apply_transforms(
            input_image, [clicks_list]
        )
    
        outputs, refine_outputs = self._get_prediction(image_nd, clicks_lists, self.prev_outputs)
        prediction = refine_outputs['instances_refined']
        for t in reversed(self.transforms):
            prediction = t.inv_transform(prediction)
        self.prev_outputs = outputs
        self.prev_prediction = prediction
        return prediction.cpu().numpy()[0, 0]

    def _get_prediction(self, image_nd, clicks_lists, prev_outputs=None):
        points_nd = self.get_points_nd(clicks_lists)
        outputs, refine_outputs =  self.net(image_nd, points_nd, prev_outputs)
        return outputs, refine_outputs

    def mapp_roi(self, focus_roi, global_roi):
        yg1,yg2,xg1,xg2 = global_roi
        hg,wg = yg2-yg1, xg2-xg1
        yf1,yf2,xf1,xf2 = focus_roi
        
        yf1_n = (yf1-yg1) * (self.crop_l/hg)
        yf2_n = (yf2-yg1) * (self.crop_l/hg)
        xf1_n = (xf1-xg1) * (self.crop_l/wg)
        xf2_n = (xf2-xg1) * (self.crop_l/wg)

        yf1_n = max(yf1_n,0)
        yf2_n = min(yf2_n,self.crop_l)
        xf1_n = max(xf1_n,0)
        xf2_n = min(xf2_n,self.crop_l)
        return (yf1_n,yf2_n,xf1_n,xf2_n)

    def _get_transform_states(self):
        return [x.get_state() for x in self.transforms]

    def _set_transform_states(self, states):
        assert len(states) == len(self.transforms)
        for state, transform in zip(states, self.transforms):
            transform.set_state(state)
        print('_set_transform_states')

    def apply_transforms(self, image_nd, clicks_lists):
        is_image_changed = False
        for t in self.transforms:
            image_nd, clicks_lists = t.transform(image_nd, clicks_lists)
            is_image_changed |= t.image_changed

        return image_nd, clicks_lists, is_image_changed

    def get_points_nd(self, clicks_lists):
        total_clicks = []
        num_pos_clicks = [sum(x.is_positive for x in clicks_list) for clicks_list in clicks_lists]
        num_neg_clicks = [len(clicks_list) - num_pos for clicks_list, num_pos in zip(clicks_lists, num_pos_clicks)]
        num_max_points = max(num_pos_clicks + num_neg_clicks)
        if self.net_clicks_limit is not None:
            num_max_points = min(self.net_clicks_limit, num_max_points)
        num_max_points = max(1, num_max_points)

        for clicks_list in clicks_lists:
            clicks_list = clicks_list[:self.net_clicks_limit]
            pos_clicks = [click.coords_and_indx for click in clicks_list if click.is_positive]
            pos_clicks = pos_clicks + (num_max_points - len(pos_clicks)) * [(-1, -1, -1)]

            neg_clicks = [click.coords_and_indx for click in clicks_list if not click.is_positive]
            neg_clicks = neg_clicks + (num_max_points - len(neg_clicks)) * [(-1, -1, -1)]
            total_clicks.append(pos_clicks + neg_clicks)

        return torch.tensor(total_clicks, device=self.device)

    def get_points_nd_inbbox(self, clicks_list, y1,y2,x1,x2):
        total_clicks = []
        num_pos = sum(x.is_positive for x in clicks_list)
        num_neg =len(clicks_list) - num_pos 
        num_max_points = max(num_pos, num_neg)
        num_max_points = max(1, num_max_points)
        pos_clicks, neg_clicks = [],[]
        for click in clicks_list:
            flag,y,x,index = click.is_positive, click.coords[0],click.coords[1], 0
            y,x = map_point_in_bbox(y,x,y1,y2,x1,x2,self.crop_l)
            if flag:
                pos_clicks.append( (y,x,index))
            else:
                neg_clicks.append( (y,x,index) )

        pos_clicks = pos_clicks + (num_max_points - len(pos_clicks)) * [(-1, -1, -1)]
        neg_clicks = neg_clicks + (num_max_points - len(neg_clicks)) * [(-1, -1, -1)]
        total_clicks.append(pos_clicks + neg_clicks)
        return torch.tensor(total_clicks, device=self.device)

    def get_states(self):
        return {
            'transform_states': self._get_transform_states(),
            'prev_prediction': self.prev_prediction.clone(),
            'prev_outputs': self.prev_outputs
        }

    def set_states(self, states):
        self._set_transform_states(states['transform_states'])
        self.prev_prediction = states['prev_prediction']
        self.prev_outputs = states['prev_outputs']
        print('set')
