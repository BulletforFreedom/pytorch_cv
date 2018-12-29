# -*- coding: utf-8 -*-
import six
import numpy as np

import torch

class Boxes(object):    
    
    @staticmethod
    def bbox_iou(box1,boxn,x1y1x2y2=True):
        """
        gpu
        Returns the IoU of two bounding boxes
        """
        if not x1y1x2y2:
            # Transform from center and width to exact coordinates
            b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
            b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
            b2_x1, b2_x2 = boxn[:, 0] - boxn[:, 2] / 2, boxn[:, 0] + boxn[:, 2] / 2
            b2_y1, b2_y2 = boxn[:, 1] - boxn[:, 3] / 2, boxn[:, 1] + boxn[:, 3] / 2
        else:
            # Get the coordinates of bounding boxes
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
            b2_x1, b2_y1, b2_x2, b2_y2 = boxn[:,0], boxn[:,1], boxn[:,2], boxn[:,3]
    
        # get the corrdinates of the intersection rectangle
        inter_rect_x1 =  torch.max(b1_x1, b2_x1)
        inter_rect_y1 =  torch.max(b1_y1, b2_y1)
        inter_rect_x2 =  torch.min(b1_x2, b2_x2)
        inter_rect_y2 =  torch.min(b1_y2, b2_y2)
        # Intersection area
        inter_area =    torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
                        torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
        # Union Area
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    
        iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
    
        return iou
    
    @staticmethod
    def get_nms(image_pred_class,nms_conf):
        i=0
        while i < image_pred_class.size(0)-1:
            
            ious=Boxes.bbox_iou(image_pred_class[i].unsqueeze(0),image_pred_class[i+1:])
            #Zero out all the detections that have IoU > treshhold
            iou_mask = (ious < nms_conf).float().unsqueeze(1)
            image_pred_class[i+1:] *= iou_mask       
            
            #Remove the non-zero entries
            non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()
            image_pred_class = image_pred_class[non_zero_ind]
            try:
                image_pred_class.size(1)
            except:
                return image_pred_class.unsqueeze(0)
            i+=1
        return image_pred_class
      
    @staticmethod 
    def non_max_suppression(prediction, num_classes, conf_thres=0.5, nms_thres=0.4):
        """
        Removes detections with lower object confidence score than 'conf_thres' and performs
        Non-Maximum Suppression to further filter detections.
        Returns detections with shape:
            (x1, y1, x2, y2, object_conf, class_score, class_pred)
        """
    
        # From (center x, center y, width, height) to (x1, y1, x2, y2)
        box_corner = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]
    
        output = [None for _ in range(len(prediction))]
        for image_i, image_pred in enumerate(prediction):
            # Filter out confidence scores below threshold
            conf_mask = (image_pred[:, 4] >= conf_thres).squeeze()
            image_pred = image_pred[conf_mask]
            # If none are remaining => process next image
            if not image_pred.size(0):
                continue
            # Get score and class with highest confidence
            class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1,  keepdim=True)
            # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
            detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)
            # Iterate through all predicted classes
            unique_labels = detections[:, -1].cpu().unique()
            if prediction.is_cuda:
                unique_labels = unique_labels.cuda()
            for c in unique_labels:
                # Get the detections with the particular class
                detections_class = detections[detections[:, -1] == c]
                # Sort the detections by maximum objectness confidence
                _, conf_sort_index = torch.sort(detections_class[:, 4], descending=True)
                detections_class = detections_class[conf_sort_index]
                # Perform non-maximum suppression
                max_detections = []
                while detections_class.size(0):
                    # Get detection with highest confidence and save as max detection
                    max_detections.append(detections_class[0].unsqueeze(0))
                    # Stop if we're at the last detection
                    if len(detections_class) == 1:
                        break
                    # Get the IOUs for all boxes with lower confidence
                    ious = Boxes.bbox_iou(max_detections[-1], detections_class[1:])
                    # Remove detections with IoU >= NMS threshold
                    detections_class = detections_class[1:][ious < nms_thres]
    
                max_detections = torch.cat(max_detections).data
                # Add max detections to outputs
                output[image_i] = max_detections if output[image_i] is None else torch.cat((output[image_i], max_detections))
    
        return output
    
    @staticmethod
    def generate_anchor_base(base_size=16, ratios=[0.5, 1, 2],
                         anchor_scales=[8, 16, 32]):
        """Generate anchor base windows by enumerating aspect ratio and scales.
    
        Generate anchors that are scaled and modified to the given aspect ratios.
        Area of a scaled anchor is preserved when modifying to the given aspect
        ratio.
    
        :obj:`R = len(ratios) * len(anchor_scales)` anchors are generated by this
        function.
        The :obj:`i * len(anchor_scales) + j` th anchor corresponds to an anchor
        generated by :obj:`ratios[i]` and :obj:`anchor_scales[j]`.
    
        For example, if the scale is :math:`8` and the ratio is :math:`0.25`,
        the width and the height of the base window will be stretched by :math:`8`.
        For modifying the anchor to the given aspect ratio,
        the height is halved and the width is doubled.
    
        Args:
            base_size (number): The width and the height of the reference window.
            ratios (list of floats): This is ratios of width to height of
                the anchors.
            anchor_scales (list of numbers): This is areas of anchors.
                Those areas will be the product of the square of an element in
                :obj:`anchor_scales` and the original area of the reference
                window.
    
        Returns:
            ~numpy.ndarray:
            An array of shape :math:`(R, 4)`.
            Each element is a set of coordinates of a bounding box.
            The second axis corresponds to
            :math:`(y_{min}, x_{min}, y_{max}, x_{max})` of a bounding box.
    
        """
        py = base_size / 2.
        px = base_size / 2.
    
        anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4),
                               dtype=np.float32)
        for i in six.moves.range(len(ratios)):
            for j in six.moves.range(len(anchor_scales)):
                h = base_size * anchor_scales[j] * np.sqrt(ratios[i])
                w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])
    
                index = i * len(anchor_scales) + j
                anchor_base[index, 0] = py - h / 2.
                anchor_base[index, 1] = px - w / 2.
                anchor_base[index, 2] = py + h / 2.
                anchor_base[index, 3] = px + w / 2.
        return anchor_base
    
if __name__=='__main__':
    achors=Boxes.generate_anchor_base()
