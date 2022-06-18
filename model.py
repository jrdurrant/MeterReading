from functools import lru_cache

import torch
import torchvision
from torchvision.models.detection import mask_rcnn, faster_rcnn


@lru_cache(maxsize=None)
def load_model(model_path=None):
    num_classes = 11  # 10 digits + background

    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    # replace the mask predictor with a new one
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = mask_rcnn.MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    if model_path is not None:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))

    return model
