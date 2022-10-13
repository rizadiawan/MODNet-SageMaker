import os
import sys
import io
import json
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from src.models.modnet import MODNet

# defining model and loading weights to it.
def model_fn(model_dir):
    try:
        modnet = MODNet(backbone_pretrained=False)
        modnet = nn.DataParallel(modnet)

        with open(os.path.join(model_dir, "pretrained/modnet_photographic_portrait_matting.ckpt"), "rb") as f:
            if torch.cuda.is_available():
                modnet = modnet.cuda()
                weights = torch.load(f)
            else:
                weights = torch.load(f, map_location=torch.device('cpu'))
            modnet.load_state_dict(weights)

        modnet.eval()
        return modnet
    except Exception as e:
        print("Error model_fn")
        print(e)
        return 0

# data preprocessing
def input_fn(request_body, request_content_type):
    try:
        # assert (request_content_type == "application/x-image" or request_content_type.startswith("image/"))

        # define hyper-parameters
        ref_size = 512

        # define image to tensor transform
        im_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

        im = Image.open(io.BytesIO(request_body))
        im_format = im.format

        # unify image channels to 3
        im = np.asarray(im)
        if len(im.shape) == 2:
            im = im[:, :, None]
        if im.shape[2] == 1:
            im = np.repeat(im, 3, axis=2)
        elif im.shape[2] == 4:
            im = im[:, :, 0:3]

        # convert image to PyTorch tensor
        im = Image.fromarray(im)
        im = im_transform(im)

        # add mini-batch dim
        im = im[None, :, :, :]

        # resize image for input
        im_b, im_c, im_h, im_w = im.shape
        if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
            if im_w >= im_h:
                im_rh = ref_size
                im_rw = int(im_w / im_h * ref_size)
            elif im_w < im_h:
                im_rw = ref_size
                im_rh = int(im_h / im_w * ref_size)
        else:
            im_rh = im_h
            im_rw = im_w

        im_rw = im_rw - im_rw % 32
        im_rh = im_rh - im_rh % 32
        im = F.interpolate(im, size=(im_rh, im_rw), mode='area')

        return [im, im_h, im_w, im_format]
    except Exception as e:
        print("Error input_fn")
        print(e)
        return 0
        

# inference
def predict_fn(input_object, model):
    try:
        im, im_h, im_w, im_format = input_object
        _, _, matte = model(im.cuda() if torch.cuda.is_available() else im, True)

        return [matte, im_h, im_w, im_format]
    except Exception as e:
        print("Error predict_fn")
        print(e)
        return 0

# postprocess
def output_fn(predictions, content_type):
    try:
        matte, im_h, im_w, im_format = predictions
        # assert content_type == "application/json"
        matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
        matte = matte[0][0].data.cpu().numpy()
        # matte_name = im_name.split('.')[0] + '.png'
        # Image.fromarray(((matte * 255).astype('uint8')), mode='L').save(os.path.join(args.output_path, matte_name))
        im = Image.fromarray(((matte * 255).astype('uint8')), mode='L')
        
        img_byte_arr = io.BytesIO()
        im.save(img_byte_arr, format=im_format)
        img_byte_arr = img_byte_arr.getvalue()

        return img_byte_arr
        # return json.dumps({ "result": im })
    except Exception as e:
        print("Error output_fn")
        print(e)
        return 0