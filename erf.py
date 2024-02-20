#
# Copyright (c) Microsoft Corporation.
#

#
# Methods for compressing a network using an ensemble of interpolants
#

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import time
import pandas as pd
import argparse
import random
import cv2
from wrapper import Wrapper
from torchvision import datasets, transforms,models
from matplotlib import pyplot as plt
import json


def erf_project(inp, wrapped_model, out_l, layer_idx, inp_l = None, attr = None, recover =True):
    inp.requires_grad = True
    if out_l == 0:
        return attr
    inp.grad = None
    if not wrapped_model.const:
        wrapped_model.set_const()
    wrapped_model.set_cur_idx_fix(layer_idx)
    wrapped_model.relu_remove_hook(layer_idx)
    #wrapped_model.maxpool_remove_hook(layer_idx)
    if inp_l is not None:
        #hook = wrapped_model.order_layers[inp_l].register_full_backward_hook(wrapped_model.backward_grad_hook())
        hook = inp_l.register_full_backward_hook(wrapped_model.backward_grad_hook())

    mid = wrapped_model.get_mid_out(inp, out_l)
    if attr is None:
        mid.backward(gradient = torch.ones_like(mid))
    else:
        mid.backward(gradient = attr)
    if inp_l is None:
        erf_result = inp.grad
    else:
        erf_result = wrapped_model.mid_grad
        hook.remove()

    wrapped_model.remove_relu_remove_hook()
    #wrapped_model.remove_maxpool_remove_hook()
    if recover:
        wrapped_model.model_recover()
    return erf_result

def erf(inp, wrapped_model, out_l,inp_l = None, attr = None):
    inp.requires_grad = True
    inp.grad = None
    if inp_l is not None:
        #hook = wrapped_model.order_layers[inp_l].register_full_backward_hook(wrapped_model.backward_grad_hook())
        hook = inp_l.register_full_backward_hook(wrapped_model.backward_grad_hook())
    mid = wrapped_model.get_mid_out(inp, out_l)
    #mid = module(mid_in)
    if attr is None:
        mid.backward(gradient = torch.ones_like(mid))
    else:
        mid.backward(gradient = attr)
    if inp_l is None:
        erf_result = inp.grad
    else:
        #erf_result = wrapped_model.mid_grad.abs().max(1, keepdim=True)[0]
        erf_result = wrapped_model.mid_grad
        hook.remove()
    return erf_result

def erf_const(inp, wrapped_model, out_l, layer_idx, inp_l = None, attr = None, recover = True):
    inp.requires_grad = True
    inp.grad = None
    if not wrapped_model.const:
        wrapped_model.set_const()
    wrapped_model.set_cur_idx_fix(layer_idx)
    if inp_l is not None:
        hook = inp_l.register_full_backward_hook(wrapped_model.backward_grad_hook())
    mid = wrapped_model.get_mid_out(inp, out_l)
    #mid = module(mid_in)
    if attr is None:
        mid.backward(gradient = torch.ones_like(mid))
    else:
        mid.backward(gradient = attr)
    if inp_l is None:
        erf_result = inp.grad
    else:
        erf_result = wrapped_model.mid_grad
        hook.remove()
    if recover:
        wrapped_model.model_recover()
    return erf_result

def erf_const_total(inp, wrapped_model, target = None, recover = True):
    inp.requires_grad = True
    inp.grad = None
    if not wrapped_model.const:
        wrapped_model.set_const()
    wrapped_model.set_cur_idx_fix(len(wrapped_model.order_layers)-1)
    out = wrapped_model.model(inp)
    out[target].sum().backward()
    erf_result = inp.grad
    if recover:
        wrapped_model.model_recover()
    return erf_result


def get_regular_erf(wrapped_model, original_imgs,  args, visualize):
    masks = dict()

    for i,m in enumerate(wrapped_model.order_layers):
        #if not isinstance(m, nn.ReLU):
        if not (isinstance(m, nn.ReLU) or isinstance(m, nn.MaxPool2d)):
            continue

        wrapped_model.reg_norm()
        erf_norm = erf(original_imgs, wrapped_model, 0,i)
        erf_norm = erf_norm.abs().max(1, keepdim=True)[0]
        wrapped_model.remove_norm_hook()

        wrapped_model.reg_guide()
        erf_guide = erf(original_imgs, wrapped_model, 0,i)
        erf_guide = erf_guide.abs().max(1, keepdim=True)[0]
        wrapped_model.remove_guide_hook()

        wrapped_model.maxpool_guide()
        erf_maxpool = erf(original_imgs, wrapped_model,0, i)
        erf_maxpool = erf_maxpool.abs().max(1, keepdim=True)[0]
        wrapped_model.remove_maxpool_guide_hook()

        #hook = wrapped_model.order_layers[1].register_full_backward_hook(wrapped_model.backward_guide_hook())
        erf_res = erf(original_imgs, wrapped_model, 0, i)
        erf_res = erf_res.abs().max(1, keepdim=True)[0]
        #hook.remove()
        #masks[i] = mask
        if visualize:
            cur_dir = "{}/{}/layer_{}".format(args.save_dir, args.input_idx, i)
            if not os.path.exists(cur_dir):
                os.makedirs(cur_dir)
            visualize_attrs(erf_res, original_imgs, args.dataset, cur_dir,"erf")
            visualize_attrs(erf_norm, original_imgs, args.dataset, cur_dir,"norm")
            visualize_attrs(erf_guide, original_imgs, args.dataset, cur_dir,"guide")
            visualize_attrs(erf_maxpool, original_imgs, args.dataset, cur_dir,"maxpool_guide")


    wrapped_model.maxpool_fix()
    wrapped_model.relu_fix()
    for i,m in enumerate(wrapped_model.order_layers):
        if not (isinstance(m, nn.ReLU) or isinstance(m, nn.MaxPool2d)):
            continue

        #wrapped_model.set_cur_idx_fix(i)

        erf_const_res = erf_const(original_imgs, wrapped_model, 0,i)
        erf_const_res = erf_const_res.abs().max(1, keepdim=True)[0]

        if visualize:
            cur_dir = "{}/{}/layer_{}".format(args.save_dir, args.input_idx, i)
            visualize_attrs(erf_const_res, original_imgs, args.dataset, cur_dir, "const")
            with open('{}/commandline_args.txt'.format(cur_dir), 'w') as f:
                json.dump(args.__dict__, f, indent=2)
    wrapped_model.sanity_recover()
    wrapped_model.remove_maxpool_hook()
    wrapped_model.remove_relu_hook()



