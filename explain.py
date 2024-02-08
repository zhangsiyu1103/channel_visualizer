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
import argparse
import random
from wrapper import Wrapper
import utils
from torchvision import datasets, transforms,models
from matplotlib import pyplot as plt
import json


def get_random_reference(inp):
    shape = inp.shape
    color = range(256)
    ref = np.array([random.choice(color), random.choice(color), random.choice(color)])
    reference = np.repeat(ref, shape[2]*shape[3], axis =0).reshape(1,3,shape[2],shape[3]).transpose(0,2,3,1)/255.0
    reference = utils.numpy_to_tensor(reference)
    return reference


def channel_greedy(model, inp, target, batch_size = 256, attr = None, preservation = True, init_idx = None, threshold = False, reference_func = torch.zeros_like, heuristic = "softmax"):
    if not heuristic in ("softmax","crossentropy","logit"):
        raise Exception("heuristic not defined")
        return
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    #attr_step = torch.zeros((*inp.shape[:2], 1,1)).to(device)
    criterion = torch.nn.CrossEntropyLoss(reduction = 'none')
    if threshold:
        with torch.no_grad():
            orig_out = model(inp)
            if heuristic == "softmax":
                orig_out = F.softmax(orig_out/3, 1)[:,target]
                print(orig_out)
            elif heuristic == "crossentropy":
                orig_out = criterion(orig_out, target)
                print(orig_out)
            elif heuristic == "logit":
                orig_out = orig_out[:,target]
    B,C,H,W= inp.shape
    if preservation:
        attr_step = torch.zeros(1,C).to(device)
        if init_idx:
            attr_step[0,init_idx] = 1

        last_val = -torch.tensor(float("inf")).to(device)
    else:
        attr_step = torch.ones(1,C).to(device)
        if init_idx:
            attr_step[0,init_idx] = 0

        last_val = torch.tensor(float("inf")).to(device)
    #attr_step = torch.zeros(1,C)
    #greedy_step = torch.ones(C)
    #greedy_step = torch.diag(greedy_step).unsqueeze(-1).unsqueeze(-1)

    all_idx = []

    if init_idx:
        cur_inp = torch.mul(inp, attr_step.unsqueeze(-1).unsqueeze(-1))
        #cur_inp = torch.einsum('bijk,bi->bijk', inp, cur_step)
        with torch.no_grad():
            out = model(cur_inp)
            out = F.softmax(out,1)
        if preservation:
            if threshold:
                if (out[:,target] >= orig_out).all():
                    return init_idx
            else:
                if (out.max(1)[1] == target).all():
                    return init_idx
        else:
            if (out.max(1)[1] != target).all():
                return init_idx
        all_idx = init_idx


    greedy_step = torch.ones(C).to(device)
    greedy_step = torch.diag(greedy_step)

    while True:
        if preservation:
            cur_step = ((greedy_step + attr_step)>0)
        else:
            cur_step = ((attr_step - greedy_step)>0)
        a = time.time()
        cur_inp = torch.einsum('bijk,mi->mbijk', inp.cpu(), cur_step.cpu())
        #for i in range((B + 127) // 128):
        #    print(i)
        #    cur_inp.append(torch.einsum('bijk,mi->mbijk', inp[i*128:(i+1)*128], cur_step).cpu())
        #cur_inp = torch.cat(cur_inp, dim = 1)
        b = time.time()
        #print(b-a)
        cur_inp = cur_inp.view((-1,C, H,W))
        n_samples = cur_inp.shape[0]
        #print(n_samples)
        all_out = []
        for i in range((n_samples + batch_size -1) // batch_size):
            #print(i)
            with torch.no_grad():
                out = model(cur_inp[i*batch_size:(i+1)*batch_size].to(device))
                #if not threshold:
                    #out = F.softmax(out,1)
                    #out = criterion(out,torch.tensor(target).expand(out.shape[0]).to(device))
                if heuristic == "softmax":
                    out = F.softmax(out/3, 1)
                all_out.append(out)
        all_out = torch.cat(all_out)
        #all_out = all_out.view(C,B,-1)
        #cur_out = criterion(all_out,torch.tensor(target).expand(all_out.shape[0]).to(device))
        #cur_out = cur_out.view(C,B)
        #all_out = all_out.view(C,B,-1)
        if heuristic == "crossentropy":
            c_out = criterion(all_out,target.expand(all_out.shape[0]).to(device))
            all_out = all_out.view(C,B, -1)
            c_out = c_out.view(C,B)
            #cur_out = cur_out.view(C,B)
            cur_out = c_out.sum(1)
            #print(cur_out.shape)
        else:
            all_out = all_out.view(C,B,-1)
            cur_out = all_out.sum(1)[:,target.item()]
        #if heuristic == "crossentropy":
        #    cur_out = cur_out.sum(1)
        #else:
        #    cur_out = all_out.sum(1)[:,target.item()]
        #cur_out = all_out.max(2)[1]
        #print(cur_out)
        #print(cur_out.shape)
        #cur_out = torch.count_nonzero(all_out.max(2)[1] == target, dim = 1)
        #print(cur_out.shape)
        #print(all_out.shape)

        #idx = all_out.max(2)[1]
        #print(idx.shape)
        #idx = torch.argwhere(idx == target)
        #print(idx.shape)
        #cur_out = all_out.sum(1)[:,target]
        #cur_out = all_out[:,:, target]
        #cur_out = torch.where(idx == target, torch.zeros_like(cur_out), cur_out)
        #cur_out = all_out.sum(1)[:,target]
        
        #print(cur_out.shape)
        if preservation:
            #idx = cur_out.argmax().squeeze().item()
            #cur_max, cur_target = all_out.max(2)
            #cur_max = cur_max.sum(1)
            if heuristic == "crossentropy":
                cur_out[all_idx] = float('inf')
                cur_val, idx = cur_out.min(0)
            else:
                cur_out[all_idx] = float('-inf')
                cur_val, idx = cur_out.max(0)
                #print(cur_val)
            #cur_out  = cur_out-cur_max
            #cur_val, idx = cur_out.max(0
            idx = idx.squeeze().item()
            attr_step[0,idx]=1
        else:
            #idx = cur_out.argmin().squeeze().item()
            #cur_val, idx = cur_out.min(0)
            if heuristic == "crossentropy":
                cur_out[all_idx] = float('-inf')
                cur_val, idx = cur_out.max(0)
            else:
                cur_out[all_idx] = float('inf')
                cur_val, idx = cur_out.min(0)
            #cur_out[all_idx] = float('-inf')
            #cur_val, idx = cur_out.max(0)
            idx = idx.squeeze().item()
            attr_step[0,idx]= 0
        all_idx.append(idx)
        #print(len(all_idx))
        print(idx)
        #print(target)
        #print(all_out[idx].max(1)[1])
        #print(cur_out[idx])
        #print(cur_val)
        #print(last_val)
        #print(all_out[idx][:,target])
        #print(all_out[idx][:,639].sum())
        #print(c_out[idx])
        #print(orig_out)
        if preservation:
            if (all_out[idx].max(1)[1] == target).all():
                if threshold:
                    #break
                    if heuristic == "crossentropy":
                        if (c_out[idx] <= orig_out).all():
                            break
                    elif (all_out[idx][:,target] >= orig_out).all():
                        break
                else:
                    break

            last_val = cur_val
        else:
            if (all_out[idx].max(1)[1] != target).all():
                if threshold:
                    if heuristic == "crossentropy":
                        if (c_out[idx] >= orig_out).all():
                            break
                    elif (all_out[idx][:,target] <= orig_out).all():
                        break
                else:
                    break

            last_val = cur_val

    return all_idx
        

def channel_greedy_modify(model, inp, target_pred, target_gt, batch_size = 256, attr = None, preservation = True,  threshold = False, reference_func = torch.zeros_like):
    #attr_step = torch.zeros((*inp.shape[:2], 1,1)).to(device)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    with torch.no_grad():
        orig_out = model(inp).softmax(1)[0,target_pred]
    B,C,H,W= inp.shape
    if preservation:
        attr_step = torch.zeros(1,C).to(device)
        last_val = -torch.tensor(float("inf")).to(device)
    else:
        attr_step = torch.ones(1,C).to(device)
        last_val = torch.tensor(float("inf")).to(device)
    #attr_step = torch.ones(1,C).to(device)
    #last_val = torch.tensor(float("inf")).to(device)
    greedy_step = torch.ones(C).to(device)
    #attr_step = torch.zeros(1,C)
    #greedy_step = torch.ones(C)
    #greedy_step = torch.diag(greedy_step).unsqueeze(-1).unsqueeze(-1)
    greedy_step = torch.diag(greedy_step)
    all_idx = []
    while True:
        if preservation:
            cur_step = ((greedy_step + attr_step)>0)
        else:
            cur_step = ((attr_step - greedy_step)>0)
        #cur_step = ((attr_step - greedy_step)>0)

        cur_inp = torch.einsum('bijk,mi->mbijk', inp, cur_step)
        cur_inp = cur_inp.view((-1,C, H,W))
        n_samples = cur_inp.shape[0]
        all_out = []
        for i in range((n_samples + batch_size -1) // batch_size):
            with torch.no_grad():
                out = model(cur_inp[i*batch_size:(i+1)*batch_size].to(device))
                out = F.softmax(out,1)
                all_out.append(out)
        all_out = torch.cat(all_out)
        all_out = all_out.view(C,B,-1)
        cur_out = all_out.sum(1)
        if preservation:
            #idx = cur_out.argmax().squeeze().item()
            #cur_max, cur_target = all_out.max(2)
            #cur_max = cur_max.sum(1)
            #cur_out  = cur_out-cur_max
            cur_out = cur_out[:, target_pred]
            cur_val, idx = cur_out.max(0)
            idx = idx.squeeze().item()
            attr_step[0,idx]=1
        else:
            #idx = cur_out.argmin().squeeze().item()
            cur_out = cur_out[:, target_pred] - cur_out[:, target_gt]
            cur_val, idx = cur_out.min(0)
            idx = idx.squeeze().item()
            attr_step[0,idx]= 0
        
        all_idx.append(idx)
        if preservation:
            if threshold:
                if (all_out[idx][0,target_pred] >= orig_out).all():
                    break
            else:
                if (all_out[idx].max(1)[1] == target_pred).all():
                    break
            #if cur_val <= last_val:
            #    break
            last_val = cur_val
        else:
            if (all_out[idx].max(1)[1] == target_gt).all():
                break
            #if cur_val >= last_val:
            #    break
            last_val = cur_val

    return all_idx
        






def generate_attr_channel_ind(wrapped_model, inp, target, area,channel_attr, attr = None, mode = "preservation", defense_mode = "IBM", shape = None, lr = 0.01, epochs = 100, beta = 1e-2, reference_func = get_random_reference,  binary = False, threshold = None, area_regulation = True, area_as_ratio = True, mse = False):

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    model = wrapped_model.model
    if mse:
        with torch.no_grad():
            out = model(inp)
        criterion = torch.nn.MSELoss()

    #print(torch.count_nonzero(channel_attr))
    inp = inp.to(device)
    if attr is None:
        if shape is None:
            attr = torch.ones_like(inp)
        else:
            attr = torch.ones(shape).to(device)
    ind_attr = attr.clone()

    ind_attr.requires_grad = True

    n_ele = torch.numel(ind_attr[0])

    optimizer = optim.Adam([ind_attr], lr = lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    if area_as_ratio:
        n = int(n_ele*area)
    else:
        n = area

    wrapped_model.defense()

    #inp_save = inp.clone()
    for i in range(epochs):
        attr = torch.mul(ind_attr, channel_attr)
        reference = reference_func(inp).to(device)
        if area_regulation:
            loss_l1 = i*beta*torch.abs(torch.sum(attr, [1,2,3]) - n).mean()
        else:
            loss_l1 = torch.tensor(0).to(device)

        if binary and i >= threshold:
            loss_force = i*beta*torch.mul(attr, 1 - attr).mean(0).sum()
        else:
            loss_force = torch.tensor(0).to(device)
        loss = loss_l1 + loss_force

        new_attr = attr.expand(*inp.shape)

        loss_pre = torch.tensor(0).to(device)
        loss_del = torch.tensor(0).to(device)
        
        if mode == "preservation" or mode == "hybrid":
            input_pre = torch.mul(inp, new_attr) + torch.mul(reference, 1 - new_attr)
            out_pre = model(input_pre)
            loss_pre = -out_pre[:,target].mean()
            if mse:
                loss_pre = criterion(out_pre[:,target], out[:,target])
        if mode == "deletion" or mode == "hybrid":
            input_del = torch.mul(inp, 1 - new_attr) + torch.mul(reference, new_attr)
            out_del = model(input_del)
            loss_del = out_del[:,target].mean()



        loss = loss + loss_pre + loss_del


        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        scheduler.step()

        ind_attr.data.clamp_(0,1)
        print("epoch {}, loss: {:.4f}, loss l1:{:.4f}, loss force:{:.4f}, loss pre:{:.4f}, loss del:{:.4f}".format(str(i), loss.item(), loss_l1.item(), loss_force.item(),  loss_pre.item(), loss_del.item()))
    wrapped_model.remove_bhooks()
    
    return ind_attr.detach()













def generate_attr(wrapped_model, inp, target, area, attr = None, mode = "preservation", defense_mode = "IBM", shape = None, lr = 0.01, epochs = 100, beta = 1e-2, reference_func = get_random_reference,  binary = False, threshold = None, area_regulation = True, area_as_ratio = True, mse = False):

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    model = wrapped_model.model
    if mse:
        with torch.no_grad():
            out = model(inp)
        criterion = torch.nn.MSELoss()


    inp = inp.to(device)
    if attr is None:
        if shape is None:
            attr = torch.zeros((inp.shape[0],1,*inp.shape[2:])).to(device)
        else:
            attr = torch.zeros(shape).to(device)

    attr.requires_grad = True

    n_ele = torch.numel(attr[0])

    optimizer = optim.Adam([attr], lr = lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    if area_as_ratio:
        n = int(n_ele*area)
    else:
        n = area

    wrapped_model.defense()

    #inp_save = inp.clone()
    for i in range(epochs):
        reference = reference_func(inp).to(device)
        if area_regulation:
            loss_l1 = i*beta*torch.abs(torch.sum(attr, [1,2,3]) - n).mean()
        else:
            loss_l1 = torch.tensor(0).to(device)

        if binary and i >= threshold:
            loss_force = i*beta*torch.mul(attr, 1 - attr).mean(0).sum()
        else:
            loss_force = torch.tensor(0).to(device)
        loss = loss_l1 + loss_force

        new_attr = attr.expand(*inp.shape)

        loss_pre = torch.tensor(0).to(device)
        loss_del = torch.tensor(0).to(device)
        
        if mode == "preservation" or mode == "hybrid":
            input_pre = torch.mul(inp, new_attr) + torch.mul(reference, 1 - new_attr)
            out_pre = model(input_pre)
            loss_pre = -out_pre[:,target].mean()
            if mse:
                loss_pre = criterion(out_pre[:,target], out[:,target])
        if mode == "deletion" or mode == "hybrid":
            input_del = torch.mul(inp, 1 - new_attr) + torch.mul(reference, new_attr)
            out_del = model(input_del)
            loss_del = out_del[:,target].mean()



        loss = loss + loss_pre + loss_del


        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        scheduler.step()

        attr.data.clamp_(0,1)
        print("epoch {}, loss: {:.4f}, loss l1:{:.4f}, loss force:{:.4f}, loss pre:{:.4f}, loss del:{:.4f}".format(str(i), loss.item(), loss_l1.item(), loss_force.item(),  loss_pre.item(), loss_del.item()))
    wrapped_model.remove_bhooks()
    
    return attr.detach()



def generate_attr_all(wrapped_model, inp, target, attr = None, mode = "preservation", defense_mode = "IBM", shape = None, lr = 0.01, epochs = 100, beta = 1e-2, reference_func = get_random_reference, areas = None):

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    model = wrapped_model.model

    inp = inp.to(device)
    if attr is None:
        if shape is None:
            attr = torch.zeros((inp.shape[0],1,*inp.shape[2:])).to(device)
        else:
            attr = torch.zeros(shape).to(device)

    attr.requires_grad = True

    n_ele = torch.numel(attr[0])

    optimizer = optim.Adam([attr], lr = lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    if areas is None:
        areas = [0.1,0.3,0.5,0.7,0.9,1]

    all_n = []
    for area in areas:
        all_n.append(int(n_ele*area))

    wrapped_model.defense()

    for i in range(epochs):
        reference = reference_func(inp).to(device)

        #attr_indicate = attr.clone()
        attr_flat = attr.view(attr.shape[0],-1)

        loss_pre = torch.tensor(0).float().to(device)
        loss_del = torch.tensor(0).float().to(device)

        for n in all_n:
            #print(n_ele)
            #print(n)
            threshold = attr_flat.sort(1)[0][:,-n]
            #attr_sort, indices = attr_flat.sort(1)
            #attr_sort[:]=0
            #attr_sort[:,-n:] = 1
            #attr_sort = attr_sort.gather(
            #print(torch.count_nonzero(attr_sort))
            #print(torch.count_nonzero(attr_flat))
            #print(target)
            mask = torch.where(attr >= threshold, 1, 0)
            #mask = torch.where(attr_sort ==1, 1, 0)
            new_attr = torch.mul(attr,mask)
            new_attr = new_attr.expand(*inp.shape)
            if mode == "preservation" or mode == "hybrid":
                input_pre = torch.mul(inp, new_attr) + torch.mul(reference, 1 - new_attr)
                out_pre = model(input_pre)
                loss_pre += (-out_pre[:,target].mean())
            if mode == "deletion" or mode == "hybrid":
                input_del = torch.mul(inp, 1 - new_attr) + torch.mul(reference, new_attr)
                out_del = model(input_del)
                loss_del += out_del[:,target].mean()



        loss = loss_pre + loss_del


        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        scheduler.step()

        attr.data.clamp_(0,1)
        #print("epoch {}, loss: {:.4f}, loss l1:{:.4f}, loss force:{:.4f}, loss pre:{:.4f}, loss del:{:.4f}".format(str(i), loss.item(), loss_l1.item(), loss_force.item(),  loss_pre.item(), loss_del.item()))
        print("epoch {}, loss: {:.4f}, loss pre:{:.4f}, loss del:{:.4f}".format(str(i), loss.item(), loss_pre.item(), loss_del.item()))
    wrapped_model.remove_bhooks()
    
    return attr.detach()




def channel_visualize_image(model, inp, target, defense_mode = "IBM",  lr = 0.1, epochs = 100, save_dir = None):

    wrapped_model = Wrapper(model, defense_mode)

    wrapped_model.pre_defense(inp, start = 25)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    model = wrapped_model.model

    inp = inp.to(device)
    black = np.zeros((1,224,224,3))
    black = utils.numpy_to_tensor(black).to(device)
    white = np.ones((1,224,224,3))
    white = utils.numpy_to_tensor(white).to(device)
    #rec_inp = (white-black)*torch.rand_like(inp)+black
    rec_inp = inp.clone()
    rec_inp.requires_grad = True


    optimizer = optim.Adam([rec_inp], lr = lr)

    wrapped_model.defense()

    for i in range(epochs):
        out_pre = model(rec_inp)

        loss = -out_pre[:,target].mean()


        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        rec_inp.data.clamp_(black, white)
        print("epoch {}, loss: {:.7f}".format(str(i), loss.item()))
    utils.visualize_imgs(inp, rec_inp, save_dir)
    wrapped_model.remove_bhooks()

    return rec_inp.detach()










def image_recover(model, inp, target, defense_mode = "IBM",  lr = 0.1, epochs = 100, save_dir = None):

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    wrapped_model = Wrapper(model, defense_mode)

    wrapped_model.pre_defense(inp)

    model = wrapped_model.model

    inp = inp.to(device)
    black = np.zeros((1,224,224,3))
    black = utils.numpy_to_tensor(black).to(device)
    white = np.ones((1,224,224,3))
    white = utils.numpy_to_tensor(white).to(device)
    rec_inp = (white-black)*torch.rand_like(inp)+black
    rec_inp.requires_grad = True


    optimizer = optim.Adam([rec_inp], lr = lr)

    wrapped_model.defense()

    for i in range(epochs):
        out_pre = model(rec_inp)

        loss = -out_pre[:,target].mean()


        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        rec_inp.data.clamp_(black, white)
        print("epoch {}, loss: {:.7f}".format(str(i), loss.item()))
    utils.visualize_imgs(inp, rec_inp, save_dir)
    wrapped_model.remove_bhooks()

    return rec_inp.detach()




def channel_local_act(model, inp, channel, block_size,stride_size = 1, reference_func = torch.zeros_like, batch = 512):
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    if isinstance(block_size, int):
        block_size = (block_size,block_size)
    if isinstance(stride_size, int):
        stride_size = (stride_size,stride_size)
    B,C,H,W=inp.shape
    h,w = block_size
    dh,dw = stride_size
    end_x = H-h+1
    end_y = W-w+1
    perturbed = []
    masks = []
    inp = inp
    reference = reference_func(inp)
    sals = 0
    for i in  range(0,end_x, dh):
        for j in range(0,end_y,dw):
            mask = torch.zeros(B,1,H,W).to(device)
            mask[:,:,i:i+h,j:j+w] = 1
            perturbed.append(inp*mask + reference*(1-mask))
            masks.append(mask)
            if len(perturbed) == batch:
                perturbed = torch.cat(perturbed, 0)
                masks = torch.cat(masks, 0)
                with torch.no_grad():
                    #for i in range(0,N, batch):
                    weights = model(perturbed)[:,channel].sum(-1).sum(-1)
                    b = weights.shape[0]
                    weights = weights.view(1,b)
                    masks = masks.view(b, -1)
                    sal = torch.matmul(weights, masks).cpu()
                    sal = sal.view(-1,1,H,W)
                    sals+=sal
                perturbed = []
                masks = []
    if len(perturbed) > 0:
        perturbed = torch.cat(perturbed, 0)
        masks = torch.cat(masks, 0)
        with torch.no_grad():
            #for i in range(0,N, batch):
            weights = model(perturbed)[:,channel].sum(-1).sum(-1)
            b = weights.shape[0]
            weights = weights.view(1,b)
            masks = masks.view(b, -1)
            sal = torch.matmul(weights, masks).cpu()
            sal = sal.view(-1,1,H,W)
            sals+=sal
    return sals.to(device)
            #print(perturbed)
    #perturbed = torch.cat(perturbed, 0)
    #masks = torch.cat(masks, 0)
    #N = perturbed.shape[0]
    #with torch.no_grad():
    #    for i in range(0,N, batch):
    #        weights = model(perturbed[i:min(i+batch,N)].to(device))[:,channel].sum(-1).sum(-1)
    #sal = torch.matmul(weights.transpose(), masks.to(device))












def channel_attr(model, inp, target, img, area_mode = "ensemble", defense_mode = "IBM", shape = None, lr = 0.01, epochs = 100, beta = 1e-2, reference_func = torch.zeros_like,  binary = False, threshold = None, visualize = False, save_dir = "./result", restart = False, area_as_ratio = False, areas = None):

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    for param in model.parameters():
        param.requires_grad = False

    wrapped_model = Wrapper(model, defense_mode)

    #wrapped_model.pre_defense(inp)

    #with torch.no_grad():
    upsample = torch.nn.Upsample(size =shape[-2:], mode = "bilinear")
    if areas is None:
        areas = [5,10,15,20,25]
    for area in areas:
        attr = torch.zeros((1,inp.shape[1],1,1)).to(device)
        attr = generate_attr(wrapped_model, inp, target, area, attr, mode = "hybrid", defense_mode = defense_mode, shape = shape, lr = lr, epochs = epochs, beta = beta, reference_func = reference_func,  binary = binary, threshold = threshold, area_regulation = True, area_as_ratio = False)
        print(torch.argwhere(attr[0].int() == 1).squeeze())
        if visualize:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            new_attr = torch.mul(inp, attr).sum(1,keepdim =True)
            new_attr = upsample(new_attr)
            print(new_attr.shape)
            utils.visualize_attr(img, new_attr, "area_{}".format(area), save_dir)

    #wrapped_model.remove_hook()
    return attr


def channel_visualize(model, inp, target, area_mode = "ensemble", defense_mode = "IBM", shape = None, lr = 0.01, epochs = 100, beta = 1e-2, reference_func = torch.zeros_like,  binary = False, threshold = None, visualize = False, save_dir = "./result", restart = False, area_as_ratio = True):

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    for param in model.parameters():
        param.requires_grad = False

    wrapped_model = Wrapper(model, defense_mode)

    wrapped_model.pre_defense(inp, end = wrapped_model.mid-1)

    #with torch.no_grad():

    attr = torch.ones(1,1,224,224).to(device)
    attr = generate_attr(wrapped_model, inp, target, 0, attr, defense_mode = defense_mode, shape = shape, lr = lr, epochs = epochs, beta = beta, reference_func = reference_func,  binary = binary, threshold = threshold, area_regulation = False)
    if visualize:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        utils.visualize_attr(inp, attr, "area_none", save_dir)

    wrapped_model.remove_hook()
    return attr




def channel_visualize_ensemble(model, inp, target, area_mode = "ensemble", defense_mode = "IBM", shape = None, lr = 0.01, epochs = 100, beta = 1e-2, reference_func = torch.zeros_like,  binary = False, threshold = None, visualize = False, save_dir = "./result", restart = False, area_as_ratio = True, areas = None, mse = False):

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    for param in model.parameters():
        param.requires_grad = False

    wrapped_model = Wrapper(model, defense_mode)

    #wrapped_model.pre_defense(inp, end = 29)
    wrapped_model.pre_defense(inp)

    #with torch.no_grad():
    if areas is None:
        areas = [0.1, 0.3, 0.5, 0.7, 0.9]
    attr = torch.ones(1,1,224,224).to(device)
    cur_attr = torch.ones(1,1,224,224).to(device)
    for area in areas:
        cur_attr = generate_attr(wrapped_model, inp, target, area, attr=cur_attr, defense_mode = defense_mode, shape = shape, lr = lr, epochs = epochs, beta = beta, reference_func = reference_func,  binary = binary, threshold = threshold, mse = mse)
        if visualize:
            utils.visualize_attr(inp, cur_attr, "ensemble_{}".format(str(area)), save_dir)
        attr += cur_attr
    attr = attr/5
    #attr = generate_attr(img, wrapped_model, 0, config, cur_attr)
    if visualize:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        utils.visualize_attr(inp, cur_attr, "ensemble_sum", save_dir)

    wrapped_model.remove_hook()
    return attr






def explain_fractiles(model, inp, target, mode = "preservation", area_mode = "ensemble", defense_mode = "IBM", shape = None, lr = 0.01, epochs = 100, beta = 1e-2, reference_func = get_random_reference,  binary = False, threshold = None, visualize = False, save_dir = "./result", restart = False, areas = None):

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    for param in model.parameters():
        param.requires_grad = False

    wrapped_model = Wrapper(model, defense_mode)

    wrapped_model.pre_defense(inp)

    attr = torch.zeros(1,1,224,224).to(device)
    attr = generate_attr_all(wrapped_model, inp, target, attr, mode = mode, defense_mode = defense_mode, shape = shape, lr = lr, epochs = epochs, beta = beta, reference_func = reference_func)
    if visualize:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        n_ele = torch.numel(attr[0])
        attr_flat = attr.view(attr.shape[0],-1)

        for area in [0.1,0.3,0.5,0.7,0.9]:
            n = int(area*n_ele)
            threshold = attr_flat.sort(1)[0][:,-n]
            mask = torch.where(attr >= threshold, 1, 0).float()
            utils.visualize_attr(inp, mask, "mask_{}".format(area), save_dir)
        

        utils.visualize_attr(inp, attr, "fractiles", save_dir)


    wrapped_model.remove_hook()
    return attr






def explain_channel_ind(model, inp, target, channel_attr, area_mode = "l1", defense_mode = "NONE", shape = None, lr = 0.01, epochs = 100, beta = 1e-6, reference_func = torch.zeros_like,  binary = False, threshold = None, visualize = False, save_dir = "./result", restart = False, areas = None):

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    for param in model.parameters():
        param.requires_grad = False

    wrapped_model = Wrapper(model, defense_mode)

    wrapped_model.pre_defense(inp)


    if area_mode == "ensemble":
        if areas is None:
            areas = [0.1, 0.3, 0.5, 0.7, 0.9]
        attr = torch.zeros(1,1,224,224).to(device)
        cur_attr = torch.rand(1,1,224,224).to(device)
        for area in areas:
            if restart:
                cur_attr = generate_attr_channel_ind(wrapped_model, inp, target, area, chanenl_attr, attr = torch.zeros_like(cur_attr), defense_mode = defense_mode, shape = shape, lr = lr, epochs = epochs, beta = beta, reference_func = reference_func, binary = binary, threshold = threshold)
            else:
                cur_attr = generate_attr_channel_ind(wrapped_model, inp, target, area, channel_attr, attr = cur_attr, defense_mode = defense_mode, shape = shape, lr = lr, epochs = epochs, beta = beta, reference_func = reference_func, binary = binary, threshold = threshold)
            if visualize:
                utils.visualize_attr(inp, cur_attr, "ensemble_{}".format(str(area)), save_dir)
            attr += cur_attr
        attr = attr/5
        #attr = generate_attr(img, wrapped_model, 0, config, cur_attr)
        if visualize:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            utils.visualize_attr(inp, cur_attr, "ensemble_sum", save_dir)
    elif area_mode == "l1":
        #attr = torch.zeros(1,1,224,224).to(device)
        attr = generate_attr_channel_ind(wrapped_model, inp, target, 0,channel_attr, attr = None, defense_mode = defense_mode, shape = shape, lr = lr, epochs = epochs, beta = beta, reference_func = reference_func,  binary = binary, threshold = threshold)
        if visualize:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            utils.visualize_attr(inp, attr, "l1", save_dir)
    elif area_mode == "none":
        attr = torch.zeros(1,1,224,224).to(device)
        attr = generate_attr_channel_ind(wrapped_model, inp, target, 0, channel_attr, attr, defense_mode = defense_mode, shape = shape, lr = lr, epochs = epochs, beta = beta, reference_func = reference_func,  binary = binary, threshold = threshold, area_regulation = False)
        if visualize:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            utils.visualize_attr(inp, attr, "ensemble_none", save_dir)


    wrapped_model.remove_hook()
    return attr






    



    








def explain(model, inp, target, area_mode = "ensemble", defense_mode = "IBM", shape = None, lr = 0.01, epochs = 100, beta = 1e-2, reference_func = get_random_reference,  binary = False, threshold = None, visualize = False, save_dir = "./result", restart = False, areas = None):

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    for param in model.parameters():
        param.requires_grad = False

    wrapped_model = Wrapper(model, defense_mode)

    wrapped_model.pre_defense(inp)


    if area_mode == "ensemble":
        if areas is None:
            areas = [0.1, 0.3, 0.5, 0.7, 0.9]
        attr = torch.zeros(1,1,224,224).to(device)
        cur_attr = torch.rand(1,1,224,224).to(device)
        for area in areas:
            if restart:
                cur_attr = generate_attr(wrapped_model, inp, target, area, attr = torch.zeros_like(cur_attr), defense_mode = defense_mode, shape = shape, lr = lr, epochs = epochs, beta = beta, reference_func = reference_func, binary = binary, threshold = threshold)
            else:
                cur_attr = generate_attr(wrapped_model, inp, target, area, attr = cur_attr, defense_mode = defense_mode, shape = shape, lr = lr, epochs = epochs, beta = beta, reference_func = reference_func, binary = binary, threshold = threshold)
            if visualize:
                utils.visualize_attr(inp, cur_attr, "ensemble_{}".format(str(area)), save_dir)
            attr += cur_attr
        attr = attr/5
        #attr = generate_attr(img, wrapped_model, 0, config, cur_attr)
        if visualize:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            utils.visualize_attr(inp, cur_attr, "ensemble_sum", save_dir)
    elif area_mode == "l1":
        attr = torch.zeros(1,1,224,224).to(device)
        attr = generate_attr(wrapped_model, inp, target, 0, attr, defense_mode = defense_mode, shape = shape, lr = lr, epochs = epochs, beta = beta, reference_func = reference_func,  binary = binary, threshold = threshold)
        if visualize:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            utils.visualize_attr(inp, attr, "l1", save_dir)
    elif area_mode == "none":
        attr = torch.zeros(1,1,224,224).to(device)
        attr = generate_attr(wrapped_model, inp, target, 0, attr, defense_mode = defense_mode, shape = shape, lr = lr, epochs = epochs, beta = beta, reference_func = reference_func,  binary = binary, threshold = threshold, area_regulation = False)
        if visualize:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            utils.visualize_attr(inp, attr, "ensemble_none", save_dir)


    wrapped_model.remove_hook()
    return attr






    
