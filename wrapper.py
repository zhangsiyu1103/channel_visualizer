import torch
import torch.nn as nn
import torch.nn.functional as F
import math




MID_LAYER={
        'vgg19' : 36,
        'vgg16' : 30,
        'alexnet' : 12,
        'googlenet' : 183,
        'resnet18' : 49,
        'resnet34' : 89,
        'resnet50': 123,
        'resnet101': 242
        }



def get_seq_model(layers):
    cut_point = None
    for i in range(len(layers)):
        if isinstance(layers[i], nn.Linear):
            cut_point = i
            break


    class CriticalModel(nn.Module):
        def __init__(self):
            super(CriticalModel, self).__init__()
            self.features = nn.Sequential(*layers[:cut_point])
            if cut_point is not None:
                self.classifiers = nn.Sequential(*layers[cut_point:])

        def forward(self,x):
            x= self.features(x)
            if hasattr(self, "classifiers"):
                x = x.reshape(x.size(0), -1)
                x = self.classifiers(x)
            return x
    return CriticalModel()

def get_reg_layer(act):
    class RegLayer(nn.Module):
        def __init__(self):
            super(RegLayer, self).__init__()
            #self.act = act > 0
            self.act = act

        def forward(self,x):
            bu = torch.clamp(self.act, min = 0)
            bl = torch.clamp(self.act, max = 0)
            ret = torch.clamp(x, min=bl, max=bu)
            #ret = torch.mul(x, self.act)
            return ret
    return RegLayer()

class Wrapper(object):

    # Constructor from a torch model.

    def __init__(self,model, defense_mode = "none"):
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        self.model = model.to(self.device)
        self.model.eval()
        self.defense_mode = defense_mode.upper()
        self._layers=[]
        self._modules=[]
        self._layers_names=[]
        self._modules_names=[]
        self.weight_save = dict()
        self.bias_save = dict()
        self.const = False
        for  name,layer in model.named_modules():
            if len(list(layer.children()))==0:
                self._layers.append(layer)
                layer_str = str(layer)
                end_idx= layer_str.find("(")
                layer_str = layer_str[:end_idx]
                new_name = "{}: {}.{}".format(len(self._layers_names), name, layer_str)
                #self._layers_names.append(name)
                self._layers_names.append(new_name)
            else:
                self._modules.append(layer)
                layer_str = str(layer)
                end_idx= layer_str.find("(")
                layer_str = layer_str[:end_idx]
                new_name = "{}: {}.{}".format(len(self._modules_names), name, layer_str)
                self._modules_names.append(new_name)




    def layer_shape_hook(self,layer_idx):
        def hook(module,inp,out):
            self.layer_shapes[layer_idx] = out.shape
        return hook

    def module_shape_hook(self,layer_idx):
        def hook(module,inp,out):
            self.module_shapes[layer_idx] = out.shape
        return hook

    def init_layer_shape(self,inp):
        self.layer_shapes = [None for idx in range(len(self._layers))]
        self.model.eval()
        shape_hooks = []
        for i,l in enumerate(self._layers):
            shape_hooks.append(l.register_forward_hook(self.layer_shape_hook(i)))
        with torch.no_grad():
            self.model(inp)
        for hook in shape_hooks:
            hook.remove()

    def init_module_shape(self,inp):
        self.module_shapes = [None for idx in range(len(self._modules))]
        self.model.eval()
        shape_hooks = []
        for i,m in enumerate(self._modules):
            shape_hooks.append(m.register_forward_hook(self.module_shape_hook(i)))
        with torch.no_grad():
            self.model(inp)
        for hook in shape_hooks:
            hook.remove()




    def reorder_hook(self, idx):
        def hook(module, inp, out):
            self.order_indices.append(idx)
        return hook

    def reorder_layers(self, inp):
        self.order_layers = []
        self.order_layers_names = []
        self.order_indices = []
        hooks = []
        for i, l in enumerate(self._layers):
            hooks.append(l.register_forward_hook(self.reorder_hook(i)))

        with torch.no_grad():
            self.model(inp)
        for hook in hooks:
            hook.remove()

        for idx in self.order_indices:
            self.order_layers.append(self._layers[idx])
            self.order_layers_names.append(self._layers_names[idx])

    def act_hook(self):
        def hook(module,inp,out):
            #self.act_out.append(out.detach().cpu().clone())
            self.act_out.append(out.detach().clone())

        return hook


    def defense_forward_hook(self):
        def hook(module,inp,out):
            if self.act_idx == self.max_idx:
                self.act_idx = 0
            act = self.act_out[self.act_idx]
            if self.defense_mode == "IVM":
                clip = out <= act
            elif self.defense_mode == "AVM" or self.defense_mode == "VM":
                clip = out >= act
            elif self.defense_mode == "IBM":
                clip = act > 0
                clip = clip.detach()
            else:
                raise RuntimeError("defense mode not supported")
            if len(self.clips) == self.max_idx:
                self.clips[self.act_idx] = clip
            else:
                self.clips.append(clip)
            self.act_idx += 1

        return hook

    def defense_backward_hook(self):
        def hook(module,grad_inp,grad_out):
            if self.act_idx == 0:
                self.act_idx = self.max_idx
            self.act_idx -= 1
            #clip = self.clips[self.act_idx].to(self.device)
            clip = self.clips[self.act_idx]
            clip = clip.to(self.device)
            if self.defense_mode == "IBM" or self.defense_mode == "IVM":
                #print("relaxed")
                #clip = clip.to(self.device)
                cur_clip = grad_inp[0] > 0
                new_clip = torch.logical_or(cur_clip, clip)
                ret = (torch.mul(grad_inp[0], new_clip),)
                #print("relaxed")
            elif self.defense_mode == "AVM":
                #print("reverse")
                #clip = clip.to(self.device)
                cur_clip = grad_out[0] < 0
                new_clip = torch.logical_or(cur_clip, clip)
                ret = (torch.mul(grad_inp[0], new_clip),)
            elif self.defense_mode == "VM":
                cur_clip = grad_inp[0] > 0
                clip1 = torch.logical_and(cur_clip, clip)
                clip2 = torch.logical_not(torch.logical_or(cur_clip, clip))
                new_clip = torch.logical_or(clip1, clip2)
                ret = (torch.mul(grad_inp[0], new_clip),)
            return ret

        return hook

    def remove_bhooks(self):
        if hasattr(self, "bhooks"):
            for hook in self.bhooks:
                hook.remove()


        self.bhooks = []


    def defense(self):
        if self.defense_mode == "NONE":
            return
        if hasattr(self, "bhooks"):
            for hook in self.bhooks:
                hook.remove()


        self.bhooks = []
        #for i, l in enumerate(self._layers):
        for i in range(self.start, self.end):
            l = self._layers[i]
            if isinstance(l, nn.Linear):
                break
            if isinstance(l, nn.ReLU):
                self.bhooks.append(l.register_full_backward_hook(self.defense_backward_hook()))


        #return self.model


    def pre_defense(self, inp, start = None, end =None):
        if self.defense_mode == "NONE":
            return

        if start is None:
            start = 0
        if end is None:
            end = len(self._layers) 

        self.start = start
        self.end = end


        fhook = []
        self.act_out = []
        self.clips = []
        self.act_idx = 0
        inp = inp.to(self.device)
        for i in range(start, end):
            l = self._layers[i]
            if isinstance(l, nn.Linear):
                break
            if isinstance(l, nn.ReLU):
                l.inplace = False
                fhook.append(l.register_forward_hook(self.act_hook()))

        with torch.no_grad():
            out = self.model(inp)
        for hook in fhook:
            hook.remove()
        self.max_idx = len(self.act_out)

        self.fhooks = []
        for i in range(start, end):
            l = self._layers[i]
            if isinstance(l, nn.Linear):
                break
            if isinstance(l, nn.ReLU):
                self.fhooks.append(l.register_forward_hook(self.defense_forward_hook()))


    def remove_hook(self):
        self.mid_out = []
        self.clips = []
        self.act_idx = 0
        self.max_idx = None
        if hasattr(self, "fhooks"):
            for hook in self.fhooks:
                hook.remove()
        if hasattr(self, "bhooks"):
            for hook in self.bhooks:
                hook.remove()
        self.fhooks = []
        self.bhooks = []


    def sanity_random(self, idx):
        self.weight_save = dict()
        self.bias_save = dict()
        m = self._layers[idx]
        self.weight_save[idx] = m.weight.clone()
        if m.bias is not None:
            self.bias_save[idx] = m.bias.clone()
        m.reset_parameters()

    def model_recover(self):
        with torch.no_grad():
            for k,v in self.weight_save.items():
                cur_layer = self._layers[k]
                cur_layer.weight.copy_(v)
                if k in self.bias_save.keys():
                    cur_layer.bias.copy_(self.bias_save[k])

        self.weight_save = dict()
        self.bias_save = dict()

    def init_submodel(self, model_name):

        self.model_name = model_name


        self.mid = MID_LAYER[self.model_name]
        if self.model_name == "resnet18":
            self.model_layers = self._layers[:4] + [self._modules[1],self._modules[4], self._modules[8], self._modules[12]] + self._layers[self.mid+1:]
            self.model_cut = {self._modules_names[1]:5,self._modules_names[4]:6, self._modules_names[8]:7, self._modules_names[12]:8}
        elif self.model_name == "resnet34":
            self.model_layers = self._layers[:4] + [self._modules[1],self._modules[5], self._modules[11], self._modules[19]] + self._layers[self.mid+1:]
            self.model_cut = {self._modules_names[1]:5,self._modules_names[5]:6, self._modules_names[11]:7, self._modules_names[19]:8}
        elif self.model_name == "resnet50":
            self.model_layers = self._layers[:4] + [self._modules[1],self._modules[6], self._modules[12], self._modules[20]] + self._layers[self.mid+1:]
            self.model_cut = {self._modules_names[1]:5,self._modules_names[6]:6, self._modules_names[12]:7, self._modules_names[20]:8}
        elif self.model_name == "resnet101":
            self.model_layers = self._layers[:4] + [self._modules[1],self._modules[6], self._modules[12], self._modules[37]] +self._layers[self.mid+1:]
            self.model_cut = {self._modules_names[1]:5,self._modules_names[6]:6, self._modules_names[12]:7, self._modules_names[37]:8}
        elif self.model_name == "vgg16":
            self.model_layers = self._layers
            print(self._layers_names)
            self.model_cut = dict()
            for k in [4,9,16,23,30]:
                self.model_cut[self._layers_names[k]] = k
        elif self.model_name == "vgg19":
            self.model_layers = self._layers
            print(self._layers_names)
            self.model_cut = dict()
            for k in [4,9,18,27,36]:
                self.model_cut[self._layers_names[k]] = k


    def divide(self, idx):
        sub_model1= get_seq_model(self.model_layers[:idx + 1])

        sub_model2 =  get_seq_model(self.model_layers[idx + 1:])


        return sub_model1, sub_model2
    #def divide(self, model_name):
    #    #first half of model
    #    self.model_name = model_name
    #    self.mid = MID_LAYER[self.model_name]

    #    if self.model_name == "resnet18":
    #        layers = self._layers[:4] + [self._modules[1],self._modules[4], self._modules[8], self._modules[12]]
    #    elif self.model_name == "resnet34":
    #        layers = self._layers[:4] + [self._modules[1],self._modules[5], self._modules[11], self._modules[19]]
    #    elif self.model_name == "resnet50":
    #        layers = self._layers[:4] + [self._modules[1],self._modules[6], self._modules[12], self._modules[20]]
    #    elif self.model_name == "resnet101":
    #        layers = self._layers[:4] + [self._modules[1],self._modules[6], self._modules[12], self._modules[37]]
    #    elif self.model_name == "googlenet":
    #        layers = [self._modules[1], self._layers[3], self._modules[2],self._modules[3],self._layers[10], self._modules[4], self._modules[14], self._layers[49], self._modules[24], self._modules[34], self._modules[44], self._modules[54], self._modules[64], self._layers[145],self._modules[74], self._modules[84]]
    #    else:
    #        layers = self._layers[:self.mid+1]

    #    sub_model1= get_seq_model(layers)

    #    layers = self._layers[self.mid+1:]
    #    sub_model2 =  get_seq_model(layers)


    #    return sub_model1, sub_model2



    def forward_maxpool_hook(self):
        def hook(module,inp,out):
            module.return_indices = True
            out,indices = module.forward(inp[0])
            self.maxpool_indices.append(indices)
            module.return_indices = False
        return hook


    def maxpool_forward(self, inp):

        self.maxpool_indices = []

        idx = 0
        hooks = []
        for l in (self._layers):
            if isinstance(l, nn.MaxPool2d):
                hooks.append(l.register_forward_hook(self.forward_maxpool_hook()))

        with torch.no_grad():
            self.model(inp)
        for hook in hooks:
            hook.remove()

    def backward_maxpool_hook(self):
        def hook(module,grad_inp,grad_out):

            if self.maxpool_idx == -1:
                self.maxpool_idx = self.maxpool_max_idx
            indices = self.maxpool_indices[self.maxpool_idx]
            self.maxpool_idx -= 1

            with torch.no_grad():
                ret = F.max_unpool2d(grad_out[0], indices, module.kernel_size, module.stride, module.padding, output_size = grad_inp[0].size())
            return (ret,)
        return hook


    def maxpool_fix(self):

        if hasattr(self, "maxpool_hooks"):
            for hook in self.maxpool_hooks:
                hook.remove()
        self.maxpool_hooks = []
        for l in (self._layers):
            if isinstance(l, nn.MaxPool2d):
                self.maxpool_hooks.append(l.register_full_backward_hook(self.backward_maxpool_hook()))


    def remove_maxpool_hook(self):
        if hasattr(self, "maxpool_hooks"):
            for hook in self.maxpool_hooks:
                hook.remove()
        self.maxpool_hooks = []

    def set_cur_idx_fix(self, layer=None):
        if layer is None:
            layer = len(self.order_layers)-1
        maxpool_idx = -1
        relu_idx = -1
        for i in (range(layer+1)):
            l = self.order_layers[i]
            if isinstance(l, nn.MaxPool2d):
                maxpool_idx += 1
            if isinstance(l, nn.ReLU):
                relu_idx += 1
        self.maxpool_idx = maxpool_idx
        self.maxpool_max_idx = maxpool_idx
        self.relu_idx = relu_idx
        self.relu_max_idx = relu_idx
        #print("max", self.maxpool_idx)
        #print("relu", self.relu_idx)

    def maxpool_remove_hook(self, idx):
        def hook(module, inp, out):
            avg = nn.AvgPool2d(kernel_size = module.kernel_size,stride=module.stride,padding=module.padding)
            new_out = avg(inp[0])
            return out
        layer_idxs = set(self.order_indices[:(idx+1)])
        self.maxpool_remove_hooks = []
        for i in layer_idxs:
            l = self._layers[i]
            if isinstance(l, nn.MaxPool2d):
                self.maxpool_remove_hooks.append(l.register_forward_hook(hook))

    def remove_maxpool_remove_hook(self):
        for hook in self.maxpool_remove_hooks:
            hook.remove()
        self.maxpool_remove_hooks = []




    def relu_remove_hook(self, idx=None):
        if idx is None:
            idx = len(self.order_layers)-1
        def hook(module, grad_inp, grad_out):
            #clip = (torch.randn_like(grad_out[0])>0)
            #return (torch.mul(grad_out[0],clip),)
            return grad_out
        layer_idxs = set(self.order_indices[:(idx+1)])
        self.relu_remove_hooks = []
        for i in layer_idxs:
            l = self._layers[i]
            if isinstance(l, nn.ReLU):
                l.inplace = False
                self.relu_remove_hooks.append(l.register_full_backward_hook(hook))

    def remove_relu_remove_hook(self):
        for hook in self.relu_remove_hooks:
            hook.remove()
        self.relu_remove_hooks = []

    def forward_relu_avg_hook(self):
        def hook(module,inp,out):
            out = out.prod(dim=0,keepdim = True)
            self.relu_clips.append((out>0).float().detach())
        return hook


    def relu_forward_avg(self, inp):
        self.relu_clips = []
        idx = 0
        hooks = []
        #for i in range(len(self._layers)):
        for l in self._layers:
            if isinstance(l, nn.ReLU):
                l.inplace = False
                hooks.append(l.register_forward_hook(self.forward_relu_avg_hook()))

        with torch.no_grad():
            self.model(inp)
        for hook in hooks:
            hook.remove()


    def forward_relu_hook(self):
        def hook(module,inp,out):
            self.relu_clips.append((out.clone()>0).float().detach())
        return hook


    def relu_forward(self, inp):
        self.relu_clips = []
        idx = 0
        hooks = []
        #for i in range(len(self._layers)):
        for l in self._layers:
            if isinstance(l, nn.ReLU):
                l.inplace = False
                hooks.append(l.register_forward_hook(self.forward_relu_hook()))

        with torch.no_grad():
            self.model(inp)
        for hook in hooks:
            hook.remove()

    def backward_relu_hook(self):
        def hook(module,grad_inp,grad_out):
            if self.relu_idx == -1:
                self.relu_idx = self.relu_max_idx
            clip = self.relu_clips[self.relu_idx]
            #print(torch.count_nonzero(clip))
            self.relu_idx -= 1
            #print(self.relu_idx)
            #return grad_out
            ret = torch.mul(clip, grad_out[0])
            ret = (ret,)
            return ret
        return hook

    def relu_fix(self, end = None):

        if hasattr(self, "relu_hooks"):
            for hook in self.relu_hooks:
                hook.remove()
        self.relu_hooks = []
        for l in self._layers:
            if isinstance(l, nn.ReLU):
                self.relu_hooks.append(l.register_full_backward_hook(self.backward_relu_hook()))

    def remove_relu_hook(self):
        if hasattr(self, "relu_hooks"):
            for hook in self.relu_hooks:
                hook.remove()
        self.relu_hooks = []


    def mid_pre_hook(self, layer):
        def hook(module,inp):
            self.mid_module = module
            self.mid_in = inp
        return hook

    def get_mid_out(self, inp, layer):
        hooks = []
        hook = layer.register_forward_pre_hook(self.mid_pre_hook(layer))
        self.model(inp)
        #for hook in hooks:
        hook.remove()
        mid = self.mid_module(self.mid_in[0])
        return mid


    def set_const(self, start = 0, end = -1):
        layer_len = len(self.order_layers)
        start = start % layer_len
        end = end % layer_len
        self.weight_save = dict()
        self.bias_save = dict()
        indices = set(self.order_indices[start:end])
        #for idx, m in enumerate(self._layers):
        for idx in indices: 
            #print(self._layers_names[idx])
            m = self._layers[idx]
            #for n,p in m.named_parameters():
            #    print(n)
            if not hasattr(m, "weight"):
                continue
            self.weight_save[idx] = m.weight.clone()
            if m.bias is not None:
                self.bias_save[idx] = m.bias.clone()
            if isinstance(m, nn.Conv2d):
                nn.init.constant_(m.weight, 1e-4)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                #m.running_mean = torch.zeros_like(m.running_mean)
                #m.running_var = torch.ones_like(m.running_var)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1e-4)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            else:
                raise Exception("module not defined")
        self.const = True


    def sanity_random(self,layer_idx):
        m = self._layers[layer_idx]
        print(m._forward_hooks)
        print(m._backward_hooks)
        #self.weight_save[layer_idx] = m.weight

        if isinstance(m, nn.Conv2d):
            m.reset_parameters()
            nn.init.constant_(m.weight,1)
            #nn.init.kaiming_uniform_(m.weight, a = math.sqrt(5))
        #if m.bias is not None:
            #nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            m.reset_parameters()
            nn.init.constant_(m.weight,1)
            #nn.init.kaiming_uniform_(m.weight, a = math.sqrt(5))
            #nn.init.kaiming_uniorm_(m.weight, 1)
            #if m.bias is not None:
            #    nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            m.reset_parameters()
            nn.init.constant_(m.weight,1)
            #nn.init.kaiming_uniform_(m.weight, a = math.sqrt(5))
            #nn.init.kaiming_uniorm_(m.weight, a = math.s)
            #if m.bias is not None:
            #    nn.init.constant_(m.bias, 0)


    def model_recover(self):
        #print("recover")
        #print(self.weight_save.keys())
        with torch.no_grad():
            for k,v in self.weight_save.items():
                cur_layer = self._layers[k]
                cur_layer.weight.copy_(v)
                #if k == 0:
                    #print(v)
                    #print(cur_layer.weight)
                if k in self.bias_save.keys():
                    cur_layer.bias.copy_(self.bias_save[k])

        self.weight_save = dict()
        self.bias_save = dict()
        self.const = False



