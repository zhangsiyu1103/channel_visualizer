import numpy as np
import torch
import torch.nn as nn
from skimage.transform import resize
from tqdm import tqdm

class RISE_SWEEP(nn.Module):
    def __init__(self, model, input_size, gpu_batch=200):
        super(RISE_SWEEP, self).__init__()
        self.model = model
        self.input_size = input_size
        self.gpu_batch = gpu_batch

    def generate_masks(self, N, p1, s_min=3,s_max=40, savepath='masks.npy'):
        #cell_size = np.ceil(np.array(self.input_size) / s)
        #up_size = (s + 1) * cell_size

        #grid = np.random.rand(N, s, s) < p1
        #grid = grid.astype('float32')

        self.N =N
        self.masks = np.zeros((N, 1,*self.input_size))
        for i in range(5):
            s = np.random.randint(s_min,s_max,size = (N))
            hs = np.random.randint(0,self.input_size[0],size = (N))
            ws = np.random.randint(0,self.input_size[1],size = (N))
            for i in range(N):
                self.masks[i,0,slice(hs[i],hs[i]+s[i]),slice(ws[i],ws[i]+s[i])]=1
        
        self.masks = torch.from_numpy(self.masks).float()
        #self.masks = self.masks.reshape(-1, 1, *self.input_size)
        self.masks = self.masks.cuda()
        #self.masks = np.empty((N, *self.input_size))

        #print(p1)
        #print(s_min)
        #print(s_max)

        #for i in tqdm(range(N), desc='Generating filters'):
        #    #s = np.random.randint(s_min,s_max)
        #    #cell_size = np.ceil(np.array(self.input_size) / s)
        #    #up_size = (s + 1) * cell_size
        #    cell_size = np.random.randint(s_min,s_max)
        #    cell_size = np.array([cell_size, cell_size])
        #    s = np.ceil(self.input_size[0] / cell_size[0]).astype("int")

        #    up_size = (s + 1) * cell_size
        #    #print(s)
        #    grid = np.random.rand(s, s) < p1
        #    grid = grid.astype('float32')

        #    # Random shifts
        #    x = np.random.randint(0, cell_size[0])
        #    y = np.random.randint(0, cell_size[1])
        #    # Linear upsampling and cropping
        #    self.masks[i, :, :] = resize(grid, up_size, order=1, mode='reflect',
        #                                 anti_aliasing=False)[x:x + self.input_size[0], y:y + self.input_size[1]]
        #self.masks = self.masks.reshape(-1, 1, *self.input_size)
        #np.save(savepath, self.masks)
        #self.masks = torch.from_numpy(self.masks).float()
        #self.masks = self.masks.cuda()
        #self.N = N
        #self.p1 = p1

    def load_masks(self, filepath):
        self.masks = np.load(filepath)
        self.masks = torch.from_numpy(self.masks).float().cuda()
        self.N = self.masks.shape[0]


    def forward(self, x, target):
        if isinstance(target, int):
            target = [target]
        N = self.N
        _, _, H, W = x.size()
        # Apply array of filters to the image
        stack = torch.mul(self.masks, x.data)
        # p = nn.Softmax(dim=1)(model(stack)) processed in batches
        p = []
        for i in range(0, N, self.gpu_batch):
            #print(self.model(stack[i:min(i + self.gpu_batch, N)]).shape)
            #p.append(self.model(stack[i:min(i + self.gpu_batch, N)].cuda())[:,target].sum(-1).sum(-1).sum(-1))
            #p.append(self.model(stack[i:min(i + self.gpu_batch, N)].cuda())[:,target].sum(-1).sum(-1).sum(-1))
            #p.append(self.model(stack[i:min(i+self.gpu_batch, N)].cuda())[:,target].sum(-1).sum(-1).cpu())
            with torch.no_grad():
                p.append(self.model(stack[i:min(i+self.gpu_batch, N)].cuda())[:,target].sum(-1).sum(-1))
            #p.append(torch.randn(100,1).cuda())
            #p.append(res)
        p = torch.cat(p).cuda()
        #p = p.
        #p = torch.stack(p)
        # Number of classes
        CL = p.size(1)
        #CL = 1
        self.masks.cuda()
        sal = torch.matmul(p.data.transpose(0, 1), self.masks.view(N, H * W))
        #sal = torch.matmul(p.data, self.masks.view(N, H * W))
        #sal = torch.mul(p.data, self.masks)
        sal = sal.view((CL,1, H, W))
        mask_sum = torch.sum(self.masks, 0,  keepdim = True)
        print(mask_sum.shape)
        print(sal.shape)
        sal = sal/mask_sum
        self.masks.cpu()
        p.cpu()
        #sal = sal / N / self.p1
        return sal
    

class RISE(nn.Module):
    def __init__(self, model, input_size, gpu_batch=200):
        super(RISE, self).__init__()
        self.model = model
        self.input_size = input_size
        self.gpu_batch = gpu_batch

    def generate_masks(self, N, s, p1, savepath='masks.npy'):
        cell_size = np.ceil(np.array(self.input_size) / s)
        up_size = (s + 1) * cell_size

        grid = np.random.rand(N, s, s) < p1
        grid = grid.astype('float32')

        self.masks = np.empty((N, *self.input_size))

        for i in tqdm(range(N), desc='Generating filters'):
            # Random shifts
            x = np.random.randint(0, cell_size[0])
            y = np.random.randint(0, cell_size[1])
            # Linear upsampling and cropping
            self.masks[i, :, :] = resize(grid[i], up_size, order=1, mode='reflect',
                                         anti_aliasing=False)[x:x + self.input_size[0], y:y + self.input_size[1]]
        self.masks = self.masks.reshape(-1, 1, *self.input_size)
        np.save(savepath, self.masks)
        self.masks = torch.from_numpy(self.masks).float()
        self.masks = self.masks.cuda()
        self.N = N
        self.p1 = p1

    def load_masks(self, filepath):
        self.masks = np.load(filepath)
        self.masks = torch.from_numpy(self.masks).float().cuda()
        self.N = self.masks.shape[0]


    def forward(self, x, target):
        if isinstance(target, int):
            target = [target]
        N = self.N
        _, _, H, W = x.size()
        # Apply array of filters to the image
        stack = torch.mul(self.masks, x.data)
        # p = nn.Softmax(dim=1)(model(stack)) processed in batches
        p = []
        for i in range(0, N, self.gpu_batch):
            #print(self.model(stack[i:min(i + self.gpu_batch, N)]).shape)
            #p.append(self.model(stack[i:min(i + self.gpu_batch, N)].cuda())[:,target].sum(-1).sum(-1).sum(-1))
            #p.append(self.model(stack[i:min(i + self.gpu_batch, N)].cuda())[:,target].sum(-1).sum(-1).sum(-1))
            #p.append(self.model(stack[i:min(i+self.gpu_batch, N)].cuda())[:,target].sum(-1).sum(-1).cpu())
            with torch.no_grad():
                p.append(self.model(stack[i:min(i+self.gpu_batch, N)].cuda())[:,target].sum(-1).sum(-1))
            #p.append(torch.randn(100,1).cuda())
            #p.append(res)
        p = torch.cat(p).cuda()
        #p = p.
        #p = torch.stack(p)
        # Number of classes
        CL = p.size(1)
        #CL = 1
        sal = torch.matmul(p.data.transpose(0, 1), self.masks.view(N, H * W))
        #sal = torch.matmul(p.data, self.masks.view(N, H * W))
        #sal = torch.mul(p.data, self.masks)
        sal = sal.view((CL,1, H, W))
        sal = sal / N / self.p1
        return sal
    
    
class RISEBatch(RISE):
    def forward(self, x, target):
        # Apply array of filters to the image
        N = self.N
        B, C, H, W = x.size()
        stack = torch.mul(self.masks.view(N, 1, H, W), x.data.view(B * C, H, W))
        stack = stack.view(B * N, C, H, W)
        stack = stack

        #p = nn.Softmax(dim=1)(model(stack)) in batches
        p = []
        for i in range(0, N*B, self.gpu_batch):
            p.append(self.model(stack[i:min(i + self.gpu_batch, N*B)])[:,target].sum())
        p = torch.cat(p)
        CL = p.size(1)
        p = p.view(N, B, CL)
        sal = torch.matmul(p.permute(1, 2, 0), self.masks.view(N, H * W))
        sal = sal.view(B, CL, H, W)
        return sal

# To process in batches
# def explain_all_batch(data_loader, explainer):
#     n_batch = len(data_loader)
#     b_size = data_loader.batch_size
#     total = n_batch * b_size
#     # Get all predicted labels first
#     target = np.empty(total, 'int64')
#     for i, (imgs, _) in enumerate(tqdm(data_loader, total=n_batch, desc='Predicting labels')):
#         p, c = torch.max(nn.Softmax(1)(explainer.model(imgs.cuda())), dim=1)
#         target[i * b_size:(i + 1) * b_size] = c
#     image_size = imgs.shape[-2:]
#
#     # Get saliency maps for all images in val loader
#     explanations = np.empty((total, *image_size))
#     for i, (imgs, _) in enumerate(tqdm(data_loader, total=n_batch, desc='Explaining images')):
#         saliency_maps = explainer(imgs.cuda())
#         explanations[i * b_size:(i + 1) * b_size] = saliency_maps[
#             range(b_size), target[i * b_size:(i + 1) * b_size]].data.cpu().numpy()
#     return explanations
