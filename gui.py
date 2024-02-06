import tkinter as tk
import cv2
from PIL import Image, ImageTk, ImageGrab
from threading import Thread
from tkinter import ttk
from tkinter import filedialog
import os
import numpy as np
import torch

from wrapper import Wrapper
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from explain import *
from scipy.ndimage import gaussian_filter
#def display_roi(event):
#    global selected_photo
#
#    if selected_img:
#        selected_photo = ImageTk.PhotoImage(selected_img)
#        canvas2.create_image(0,0, anchor =tk.NW, image = selected_photo)
#        #cropped_lbl.config(image=tkimg)
#
#def select_roi():
#    global  selected_img
#    #img = cv2.imread("sample.JPEG")
#    roi  = cv2.selectROI(np.array(img))
#
#    imCrop = img[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
#    
#    if len(imCrop)>0:
#        selected_img = Image.fromarray(cv2.cvtColor(imCrop, cv2.COLOR_BGR2RGB))
#
#    cv2.destroyAllWindows()
#    root.event_generate("<<ROISELECTED>>")
#
#def start_thread():
#
#    thread = Thread(target=select_roi, daemon=True)
#    thread.start()
#
#def clearImage(canvas):
#    canvas.delete("all")
#    return

#def loadImage(canvas, img):
#    filename = 


#root = tk.Tk()

def get_class_name(c):
    labels = np.loadtxt('synset_words.txt', str, delimiter='\t')
    return ' '.join(labels[c].split(',')[0].split()[1:])

def tensor_to_numpy(img,  is_image = True):
    if isinstance(img, np.ndarray):
        return img

    if len(img.shape) == 3:
        img = img.unsqueeze(0)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    denormalize = transforms.Normalize(
            mean= [-m/float(s) for m, s in zip(mean, std)],
            std= [1.0/s for s in std]
    )
    if is_image:
        img = denormalize(img)

    return np.uint8(255*np.transpose(img.cpu().detach().numpy(), (0, 2, 3, 1))).squeeze()


def numpy_to_tensor(img, is_image = True):
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    if torch.is_tensor(img):
        return img

    if len(img.shape) == 3:
        img = img[None,:]

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    normalize = transforms.Normalize(mean = mean, std = std)
    new_tensor = torch.from_numpy(np.transpose(img, (0, 3, 1, 2))).float()
    if is_image:
        return normalize(new_tensor).to(device)
    else:
        return new_tensor.to(device)
class GUI():
    def __init__(self):
        #super(Root, self).__init__()
#root.title("Python Tkinter Dialog Widget")
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        self.window = tk.Tk()
        self.window.geometry("1200x850")
#root.minsize(700,450)

        
        self.dataframe = ttk.LabelFrame(self.window, text ="Load Data")
        self.dataframe.grid(column = 0, row = 0, padx = 10, pady =5,sticky = tk.W, columnspan = 3,rowspan =3)
        self.browseframe = ttk.LabelFrame(self.dataframe, text = "Open Image")
        self.browseframe.grid(column = 0, row = 0, padx = 10, pady = 10,sticky = tk.W, columnspan = 1)
        self.editframe = ttk.LabelFrame(self.window, text = "Edit Image")
        self.editframe.grid(column = 3, row = 1, padx = 30, pady = 0,sticky = tk.W, columnspan = 2)
        self.bgframe = ttk.LabelFrame(self.window, text = "Choose Background")
        self.bgframe.grid(column = 3, row = 2, padx = 10, pady = 0,sticky = tk.W, columnspan = 4)
        self.channelframe = ttk.LabelFrame(self.window, text = "Channel Edit")
        self.channelframe.grid(column = 6, row = 3, padx = 10, pady = 0,sticky = tk.W, columnspan = 3)

        self.categoryframe = ttk.LabelFrame(self.dataframe, text = "Load Category")
        self.categoryframe.grid(column = 0, row = 1, padx = 0, pady = 0,sticky = tk.W, columnspan = 2)

        self.cleardisplaybutton = ttk.Button(self.dataframe, text = "Clear Data",command = self.clear_display)
        self.cleardisplaybutton.grid(column = 2, row =1, padx = 20)


        self.img_canvas = tk.Canvas(self.window, width = 300, height = 300, bg = 'white')
        self.img_canvas.grid(column = 0,row = 3,padx = 10, pady = 10, sticky = tk.W, columnspan = 3, rowspan = 3)
        #self.select_canvas = tk.Canvas(self.window, width = 300, height = 300, bg = 'white')
        #self.select_canvas.grid(column = 1,row = 3,sticky = tk.W)
        self.channel_canvas = tk.Canvas(self.window, width = 300, height = 300, bg = 'white')
        self.channel_canvas.grid(column = 3,row = 3,padx = 10, pady = 10, sticky = tk.W, columnspan = 3, rowspan = 3)

        self.displayframe = tk.Frame(self.window, width = 660, height = 120)
        self.displayframe.grid(column = 0,row = 7,padx = 10, pady = 10, sticky = "wnes", columnspan = 8, rowspan = 1)
        self.display_canvas = tk.Canvas(self.displayframe, width = 660, height = 120, bg = 'white')
        self.display_canvas.grid(column = 0,row = 7, sticky = "wnes", columnspan = 8, rowspan = 1)

        self.displayscroll = tk.Scrollbar(self.displayframe, orient=tk.HORIZONTAL)
        self.displayscroll.grid(column = 0, row = 8, sticky = "wse", columnspan = 8)

        #self.display_canvas.config(xscrollcommand=self.displayscroll.set, scrollregion=self.display_canvas.bbox(tk.ALL))


        self.custom_category=tk.StringVar()
        self.Entrylabel = ttk.Label(self.categoryframe, text = 'Category')
        self.Entrylabel.grid(column = 1, row =0, padx = 2, sticky = tk.E)
        self.channelEntry = ttk.Entry(self.categoryframe, textvariable = self.custom_category, width = 5)
        self.channelEntry.grid(column = 2, row =0,padx = 2, sticky = tk.W)
        self.load_cat_button = ttk.Button(self.categoryframe, text = "Load",command = self.load_category)
        self.load_cat_button.grid(column = 3, row =0, padx = 2, sticky = tk.W)
        self.category_msg = ttk.Label(self.categoryframe)
        self.category_msg.grid(column = 1, row = 1, columnspan = 3)



#canvas.grid(column = 0, row = 3)
        self.browsebutton = ttk.Button(self.browseframe, text = "Select An Image",command = self.browse_and_display)
        self.browsebutton.grid(column = 0, row = 0, sticky = tk.W)
        self.refreshbutton = ttk.Button(self.editframe, text = "Reset Image",command = self.refresh)
        self.refreshbutton.grid(column = 3, row = 2, sticky = tk.W)
        self.selectbutton = ttk.Button(self.editframe, text="Select ROI", command=self.drawROI)
        self.selectbutton.grid(column = 4, row = 2, sticky = tk.W)
        #self.addbutton = ttk.Button(self.editframe, text="Add ROI", command=self.addROI)
        #self.addbutton.grid(column = 2, row = 2, sticky = tk.W)
        #self.undobutton = ttk.Button(self.editframe, text="Undo", command=self.undo)
        #self.undobutton.grid(column = 2, row = 2, sticky = tk.W)

        #rect = tkinter.Radiobutton(self.root, text = "Rectangle", variable =roishape, value = 0)
        #rect.grid(row = 1, column = 0, columnspan = 1)
        #circ = tkinter.Radiobutton(self.root, text = "Circle", variable = roishape, value = 1)
        #circ.grid(row = 1, column = 2, columnspan = 1)

        self.bg=tk.StringVar()
        self.bg.set("white")
        self.convertbutton = ttk.Button(self.bgframe, text="show background", command=self.show_bg)
        self.convertbutton.grid(row = 2,column = 3, padx = 5,sticky = tk.W)
        white = ttk.Radiobutton(self.bgframe, text = "White", variable = self.bg, value = "white", command = self.change_bg)
        white.grid(row = 2, column = 0, columnspan = 1)
        noise = ttk.Radiobutton(self.bgframe, text = "Noise", variable = self.bg, value = "noise", command = self.change_bg)
        noise.grid(row = 2, column = 1, columnspan = 1)
        gauss = ttk.Radiobutton(self.bgframe, text = "Gaussian", variable = self.bg, value = "gauss", command = self.change_bg)
        gauss.grid(row = 2, column = 2, columnspan = 1)
        

        self.modelframe = ttk.LabelFrame(self.window, text = "Choose Model")
        self.modelframe.grid(column = 3, row = 0, padx = 30, pady = 20,sticky = tk.W, columnspan = 1)
        self.modelcombo = ttk.Combobox(self.modelframe, state = "readonly", values = ["vgg16","vgg19","resnet18","resnet34"], width = 10)
        self.modelcombo.grid(column =3,row = 0,sticky = tk.W)
        self.modelcombo.bind("<<ComboboxSelected>>",lambda e: Thread(target=self.model_selection_change, args = (e,)).start())


        self.modelcutframe = ttk.LabelFrame(self.window, text = "Select model cut")
        self.modelcutframe.grid(column = 4, row = 0, padx = 10, pady = 20,sticky = tk.W, columnspan = 1)
        self.modelcutcombo = ttk.Combobox(self.modelcutframe, state = "readonly", values = [])
        self.modelcutcombo.grid(column =4,row = 0,sticky = tk.W)
        self.modelcutcombo.bind("<<ComboboxSelected>>",lambda e: Thread(target=self.model_divide_change, args = (e,)).start())


        self.datasetframe = ttk.LabelFrame(self.dataframe, text = "Load Dataset")
        self.datasetframe.grid(column = 1, row = 0, padx = 0, pady = 0,sticky = tk.W, columnspan = 2)
        self.databutton = ttk.Button(self.datasetframe, text = "Load ImageNet",command = lambda: Thread(target=self.load_dataset).start())
        self.databutton.grid(column = 1, row = 2, sticky = tk.W)
        #self.dataset_msg = ttk.Label(self.datasetframe)
        #self.dataset_msg.grid(column = 1, row = 3, columnspan = 1)
        self.datasettype=tk.StringVar()
        self.datasettype.set("val")
        train = ttk.Radiobutton(self.datasetframe, text = "Train", variable = self.datasettype, value = "train")
        train.grid(row = 1, column = 0, columnspan = 1)
        val = ttk.Radiobutton(self.datasetframe, text = "Val", variable = self.datasettype, value = "val")
        val.grid(row = 1, column = 1, columnspan = 1)
        self.load = tk.Label(self.datasetframe, text = "")
        self.load.grid(row = 3, column = 1, columnspan = 1)


        self.autochannelframe = ttk.LabelFrame(self.window, text = "Greedy Search of Salient Channel")
        self.autochannelframe.grid(column = 6, row = 0, padx = 0, pady = 20,sticky = tk.W, columnspan = 3, rowspan =2)
        self.searchmodeframe = ttk.LabelFrame(self.autochannelframe, text = "Search Mode")
        self.searchmodeframe.grid(column = 6, row = 1, padx = 10, pady = 0,sticky = tk.W, columnspan = 3)
        self.datamodeframe = ttk.LabelFrame(self.autochannelframe, text = "Data Mode")
        self.datamodeframe.grid(column = 6, row = 2, padx = 10, pady = 0,sticky = tk.W, columnspan = 3)
        self.preserved=tk.BooleanVar()
        self.preserved.set(True)
        pre_ = ttk.Radiobutton(self.searchmodeframe, text = "Preservation", variable = self.preserved, value = True)
        pre_.grid(row = 0, column = 6, columnspan = 1)
        del_ = ttk.Radiobutton(self.searchmodeframe, text = "deletetion", variable = self.preserved, value = False)
        del_.grid(row = 0, column = 7, padx = 10, columnspan = 1)

        self.ind = tk.BooleanVar()
        self.ind.set(True)
        ind_ = ttk.Radiobutton(self.datamodeframe, text = "Selected Image", variable = self.ind, value = True)
        ind_.grid(row = 2, column = 6, columnspan = 1)
        category_ = ttk.Radiobutton(self.datamodeframe, text = "Model Category", variable = self.ind, value = False)
        category_.grid(row = 2, column = 7, padx = 10, columnspan = 1)
        self.channelbutton = ttk.Button(self.autochannelframe, text="Compute Salient Channels", command = lambda : Thread(target=self.channel_attr).start())
        self.channelbutton.grid(column = 6, row = 4,pady = 10, columnspan = 3)
        self.spatialbutton = ttk.Button(self.autochannelframe, text="Update channel spatial", command = lambda : Thread(target=self.ind_attr).start())
        self.spatialbutton.grid(column = 6, row = 5,pady = 0, columnspan = 3)
        self.channel_msg = ttk.Label(self.autochannelframe, text="")
        self.channel_msg.grid(column = 6, row = 7, columnspan = 1)


        self.custom_channel=tk.IntVar()
        self.Entrylabel = ttk.Label(self.channelframe, text = 'Channel')
        self.Entrylabel.grid(column = 6, row =3, padx = 2, sticky = tk.E)
        self.channelEntry = ttk.Entry(self.channelframe, textvariable = self.custom_channel, width = 5)
        self.channelEntry.grid(column = 7, row =3,padx = 2, sticky = tk.W)
        self.addbutton = ttk.Button(self.channelframe, text = "Add",command = self.add_channel)
        self.addbutton.grid(column = 8, row =3, padx = 2, sticky = tk.W)
        self.clearbutton = ttk.Button(self.channelframe, text = "Clear",command = self.clear_channel)
        self.clearbutton.grid(column = 9, row =3, padx = 2, sticky = tk.W)
        self.add_channel_msg = ttk.Label(self.channelframe, text="")
        self.add_channel_msg.grid(column = 6, row = 5, columnspan = 1)


        self.listbox = tk.Listbox(self.window, width = 30, selectmode = "single")
        self.listbox.grid(column = 6, row = 4, padx = 10, sticky = "nwse", columnspan = 3, rowspan = 1)

        self.listbox.bind('<<ListboxSelect>>', self.channelonselect)

        self.listscroll = tk.Scrollbar(self.window, orient="vertical", command = self.listbox.yview)
        self.listbox.config(yscrollcommand = self.listscroll.set)
        self.listscroll.grid(column = 8, row = 4, sticky = "nes")



        self.block_var=tk.IntVar()
        self.block_var.set(3)
        self.localframe = ttk.LabelFrame(self.window, text = "Channel Local Visualization")
        self.localframe.grid(column = 7, row = 5, padx = 0, pady = 0,sticky = tk.W, columnspan = 2)

        self.blocklabel = ttk.Label(self.localframe, text = 'block size')
        self.blocklabel.grid(column = 7, row =6, padx = 2, sticky = tk.E)

        self.blockEntry = ttk.Entry(self.localframe, textvariable = self.block_var, width = 5)
        self.blockEntry.grid(column = 8, row =6,padx = 2, sticky = tk.W)

        self.localbutton = ttk.Button(self.localframe, text = "local visualization",command = lambda : Thread(target=self.channel_local).start())
        self.localbutton.grid(column = 9, row = 6, sticky = tk.W)
        self.local_msg = ttk.Label(self.localframe, text="")
        self.local_msg.grid(column = 7, row = 7, columnspan = 1)



        self.savebutton = ttk.Button(self.window, text = "Save",command = self.save)
        self.savebutton.grid(column = 8, row = 7, sticky = tk.W)

        self.target = tk.Label(self.modelframe, text = "")
        self.target.grid(row = 1, column = 3, stick = tk.W, columnspan = 2)

        self.mid_max = {}
        #if hasattr(self, "base_width"):
        #    del self.base_width
        #    del self.base_height
        self.fix_shape = False
        self.selected = False
        self.show = False
        self.dataset_loaded = False
        self.display_channel_ = False
        self.display_imgs = []
        self.model_loaded = False
        self.channel = None

        self.category_data = {"train":dict(), "val":dict()}

    def save(self):
        filename = filedialog.asksaveasfile(initialfile = "result.jpg", mode='w', title = "Save the file", defaultextension=".jpg")
        if not filename:
            return
        ImageGrab.grab().save(filename)




    def insert_display(self, image):
        #global cur_photo
        image_np = tensor_to_numpy(image)
        image = Image.fromarray(image_np)
        cur_photo = ImageTk.PhotoImage(image.resize((100,100)))
        n = len(self.display_imgs)
        self.display_imgs.append(cur_photo)
        image_label = tk.Label(self.display_canvas, image = self.display_imgs[-1])
        image_label.image = image
        image_label.bind('<Button-1>', self.show_img)
        #img = self.display_canvas.create_image(n*1+0.5, 0, anchor =tk.NW, image = cur_photo)
        self.display_canvas.create_window(n*110+10, 10, anchor =tk.NW, window = image_label)
        #img = self.display_canvas.create_image(n*110+10, 10, anchor =tk.NW, image = self.display_imgs[-1])
        #self.display_canvas.tag_bind(img,'<Button-1>',self.show_img)
        #self.display_canvas.config(width = ((n+1)*110 + 10))
        self.displayscroll.config(command = self.display_canvas.xview)
        self.display_canvas.config(height = 120, width = 660)
        self.display_canvas.config(xscrollcommand=self.displayscroll.set, scrollregion=self.display_canvas.bbox(tk.ALL))
        #img = self.display_canvas.create_image(0, 0 , anchor =tk.NW, image = photo)


    def model_divide_change(self,event):
        self.clear_channel()
        self.model_cut = self.modelcutcombo.get()
        
        self.submodel1, self.submodel2 = self.wrapped.divide(self.wrapped.model_cut[self.model_cut])

        if hasattr(self, "model_inp"):
            with torch.no_grad():
                self.mid = self.submodel1(self.model_inp)
                if not self.selected:
                    self.ori_mid = self.submodel1(self.ori_inp)
                    self.mid_max[self.model_cut] = self.ori_mid.max(-1)[0].max(-1)[0]+1
                self.max_channel =  self.mid.shape[1]
            if self.display_channel_:
                self.display_channel()

    def model_selection_change(self,event):
        self.model_name = self.modelcombo.get()

        if self.model_name == "vgg16":
            self.model = models.vgg16(pretrained=True)
        elif self.model_name == "vgg19":
            self.model = models.vgg19(pretrained=True)
        elif self.model_name == "resnet18":
            self.model = models.resnet18(pretrained=True)
        elif self.model_name == "resnet34":
            self.model = models.resnet34(pretrained=True)
        elif self.model_name == "resnet50":
            self.model = models.resnet50(pretrained=True)
        self.model.to(self.device)
        self.wrapped = Wrapper(self.model)
        self.wrapped.init_submodel(self.model_name)
        cut_points = list(self.wrapped.model_cut.keys())
        cut_points.sort()
        self.modelcutcombo.config(values = cut_points)
        #self.submodel1, self.submodel2 = self.wrapped.divide(model_name)
        if hasattr(self, "model_inp"):
            with torch.no_grad():
                if hasattr(self, "submodel1"):
                    self.mid = self.submodel1(self.model_inp)
                    if not self.selected:
                        self.ori_mid = self.submodel1(self.ori_inp)
                        self.mid_max[self.model_cut] = self.ori_mid.max(-1)[0].max(-1)[0]+1


                    self.max_channel =  self.mid.shape[1]
                    model_target = self.submodel2(self.mid).max(1)[1]
                    if self.display_channel_:
                        self.display_channel()
                else:
                    model_target = self.model(self.model_inp).max(1)[1]
                self.target.config(text = "Model Prediction: "+str(model_target.item()) + " " + get_class_name(model_target.item()))
                #if hasattr(self, "target"):
                #self.target.config(text = "Model Prediction: " + get_class_name(model_target.item()))
        self.model_loaded = True
                #else:
                #    self.target = tk.Label(self.window, text = "Model Prediction: " + get_class_name(model_target.item()) )
                #    self.target.grid(row = 1, column = 4, stick = tk.SW, columnspan = 3)


    def load_category(self):
        #if not hasattr(self,"model"):
            #if hasattr(self, "category_msg"):
        #    self.category_msg.config(text = "Model not loaded")
            #else:
            #    self.category_msg = ttk.Label(self.categoryframe, text="Model not loaded")
            #    self.category_msg.grid(column = 0, row = 2, columnspan = 1)
        #    return

        if not self.dataset_loaded:
            #if hasattr(self, "category_msg"):
            self.category_msg.config(text = "Dataset not loaded")
            #else:
            #    self.category_msg = ttk.Label(self.categoryframe, text="Dataset not loaded")
            #    self.category_msg.grid(column = 0, row = 2, columnspan = 1)
            return
        #if hasattr(self, "category_msg"):

        self.category_msg.config(text = "")
        category = self.custom_category.get()
        if not category:
            return
        category = int(category)
        if self.datasettype.get() == "train":
            data = self.train_data
        else:
            data = self.val_data

        targets = torch.tensor(data.targets)
        idxs = torch.argwhere(targets==category).squeeze().cpu().numpy().tolist()
        subset = torch.utils.data.Subset(data, idxs)
        for img,label in subset:
            self.insert_display(img.unsqueeze(0))
        self.category_msg.config(text = get_class_name(category) +" loaded")
        #dataloader = DataLoader(subset, batch_size=256, shuffle = False)
        #inps = []
        #for i, (inp,tar) in enumerate(dataloader):
        #    inp = inp.to(self.device)
        #    tar = tar.to(self.device)
        #    out = self.model(inp).max(1)[1]
        #    inp = inp[out==tar]
        #    inps.append(inp)
        #imgs = torch.cat(inps)
        #self.category_data[self.datasettype.get()][category] = imgs.cpu()

        #self.target.config(text = "Model Prediction: " + get_class_name(category.item()))


    def ind_attr(self):
        if not hasattr(self,"mid_attr"):
            #if hasattr(self, "channel_msg"):
            self.channel_msg.config(text = "Salient channels not computed")
            return
        if not hasattr(self,"ori_inp"):
            #if hasattr(self, "channel_msg"):
            self.channel_msg.config(text = "Input Image not loaded")
            return
        if not self.model_loaded:
            #if hasattr(self, "channel_msg"):
            self.channel_msg.config(text = "Model not loaded")
            return

        if not hasattr(self,"submodel1"):
            self.channel_msg.config(text = "Model Cut point not selected")
            return
        if not hasattr(self,"train_data") and not self.ind.get():
            self.channel_msg.config(text = "Input dataset not loaded")
            return
        if not hasattr(self,"train_data") and not self.ind.get():
            self.channel_msg.config(text = "Input dataset not loaded")
            return
        #if hasattr(self, "channel_msg"):
        #self.channel_msg.destroy()
        self.spatialbutton.config(state = tk.DISABLED)
        self.channel_msg.config(text = "Computing...")


        with torch.no_grad():
            #self.mid = self.submodel1(self.model_inp)
            mid = self.submodel1(self.ori_inp)
            model_target = self.submodel2(mid).max(1)[1]


        channel_attr = torch.zeros_like(self.mid)
        #print(self.mid_attr)

        channel_attr[:,self.mid_attr] = 1

        self.mid_spatial_attr = explain_channel_ind(self.submodel2, mid, model_target, channel_attr = channel_attr, beta = 1e-1)
        self.channel_msg.config(text = "Done...")
        #print(self.mid.shape)
        #print(self.mid[:,self.channel])
        self.mid = self.mid_spatial_attr*self.mid
        #print(self.mid_spatial_attr)
        #print(self.mid_spatial_attr[:,self.channel])
        #print(self.mid.shape)
        #print(self.mid[:,self.channel])
        self.spatialbutton.config(state = tk.NORMAL)
        if self.display_channel_:
            self.display_channel()


    def channel_attr(self):
        if not hasattr(self,"ori_inp"):
            #if hasattr(self, "channel_msg"):
            self.channel_msg.config(text = "Input Image not loaded")
            #else:
            #    self.channel_msg = ttk.Label(self.autochannelframe, text="Input Image not loaded")
            #    self.channel_msg.grid(column = 6, row = 5, columnspan = 1)
            return
        if not self.model_loaded:
            #if hasattr(self, "channel_msg"):
            self.channel_msg.config(text = "Model not loaded")
            #else:
            #    self.channel_msg = ttk.Label(self.autochannelframe, text="Model not loaded")
            #    self.channel_msg.grid(column = 6, row = 5, columnspan = 1)
            return

        if not hasattr(self,"submodel1"):
            #if hasattr(self, "channel_msg"):
            self.channel_msg.config(text = "Model Cut point not selected")
            return
        if not hasattr(self,"train_data") and not self.ind.get():
            #if hasattr(self, "channel_msg"):
            self.channel_msg.config(text = "Input dataset not loaded")
            #else:
            #    self.channel_msg = ttk.Label(self.autochannelframe, text="Input dataset not loaded")
            #    self.channel_msg.grid(column = 6, row = 5, columnspan = 1)
            return
        #if hasattr(self, "channel_msg"):
        #self.channel_msg.destroy()
        self.channelbutton.config(state = tk.DISABLED)
        self.channel_msg.config(text = "Computing...")


        with torch.no_grad():
            #self.mid = self.submodel1(self.model_inp)
            mid = self.submodel1(self.ori_inp)
            model_target = self.submodel2(mid).max(1)[1]

        if not self.ind.get():
            category = model_target.item()
            if self.datasettype.get() == "train":
                data = self.train_data
            else:
                data = self.val_data
            targets = torch.tensor(data.targets)
            idxs = torch.argwhere(targets==category).squeeze().cpu().numpy().tolist()
            subset = torch.utils.data.Subset(data, idxs)
            dataloader = DataLoader(subset, batch_size=256, shuffle = False)
            mids = []
            for i, (inp,tar) in enumerate(dataloader):
                inp = inp.to(self.device)
                tar = tar.to(self.device)
                mid = self.submodel1(inp)
                out = self.submodel2(mid).max(1)[1]
                mid = mid[out==tar]
                mids.append(mid)
            mid = torch.cat(mids)


            mid_attr = channel_greedy(self.submodel2, mid, model_target, batch_size = 256, preservation = self.preserved.get())
        else:
            mid_attr = channel_greedy(self.submodel2, mid, model_target, batch_size = 256, preservation = self.preserved.get(),threshold = True,heuristic ="softmax")

        self.channel_msg.config(text = "Done!")
        self.listbox.delete(0, tk.END)
        self.mid_attr = mid_attr
        for attr in mid_attr:
            self.listbox.insert(tk.END, attr)
        self.channelbutton.config(state = tk.NORMAL)




    def add_channel(self):
        if not hasattr(self, "mid"):
            self.add_channel_msg.config(text = "Mid activation not computed")
            return
        if self.custom_channel.get() < 0 or self.custom_channel.get() >= self.max_channel:
            self.add_channel_msg.config(text = "Channel out of range")
            return
        self.listbox.insert(tk.END, int(self.custom_channel.get()))
    def clear_channel(self):
        self.display_channel_ = False
        self.channel = None
        self.channel_canvas.delete("all")
        self.listbox.delete(0,tk.END)

    def channel_local(self):
        if not hasattr(self,"ori_inp"):
            self.local_msg.config(text = "Input Image not loaded")
            return
        if not self.model_loaded:
            self.local_msg.config(text = "Model not loaded")
            return
        if not hasattr(self,"submodel1"):
            self.local_msg.config(text = "Model Cut point not selected")
            return
        block_size = self.block_var.get()
        if block_size > 224 or block_size <= 0:
            self.local_msg.config(text = "Block size not supported")
            return

        self.localbutton.config(state = tk.DISABLED)
        #if not hasattr(self, "mid"):
        #    return
        global heat_photo
        self.local_msg.config(text = "Computing....")
        attr = channel_local_act(self.submodel1,self.ori_inp,self.channel,block_size=block_size,batch = 512)
        max_val = self.mid_max[self.model_cut][:,self.channel]
        self.local_msg.config(text = "done")
        display_value = attr/max_val*255
        upsample = torch.nn.Upsample(300, mode = "bilinear")
        display_value = upsample(display_value).cpu().detach().numpy()
        display_value = np.transpose(display_value, (0, 2, 3, 1)).squeeze()
        #heatmap = cv2.applyColorMap(display_value, cv2.COLORMAP_JET)

        #overlay = cv2.addWeighted(heatmap, 0.6, np.array(self.img.resize((300,300))), 0.4, 0)
        self.act_img= Image.fromarray(display_value)
        heat_photo = ImageTk.PhotoImage(self.act_img)
    
        self.listbox.select_set(self.channel)
        self.img_canvas.delete("all")
        self.img_canvas.create_image(0,0, anchor =tk.NW, image = heat_photo)
        self.localbutton.config(state = tk.NORMAL)


    def channelonselect(self,event):
        if not hasattr(self, "mid"):
            return
        w = event.widget
        index = w.curselection()
        if index:
            index = index[0]
            value = w.get(index)
            self.channel = value
            self.display_channel()
        
    def display_channel(self):
        global act_photo
        if self.channel is None:
            return
        self.display_channel_ = True
        #if value in 
        display_value = self.mid[:,self.channel].unsqueeze(1)
        max_val = self.mid_max[self.model_cut][:,self.channel]
        display_value = display_value/max_val*255
        upsample = torch.nn.Upsample(300, mode = "bilinear")
        display_value = upsample(display_value).cpu().detach().numpy()
        #self.act[value] = display_value
        display_value = np.transpose(display_value, (0, 2, 3, 1)).squeeze()


        self.act_img= Image.fromarray(display_value)
        act_photo = ImageTk.PhotoImage(self.act_img)
    
        self.channel_canvas.delete("all")
        self.channel_canvas.create_image(0,0, anchor =tk.NW, image = act_photo)

    def show_img(self,event):
        global show_photo
        
        self.mid_max = {}
        self.img_canvas.unbind('<Motion>')
        self.img_canvas.unbind("<ButtonPress-1>")
        self.img_canvas.unbind("<B1-Motion>")
        self.img_canvas.unbind("<ButtonRelease-1>")
        #image = event.widget.itemcget(obj, "image")
        self.img_canvas.delete("all")
        image = event.widget.image
        self.img = image.resize((224,224))
        self.select_img = self.img.resize((300,300))
        self.model_img = self.img.copy()
        self.ori_inp = numpy_to_tensor(np.array(self.model_img)/255.0)
        self.model_inp = self.ori_inp.clone()
        show_photo = ImageTk.PhotoImage(image.resize((300,300)))
        if hasattr(self, "rect"):
            del self.rect
        self.img_canvas.create_image(0,0, anchor =tk.NW, image = show_photo)

        img_np = np.array(self.img)
        self.dim_img = np.uint8(0.5*img_np)
        self.white_bg = 255*np.ones_like(img_np)
        self.noise_bg = np.uint8(255*np.random.rand(*img_np.shape))
        self.gauss_bg = gaussian_filter(img_np, sigma = (5,5,0))

        self.selected = False
        self.fix_shape = False
        if hasattr(self, "base_width"):
            del self.base_width
            del self.base_height

        if self.model_loaded:
            with torch.no_grad():
                if hasattr(self, "submodel1"):
                    self.mid = self.submodel1(self.model_inp)
                    self.ori_mid = self.submodel1(self.ori_inp)
                    self.mid_max[self.model_cut] = self.ori_mid.max(-1)[0].max(-1)[0]+1
                    self.max_channel =  self.mid.shape[1]
                    model_target = self.submodel2(self.mid).max(1)[1]
                    if self.display_channel_:
                        self.display_channel()
                else:
                    model_target = self.model(self.model_inp).max(1)[1]
                self.target.config(text = "Model Prediction: "+str(model_target.item()) + " " + get_class_name(model_target.item()))
                #if hasattr(self, "target"):
                #self.target.config(text = "Model Prediction: " + get_class_name(model_target.item()))
                #self.mid = self.submodel1(self.model_inp)
                #out = self.submodel2(self.mid).max(1)[1]
                #self.target.config(text = "Model Prediction: " + get_class_name(out.item()))


    def clear_display(self):
        self.display_canvas.delete("all")
        self.img_canvas.delete("all")
        self.channel_canvas.delete("all")
        for child in self.display_canvas.winfo_children():
            child.destroy()
        self.selected = False
        self.fix_shape = False
        self.show = False
        self.display_channel_ = False
        if hasattr(self, "base_width"):
            del self.base_width
            del self.base_height

        self.display_imgs = []
        self.displayscroll.config(command = None)
        self.display_canvas.config(height = 120, width = 660)
        self.display_canvas.config(xscrollcommand=None, scrollregion=None)
        if hasattr(self, "img"):
            del self.img
            del self.select_img
            del self.model_inp
            del self.ori_inp
        if hasattr(self, "mid"):
            del self.mid_attr
            del self.mid
            del self.max_channel
        if hasattr(self, "dim_img"):
            del self.dim_img
            del self.white_bg
            del self.noise_bg
            del self.gauss_bg
        if hasattr(self, "rect"):
            del self.rect

    def refresh(self):
        self.mid_max = {}
        if hasattr(self, "base_width"):
            del self.base_width
            del self.base_height
        self.fix_shape = False
        self.selected = False
        self.show = False
        self.img_canvas.delete("all")

        self.img_canvas.unbind('<Motion>')
        self.img_canvas.unbind("<ButtonPress-1>")
        self.img_canvas.unbind("<B1-Motion>")
        self.img_canvas.unbind("<ButtonRelease-1>")
        if hasattr(self, "rect"):
            del self.rect
        if hasattr(self, "img"):
            global refresh_photo
            self.select_img = self.img.resize((300,300))
            self.model_img = self.img
            self.model_inp = self.ori_inp.clone()
            refresh_photo = ImageTk.PhotoImage(self.img.resize((300,300)))
    
            self.img_canvas.create_image(0,0, anchor =tk.NW, image = refresh_photo)

            img_np = np.array(self.img)
            self.dim_img = np.uint8(0.5*img_np)
            self.white_bg = 255*np.ones_like(img_np)
            self.noise_bg = np.uint8(255*np.random.rand(*img_np.shape))
            self.gauss_bg = gaussian_filter(img_np, sigma = (5,5,0))


            if self.model_loaded:
                with torch.no_grad():
                    if hasattr(self, "submodel1"):
                        self.mid = self.submodel1(self.model_inp)
                        self.ori_mid = self.submodel1(self.ori_inp)
                        self.mid_max[self.model_cut] = self.ori_mid.max(-1)[0].max(-1)[0]+1
                        self.max_channel =  self.mid.shape[1]
                        model_target = self.submodel2(self.mid).max(1)[1]
                        if self.display_channel_:
                            self.display_channel()
                    else:
                        model_target = self.model(self.model_inp).max(1)[1]
                    self.target.config(text = "Model Prediction: "+str(model_target.item()) + " " + get_class_name(model_target.item()))

    def change_bg(self):
        if self.selected:
            global change_photo
            if self.bg.get() == "white":
                iminp = self.cur_white_bg
            elif self.bg.get() == "noise":
                iminp = self.cur_noise_bg
            elif self.bg.get() == "gauss":
                iminp = self.cur_gauss_bg
            self.model_img = Image.fromarray(iminp)
            self.model_inp = numpy_to_tensor(np.array(self.model_img)/255.0)
            #img_np = np.array(self.img)
            #if self.bg.get() == "white":
            #    if not hasattr(self, "white_bg"):
            #        self.white_bg = 255*np.ones_like(img_np)
            #    imselect = self.white_bg.copy()
            #elif self.bg.get() == "noise":
            #    if not hasattr(self, "noise_bg"):
            #        self.noise_bg = np.uint8(255*np.random.rand(*img_np.shape))
            #    imselect = self.noise_bg.copy()
            #elif self.bg.get() == "gauss":
            #    if not hasattr(self, "gauss_bg"):
            #        self.gauss_bg = gaussian_filter(img_np, sigma = (5,5,0))
            #    imselect = self.gauss_bg.copy()
            #imselect[int(self.start_y):int(self.end_y), int(self.start_x):int(self.end_x)] = img_np[int(self.start_y):int(self.end_y), int(self.start_x):int(self.end_x)]

            #self.img_canvas.delete("all")
            #self.select_img = Image.fromarray(imselect).resize((224,224))
            if self.model_loaded:
                with torch.no_grad():
                    if hasattr(self, "submodel1"):
                        self.mid = self.submodel1(self.model_inp)
                        self.max_channel =  self.mid.shape[1]
                        model_target = self.submodel2(self.mid).max(1)[1]
                        if self.display_channel_:
                            self.display_channel()
                    else:
                        model_target = self.model(self.model_inp).max(1)[1]
                    self.target.config(text = "Model Prediction: "+str(model_target.item()) + " " + get_class_name(model_target.item()))
            if self.show:
                change_photo = ImageTk.PhotoImage(self.model_img.resize((300,300)))
                self.img_canvas.delete("all")
                self.img_canvas.create_image(0,0, anchor =tk.NW, image = change_photo)


    def load_dataset(self):
        self.databutton.config(state = tk.DISABLED)
        cur_dir = os.getcwd()
        root = filedialog.askdirectory(initialdir =  cur_dir, title = "Select Imagenet Root")

        if not root:
            self.databutton.config(state = tk.NORMAL)
            return
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        cur_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
            ])
        traindir = os.path.join(root, "train")
        valdir = os.path.join(root, "val")
        try:
            self.train_data = datasets.ImageFolder(
                    traindir,
                    cur_transform)
            self.val_data = datasets.ImageFolder(
                    valdir,
                    cur_transform)
            self.load.config(text = "dataset loaded")
            self.dataset_loaded = True
            self.databutton.config(state = tk.NORMAL)
        except:
            self.load.config(text = "Wrong category")
            self.databutton.config(state = tk.NORMAL)

    def browse_and_display(self):
        global photo
        cur_dir = os.getcwd()
        filename = filedialog.askopenfilename(initialdir =  cur_dir, title = "Select An Image", filetypes =
        (("jpeg files","*.jpg"),('jpeg files', "*.JPEG"),("png filrs", "*.png"),("all files","*.*")))
        #label = ttk.Label(self.browseframe, text = "")
        #label.grid(column = 0, row = 2, sticky = tk.W)
        #label.configure(text = filename)
        if not filename:
            return
        if hasattr(self, "base_width"):
            del self.base_width
            del self.base_height
        self.fix_shape = False
        self.selected = False
        self.show = False
        self.img = Image.open(filename)
        #self.img = self.img.resize((300,300))
        self.img = self.img.resize((224,224))
        self.select_img = self.img.resize((300,300))
        self.model_img = self.img
        #self.ori_img = numpy_to_tensor(np.array(self.img.resize((224,224))))
        self.ori_inp = numpy_to_tensor(np.array(self.img)/255.0)
        self.model_inp = self.ori_inp.clone()

        self.insert_display(self.ori_inp)

        photo = ImageTk.PhotoImage(self.img.resize((300,300)))
        img_np = np.array(self.img)
        self.dim_img = np.uint8(0.5*img_np)
        self.white_bg = 255*np.ones_like(img_np)
        self.noise_bg = np.uint8(255*np.random.rand(*img_np.shape))
        self.gauss_bg = gaussian_filter(img_np, sigma = (5,5,0))
    
        self.img_canvas.delete("all")
        self.img_canvas.create_image(0,0, anchor =tk.NW, image = photo)
        if self.model_loaded:
            with torch.no_grad():
                if hasattr(self, "submodel1"):
                    self.mid = self.submodel1(self.model_inp)
                    self.ori_mid = self.submodel1(self.ori_inp)
                    self.mid_max[self.model_cut] = self.ori_mid.max(-1)[0].max(-1)[0]+1
                    self.max_channel =  self.mid.shape[1]
                    model_target = self.submodel2(self.mid).max(1)[1]
                    if self.display_channel_:
                        self.display_channel()
                else:
                    model_target = self.model(self.model_inp).max(1)[1]
                #if hasattr(self, "target"):
                self.target.config(text = "Model Prediction: "+str(model_target.item()) + " " + get_class_name(model_target.item()))
                #else:
                #    self.target = tk.Label(self.window, text = "Model Prediction: " + get_class_name(model_target.item()) )
                #    self.target.grid(row = 1, column = 4, stick = tk.SW, columnspan = 2)
        #canvas.grid(column = 0, row = 3)
        #canvas.update()
        #label2 = tk.Label(image=photo)
        #label2.image = photo
        #label2.grid(column=0, row=4)

    def drawROI(self):
        #If to re-draw the recch.no_grad():
        #canvas.delete('all')
        #self.rect = None
        #self.update_canvas()
        #self.myROI_button.grid_forget()
        #self.myROI_button = tkinter.Button(self.window, text="Selecting...", width=30, state = "disabled")
        #self.myROI_button['font'] = self.specialfont
        #self.myROI_button.grid(row = 3, column = 1, columnspan = 1)
        #self.ROIrect = True
        self.show = False
        self.img_canvas.bind('<Motion>', self.motion)
        self.img_canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.img_canvas.bind("<Shift-ButtonPress-1>", self.on_button_press)
        self.img_canvas.bind("<B1-Motion>", self.on_move_press)
        self.img_canvas.bind("<Shift-B1-Motion>", self.on_move_press_shift)
        self.img_canvas.bind("<ButtonRelease-1>", self.on_button_release)
        self.img_canvas.bind("<Shift-ButtonRelease-1>", self.on_button_release)
        if self.selected:
            global selected_photo_

            self.dim_img = self.cur_imselect.copy()
            self.white_bg = self.cur_white_bg.copy()
            self.noise_bg = self.cur_noise_bg.copy()
            self.gauss_bg = self.cur_gauss_bg.copy()

            selected_photo_ = ImageTk.PhotoImage(self.select_img)
            if hasattr(self, "rect"):
                del self.rect
            self.img_canvas.delete("all")
            self.img_canvas.create_image(0,0, anchor =tk.NW, image = selected_photo_)
            self.selected = False
        #self.myROI_button.grid_forget()
    
    # Shows the cursor coordinates on the photo on mycanvas.    
    def motion(self, event):
        self.x, self.y = event.x, event.y
        #if self.resize == False
        #if hasattr(self, "myxy"):
        #    self.myxy.config(text = "XY coordinates: " + str(self.x) + ", " + str(self.y))
        #else:
        #    self.myxy = tkinter.Label(self.window, text = "XY coordinates: " + str(np.round(self.x/self.scale_factor)) + ", " + str(np.round(self.y/self.scale_factor)))
        #else:
        #    self.myxy = tk.Label(self.window, text = "XY coordinates: " + str(self.x) + ", " + str(self.y))
        #    self.myxy.grid(row = 7, column = 1, columnspan = 1)
    
    # Create a rectangle or a circle on left-mouse click ONLY if there are no other rectangles or circles already present.
    def on_button_press(self, event1):
        #if not self.rect: # create rectangle/ circle if not yet exist
        #    # save mouse drag start position
        #    if self.ROIshape == 0:
        #        self.rect = self.mycanvas.create_rectangle(self.x, self.y, self.x+1, self.y+1, outline='red')
        #    else:
        #self.rect = self.img_canvas.create_oval(self.x, self.y, self.x+1, self.y+1, outline='red')


        if hasattr(self, "rect"):
            cur_x = event1.x
            cur_y = event1.y
            def close(x1,y1,x2,y2):
                return((x1-x2)**2+(y1-y2)**2) <= 13
            if close(cur_x,cur_y, self.start_x, self.start_y):
                self.start_x = self.end_x
                self.start_y = self.end_y
                self.img_canvas.bind("<B1-Motion>", self.on_move_press)
                self.img_canvas.bind("<ButtonRelease-1>", self.on_button_release)
            elif close(cur_x,cur_y, self.start_x, self.end_y):
                self.start_x = self.end_x
                self.img_canvas.bind("<B1-Motion>", self.on_move_press)
                self.img_canvas.bind("<ButtonRelease-1>", self.on_button_release)
            elif close(cur_x,cur_y, self.end_x, self.end_y):
                self.img_canvas.bind("<B1-Motion>", self.on_move_press)
                self.img_canvas.bind("<ButtonRelease-1>", self.on_button_release)
            elif close(cur_x,cur_y, self.end_x, self.start_y):
                self.start_y = self.end_y
                self.img_canvas.bind("<B1-Motion>", self.on_move_press)
                self.img_canvas.bind("<ButtonRelease-1>", self.on_button_release)
            else:
                return
            if self.selected:
                self.fix_shape = True
                self.base_width = abs(cur_x - self.start_x)
                self.base_height = abs(cur_y - self.start_y)
        else:

            self.start_x = event1.x
            self.start_y = event1.y

        #self.start_x = event1.x
        #self.start_y = event1.y

            self.rect = self.img_canvas.create_rectangle(self.x, self.y, self.x, self.y, outline='red')
    
    # Update the rectangle/circle size as the mouse performs 'move press' ONLY if the use has clicked 'draw ROI' or 'reselect ROI'
    def on_move_press(self, event2):
        #if self.myROI_button['state'] == "disabled":
        x = min(300, max(0,event2.x))
        y = min(300, max(0,event2.y))
        self.end_x = x 
        self.end_y = y
            # expand rectangle/circle as you drag the mouse
        #self.rect = self.img_canvas.create_rectangle(self.x, self.y, self.x+1, self.y+1, outline='red')
        self.img_canvas.coords(self.rect, self.start_x, self.start_y, self.end_x, self.end_y)
            #if self.resize == False:
        #if hasattr(self, "myxy"):
        #    self.myxy.config(text = "XY coordinates: " + str(self.x) + ", " + str(self.y))
        #else:
        #    self.myxy = tk.Label(self.window, text = "XY coordinates: " + str(self.curX) + ", " + str(self.curY))
        #    self.myxy.grid(row = 7, column = 1, columnspan = 1)


    def on_move_press_shift(self, event2):
        #if self.myROI_button['state'] == "disabled":
        x = min(300, max(0,event2.x))
        y = min(300, max(0,event2.y))
        x_diff = abs(x-self.start_x)
        y_diff = abs(y-self.start_y)
        if x > self.start_x:
            x_sign = 1
        else:
            x_sign = -1
        if y > self.start_y:
            y_sign = 1
        else:
            y_sign = -1
        if x_diff < y_diff:
            if self.fix_shape:
                #y_diff = x_diff/self.base_width*self.base_height
                x_diff = y_diff/self.base_height*self.base_width
            else:
                x_diff = y_diff
        else:
            if self.fix_shape:
                y_diff = x_diff/self.base_width*self.base_height
            else:
                y_diff = x_diff


        self.end_x = self.start_x + x_diff*x_sign
        self.end_y = self.start_y + y_diff*y_sign
        if self.end_x > self.start_x and self.end_x > 300:
            self.end_x = 300
            y_diff = (300-self.start_x)/self.base_width*self.base_height
            self.end_y = self.start_y + abs(y_diff)*y_sign
        elif self.end_x < self.start_x and self.end_x <= 0:
            self.end_x = 0
            y_diff = (0-self.start_x)/self.base_width*self.base_height
            self.end_y = self.start_y + abs(y_diff)*y_sign
        if self.end_y > self.start_y and self.end_y > 300:
            self.end_y = 300
            x_diff = (300-self.start_y)/self.base_height*self.base_width
            self.end_x = self.start_x + abs(x_diff)*x_sign
        elif self.end_y < self.start_y and self.end_y <=0:
            self.end_y = 0
            x_diff = (0-self.start_y)/self.base_height*self.base_width
            self.end_x = self.start_x + abs(x_diff)*x_sign




            # expand rectangle/circle as you drag the mouse
        #self.rect = self.img_canvas.create_rectangle(self.x, self.y, self.x+1, self.y+1, outline='red')
        #self.img_canvas.coords(self.rect, self.start_x, self.start_y, self.curX, self.curY)
        self.img_canvas.coords(self.rect, self.start_x, self.start_y, self.end_x, self.end_y)
            #if self.resize == False:
        #if hasattr(self, "myxy"):
        #    self.myxy.config(text = "XY coordinates: " + str(self.x) + ", " + str(self.y))
        #else:
        #    self.myxy = tk.Label(self.window, text = "XY coordinates: " + str(self.curX) + ", " + str(self.curY))
        #    self.myxy.grid(row = 7, column = 1, columnspan = 1)



    #def on_button_release(self, event3):
    #    global selected_photo
    #    self.selected = True
    #    #self.img_canvas.delete(self.rect)
    #    x = min(300, max(0,event3.x))
    #    y = min(300, max(0,event3.y))
    #    self.end_x = x
    #    self.end_y = y

    def on_button_release(self, event3):
        global selected_photo
        self.selected = True
        #self.img_canvas.delete(self.rect)
        #x = min(300, max(0,event3.x))
        #y = min(300, max(0,event3.y))
        #self.end_x = x
        #self.end_y = y


        if self.end_x < self.start_x:
            tmp = self.start_x
            self.start_x = self.end_x
            self.end_x = tmp
        if self.end_y < self.start_y:
            tmp = self.start_y
            self.start_y = self.end_y
            self.end_y = tmp
        if self.start_x == self.end_x or self.start_y == self.end_y:
            self.img_canvas.delete(self.rect)
            del self.rect
            return
        self.img_canvas.unbind("<B1-Motion>")
        self.img_canvas.unbind("<ButtonRelease-1>")
        #img_np = np.array(self.img)
        img_np = np.array(self.img)
        self.cur_imselect = self.dim_img.copy()
        #self.selected_roi.append((self.start_y, self.end_y, self.start_x, self.end_x))

        #if self.bg.get() == "white":
        #    if not hasattr(self, "white_bg"):
        #        self.white_bg = 255*np.ones_like(img_np)
        #iminp = self.white_bg
        #elif self.bg.get() == "noise":
        #    if not hasattr(self, "noise_bg"):
        #        self.noise_bg = np.uint8(255*np.random.rand(*img_np.shape))
        #    iminp = self.noise_bg
        #elif self.bg.get() == "gauss":
        #    if not hasattr(self, "gauss_bg"):
        #        self.gauss_bg = gaussian_filter(img_np, sigma = (5,5,0))
        #    iminp = self.gauss_bg
        self.cur_white_bg = self.white_bg.copy()
        self.cur_noise_bg = self.noise_bg.copy()
        self.cur_gauss_bg = self.gauss_bg.copy()

        start_x = int(self.start_x/300*224)
        start_y = int(self.start_y/300*224)
        end_x = int(self.end_x/300*224)
        end_y = int(self.end_y/300*224)


        self.cur_imselect[int(start_y):int(end_y), int(start_x):int(end_x)] = img_np[int(start_y):int(end_y), int(start_x):int(end_x)]
        self.cur_white_bg[int(start_y):int(end_y), int(start_x):int(end_x)] = img_np[int(start_y):int(end_y), int(start_x):int(end_x)]
        self.cur_noise_bg[int(start_y):int(end_y), int(start_x):int(end_x)] = img_np[int(start_y):int(end_y), int(start_x):int(end_x)]
        self.cur_gauss_bg[int(start_y):int(end_y), int(start_x):int(end_x)] = img_np[int(start_y):int(end_y), int(start_x):int(end_x)]

        self.img_canvas.delete("all")
        self.select_img = Image.fromarray(self.cur_imselect).resize((300,300))
        if self.bg.get() == "white":
            iminp = self.cur_white_bg
        elif self.bg.get() == "noise":
            iminp = self.cur_noise_bg
        elif self.bg.get() == "gauss":
            iminp = self.cur_gauss_bg
        self.model_img = Image.fromarray(iminp)
        self.model_inp = numpy_to_tensor(np.array(self.model_img)/255.0)
        if self.model_loaded:
            with torch.no_grad():
                if hasattr(self, "submodel1"):
                    self.mid = self.submodel1(self.model_inp)
                    self.max_channel =  self.mid.shape[1]
                    model_target = self.submodel2(self.mid).max(1)[1]
                    if self.display_channel_:
                        self.display_channel()
                else:
                    model_target = self.model(self.model_inp).max(1)[1]
                self.target.config(text = "Model Prediction: "+str(model_target.item()) + " " + get_class_name(model_target.item()))


        #self.select_img.resize((300,300))
        selected_photo = ImageTk.PhotoImage(self.select_img)
        #self.select_canvas.create_image(self.start_x,self.start_y, anchor =tk.NW, image = selected_photo)
        #self.img_canvas.delete("all")
        self.img_canvas.delete("all")
        self.img_canvas.create_image(0,0, anchor =tk.NW, image = selected_photo)
        self.rect = self.img_canvas.create_rectangle(self.start_x, self.start_y, self.end_x, self.end_y, outline='red')

        #self.myROI_button.grid_forget()
        #self.update_ROI()        
    
    def show_bg(self):
        self.img_canvas.unbind('<Motion>')
        self.img_canvas.unbind("<ButtonPress-1>")
        self.img_canvas.unbind("<B1-Motion>")
        self.img_canvas.unbind("<ButtonRelease-1>")
        if self.selected:
            global photo
            self.dim_img = self.cur_imselect.copy()
            self.white_bg = self.cur_white_bg.copy()
            self.noise_bg = self.cur_noise_bg.copy()
            self.gauss_bg = self.cur_gauss_bg.copy()

            self.show = True
            #if self.bg.get() == "white":
            #    iminp = self.white_bg
            #elif self.bg.get() == "noise":
            #    iminp = self.noise_bg
            #elif self.bg.get() == "gauss":
            #    iminp = self.gauss_bg
            #self.model_img = Image.fromarray(iminp.resize(224,224))
            #self.model_inp = numpy_to_tensor(np.array(self.model_img))
            photo = ImageTk.PhotoImage(self.model_img.resize((300,300)))
            self.img_canvas.delete("all")
            self.img_canvas.create_image(0,0, anchor =tk.NW, image = photo)

gui = GUI()
gui.window.mainloop()
#root.mainloop()



#cropped_lbl = tk.Label(root)
#cropped_lbl.pack(expand=True, fill="both")
#
#
#root.bind("<<ROISELECTED>>", display_roi)
#root.mainloop()


