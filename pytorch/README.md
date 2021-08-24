
# Transferable Visual Words - Official Pytorch Implementation

We provide the official <b>Pytorch</b> implementation of training TransVW from scratch on unlabeled images as well as the usage of the pre-trained TransVW reported in the following paper:

<b>Transferable Visual Words:  Exploiting the Semantics of Anatomical Patterns for Self-supervised Learning </b> <br/>
[Fatemeh Haghighi](https://github.com/fhaghighi)<sup>1</sup>, [Mohammad Reza Hosseinzadeh Taher](https://github.com/MR-HosseinzadehTaher)<sup>1</sup>, [Zongwei Zhou](https://github.com/MrGiovanni)<sup>1</sup>, [Michael B. Gotway](https://www.mayoclinic.org/biographies/gotway-michael-b-m-d/bio-20055566)<sup>2</sup>, [Jianming Liang](https://chs.asu.edu/jianming-liang)<sup>1</sup><br/>
Arizona State University<sup>1</sup>, </sup>Mayo Clinic <sup>2</sup><br/>
IEEE Transactions on Medical Imaging ([TMI](https://www.embs.org/tmi/)), 2021 <br/>
[paper](https://arxiv.org/pdf/2102.10680.pdf) | [code](https://github.com/fhaghighi/TransVW)

## Requirements

+ Linux
+ Python 3.7.5
+ PyTorch 1.3.1

## Using the pre-trained TransVW
### 1. Clone the repository and install dependencies
```bash
$ git clone https://github.com/fhaghighi/TransVW.git
$ cd TransVW/
$ pip install -r requirements.txt
```

### 2. Download the pre-trained TransVW
Download the pre-trained TransVW as following and save into `./pytorch/Checkpoints/en_de/TransVW_chest_ct.pt` directory.

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">Backbone</th>
<th valign="bottom">Platform</th>
<th valign="bottom">model</th>

<!-- TABLE BODY -->
<tr><td align="left">TransVW</td>
<td align="center"><a href="https://github.com/ellisdg/3DUnetCNN">U-Net 3D</a></td>
<td align="center">Pytorch</td>
<td align="center"><a href="https://zenodo.org/record/4625321/files/TransVW_chest_ct.pt?download=1">download</a></td>
</tr>
</tbody></table>

### 3. Fine-tune TransVW on your own target task
TransVW learns a generic semantics-enriched image representation that can be leveraged for a wide range of target tasks. Specifically, TransVW provides a pre-trained 3D U-Net network, which the encoder can be utilized for the target <i>classification</i> tasks and encoder-decoder for the target <i>segmentation</i> tasks.

As for the target classification tasks, the 3D deep model can be initialized with the pre-trained encoder using the following example:
```python


import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from models.ynet3d import *

# prepare your own data
train_loader = DataLoader(Your Dataset, batch_size=config.batch_size, shuffle=True)

# prepare the 3D model
class TargetNet(nn.Module):
    def __init__(self, base_model,n_class=1):
        super(TargetNet, self).__init__()

        self.base_model = base_model
        self.dense_1 = nn.Linear(512, n_class, bias=True)

    def forward(self, x):
        self.base_model(x)
        self.base_out = self.base_model.out512
        # This global average polling is for shape (N,C,H,W) not for (N, H, W, C)
        # where N = batch_size, C = channels, H = height, and W = Width
        self.out_glb_avg_pool = F.avg_pool3d(self.base_out, kernel_size=self.base_out.size()[2:]).view(self.base_out.size()[0],-1)
        self.linear_out = self.dense_1(self.out_glb_avg_pool)
        final_out = F.relu(self.linear_out)
        return final_out
        
base_model = UNet3D()

#Load pre-trained weights
weight_dir = 'Checkpoints/en_de/TransVW_chest_ct.pt'
checkpoint = torch.load(weight_dir)
state_dict = checkpoint['state_dict']
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
delete = [key for key in state_dict if "projection_head" in key]
for key in delete: del state_dict[key]
delete = [key for key in state_dict if "prototypes" in key]
for key in delete: del state_dict[key]
for key in state_dict.keys():
    if key in base_model.state_dict().keys():
        base_model.state_dict()[key].copy_(state_dict[key])
        print("Copying {} <---- {}".format(key, key))
    elif key.replace("classficationNet.", "") in base_model.state_dict().keys():
        base_model.state_dict()[key.replace("classficationNet.", "")].copy_(state_dict[key])
        print("Copying {} <---- {}".format(key.replace("classficationNet.", ""), key))
    else:
        print("Key {} is not found".format(key))

target_model = TargetNet(base_model)
target_model.to(device)
target_model = nn.DataParallel(target_model, device_ids = [i for i in range(torch.cuda.device_count())])
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(target_model.parameters(), config.lr, momentum=0.9, weight_decay=0.0, nesterov=False)

# train the model

for epoch in range(intial_epoch, config.nb_epoch):
    target_model.train()
    for batch_ndx, (x,y) in enumerate(train_loader):
        x, y = x.float().to(device), y.float().to(device)
        pred = F.sigmoid(target_model(x))
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

As for the target segmentation tasks, the 3D deep model can be initialized with the pre-trained encoder-decoder using the following example:
```python

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from models.ynet3d import *

#Declare the Dice Loss
def torch_dice_coef_loss(y_true,y_pred, smooth=1.):
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    intersection = torch.sum(y_true_f * y_pred_f)
    return 1. - ((2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth))

# prepare your own data
train_loader = DataLoader(Your Dataset, batch_size=config.batch_size, shuffle=True)

# prepare the 3D model

model = UNet3D()

#Load pre-trained weights
weight_dir = 'Checkpoints/en_de/TransVW_chest_ct.pt'
checkpoint = torch.load(weight_dir)
state_dict = checkpoint['state_dict']
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
delete = [key for key in state_dict if "projection_head" in key]
for key in delete: del state_dict[key]
delete = [key for key in state_dict if "prototypes" in key]
for key in delete: del state_dict[key]
for key in state_dict.keys():
    if key in model.state_dict().keys():
        model.state_dict()[key].copy_(state_dict[key])
        print("Copying {} <---- {}".format(key, key))
    elif key.replace("classficationNet.", "") in model.state_dict().keys():
        model.state_dict()[key.replace("classficationNet.", "")].copy_(state_dict[key])
        print("Copying {} <---- {}".format(key.replace("classficationNet.", ""), key))
    else:
        print("Key {} is not found".format(key))

model.to(device)
model = nn.DataParallel(model, device_ids = [i for i in range(torch.cuda.device_count())])
criterion = torch_dice_coef_loss
optimizer = torch.optim.SGD(model.parameters(), config.lr, momentum=0.9, weight_decay=0.0, nesterov=False)

# train the model

for epoch in range(intial_epoch, config.nb_epoch):
    model.train()
    for batch_ndx, (x,y) in enumerate(train_loader):
        x, y = x.float().to(device), y.float().to(device)
        pred = model(x)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

```


## Training TransVW on your own unlabeled data

### 1. Clone the repository and install dependencies
```bash
$ git clone https://github.com/fhaghighi/TransVW.git
$ cd TransVW/
$ pip install -r requirements.txt
```

### 2. Preparing data

#### For your convenience, we have provided our own self-discoverd 3D visual words from LUNA16 dataset as well as their pseudo labels.
Download the data from [this repository](https://zenodo.org/record/4625321/files/TransVW_data.zip?download=1). We have provided the training and validation samples for C=50 classes of visual words. For each instance of a visual word, we have extracted 3 multi-resolution cubes from each patient, where each of the three resolutions are saved in files named as 'train_dataN_vwGen_ex_ref_fold1.0.npy',  *N*=1,2,3. For each 'train_dataN_vwGen_ex_ref_fold1.0.npy' file, there is a corresponding 'train_labelN_vwGen_ex_ref_fold1.0.npy' file, which contains the pseudo labels of the discovered visual words.  


- The processed anatomical patterns directory structure
```
TransVW_data/
    |--  train_data1_vwGen_ex_ref_fold1.0.npy  : training data - resolution 1
    |--  train_data2_vwGen_ex_ref_fold1.0.npy  : training data - resolution 2
    |--  train_data3_vwGen_ex_ref_fold1.0.npy  : training data - resolution 3
    |--  val_data1_vwGen_ex_ref_fold1.0.npy    : validation data
    |--  train_label1_vwGen_ex_ref_fold1.0.npy : training labels - resolution 1
    |--  train_label2_vwGen_ex_ref_fold1.0.npy : training labels - resolution 2
    |--  train_label3_vwGen_ex_ref_fold1.0.npy : training labels - resolution 3
    |--  val_label1_vwGen_ex_ref_fold1.0.npy   : validation labels
   
```

####  You can perform the self-discovery on your own dataset following the steps below:

**Step 1**: Divide your training data into the train and validation folders, and put them in the `dataset` directory. 

**Step 2**: Train an auto-encoder using your data. The pre-trained model will be saved into `self_discovery/Checkpoints/Autoencoder/` directory.  

```bash
python -W ignore self_discovery/train_autoencoder.py 
--data_dir dataset/ 
```
**Step 3**: Extract and save the deep features of each patient in the dataset using the pre-trained auto-encoder:

```bash
python -W ignore self_discovery/feature_extractor.py 
--data_dir dataset/  
--weights self_discovery/Checkpoints/Autoencoder/Unet_autoencoder.h5
```

**Step 4**: Extract 3D visual words from train and validation images. The data and their labels will be save into `self_discovery/TransVW_data` directory.

```bash
python -W ignore self_discovery/pattern_generator_3D.py 
--data_dir dataset/  
--multi_res

```

### 3. Pre-train TransVW 
```bash
python -W ignore pytorch/train.py
--data_dir self_discovery/TransVW_data
```
Your pre-trained TransVW will be saved at `./pytorch/Checkpoints/en_de/TransVW_chest_ct.pt`.

## Citation
If you use our source code and/or refer to the baseline results published in the paper, please cite our [paper](https://arxiv.org/pdf/2102.10680.pdf) by using the following BibTex entry:
```

@misc{haghighi2021transferable,
      title={Transferable Visual Words: Exploiting the Semantics of Anatomical Patterns for Self-supervised Learning}, 
      author={Fatemeh Haghighi and Mohammad Reza Hosseinzadeh Taher and Zongwei Zhou and Michael B. Gotway and Jianming Liang},
      year={2021},
      eprint={2102.10680},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgement
We thank [Zuwei Guo](https://www.linkedin.com/in/zuwei/) for implementation of TransVW in Pytorch. Credit to [Models Genesis](https://github.com/MrGiovanni/ModelsGenesis) by [Zongwei Zhou](https://github.com/MrGiovanni). We build 3D U-Net architecture by referring to the released code at [mattmacy/vnet.pytorch](https://github.com/mattmacy/vnet.pytorch). This is a patent-pending technology.
