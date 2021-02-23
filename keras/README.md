
# Transferable Visual Words - Official Keras Implementation

We provide the official <b>Keras</b> implementation of training TransVW from scratch on unlabeled images as well as the usage of the pre-trained TransVW reported in the following paper:

<b>Transferable Visual Words:  Exploiting the Semantics of Anatomical Patterns for Self-supervised Learning </b> <br/>
[Fatemeh Haghighi](https://github.com/fhaghighi)<sup>1</sup>, [Mohammad Reza Hosseinzadeh Taher](https://github.com/MR-HosseinzadehTaher)<sup>1</sup>, [Zongwei Zhou](https://github.com/MrGiovanni)<sup>1</sup>, [Michael B. Gotway](https://www.mayoclinic.org/biographies/gotway-michael-b-m-d/bio-20055566)<sup>2</sup>, [Jianming Liang](https://chs.asu.edu/jianming-liang)<sup>1</sup><br/>
Arizona State University<sup>1</sup>, </sup>Mayo Clinic <sup>2</sup><br/>
IEEE Transactions on Medical Imaging ([TMI](https://www.embs.org/tmi/)), 2021 <br/>

[paper]() | [code](https://github.com/fhaghighi/test_transvw/) | talk ([YouTube]()) 

## Requirements

+ Linux
+ Python 3.7.5
+ Keras 2.2.4+
+ TensorFlow 1.14.0+

## Using the pre-trained TransVW
### 1. Clone the repository and install dependencies
```bash
$ git clone https://github.com/fhaghighi/TransVW.git
$ cd TransVW/
$ pip install -r requirements.txt
```

### 2. Download the pre-trained TransVW
Download the pre-trained TransVW as following and save into `./keras/Checkpoints/TransVW_chest_ct.h5` directory.

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
<td align="center">Keras</td>
<td align="center"><a href="">download</a></td>
</tr>
</tbody></table>

### 3. Fine-tune TransVW on your own target task
TransVW learns a generic semantics-enriched image representation that can be leveraged for a wide range of target tasks. Specifically, TransVW provides a pre-trained 3D U-Net network, which the encoder can be utilized for the target <i>classification</i> tasks and encoder-decoder for the target <i>segmentation</i> tasks.

As for the target classification tasks, the 3D deep model can be initialized with the pre-trained encoder using the following example:
```python
# prepare your own data
X, y = your_data_loader()

# prepare the 3D model
import keras
from models.ynet3d import *
input_channels, input_rows, input_cols, input_deps = 1, 64, 64, 32
num_class, activate = 2, 'softmax'
weight_dir = './keras/Checkpoints/TransVW_chest_ct.h5'
TransVW = ynet_model_3d((input_channels, input_rows, input_cols, input_deps), batch_normalization=True)
print("Load pre-trained TransVW weights from {}".format(weight_dir))
TransVW.load_weights(weight_dir)

x = TransVW.get_layer('depth_7_relu').output
x = keras.layers.GlobalAveragePooling3D()(x)
output = keras.layers.Dense(num_class, activation=activate)(x)
model = keras.models.Model(inputs=TransVW.input, outputs=output)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy","categorical_crossentropy"])

# train the model
model.fit(X, y)
```

As for the target segmentation tasks, the 3D deep model can be initialized with the pre-trained encoder-decoder using the following example:
```python
# prepare your own data
X, Y = your_data_loader()

# prepare the 3D model
from unet3d import *
from models.ynet3d import *
input_channels, input_rows, input_cols, input_deps = 1, 64, 64, 32
num_class = 2
weight_dir = './keras/Checkpoints/TransVW_chest_ct.h5'
TransVW = ynet_model_3d((input_channels, input_rows, input_cols, input_deps), batch_normalization=True)
print("Load pre-trained TransVW weights from {}".format(weight_dir))
TransVW.load_weights(weight_dir)
model = unet_model_3d((1,config.input_rows,config.input_cols,config.input_deps), batch_normalization=True)

for layer in tuple(model.layers):
    if "input" not in layer.name and "max_pooling3d" not in layer.name \
            and "up_sampling3d" not in layer.name and "concatenate_" not in layer.name \
            and "conv3d_1" not in layer.name and "activation" not in layer.name \
            and not layer.name.startswith("conv3d"):
        layer.set_weights(TransVW.get_layer(layer.name).get_weights())
models.compile(optimizer="adam", loss=dice_coef_loss, metrics=[mean_iou,dice_coef])

# train the model
model.fit(X, Y)
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
Download the data from [Google Drive](). We have provided the training and validation samples for C=50 classes of visual words. For each instance of a visual word, we have extracted 3 multi-resolution cubes from each patient, where each of the three resolutions are saved in files named as 'train_dataN_vwGen_ex_ref_fold1.0.npy',  *N*=1,2,3. For each 'train_dataN_vwGen_ex_ref_fold1.0.npy' file, there is a corresponding 'train_labelN_vwGen_ex_ref_fold1.0.npy' file, which contains the pseudo labels of the discovered visual words.  


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
--weights self_discovery/Checkpoints/Autoencoder/Vnet_autoencoder.h5
```

**Step 4**: Extract 3D visual words from train and validation images. The data and their labels will be save into `self_discovery/TransVW_data` directory.

```bash
python -W ignore self_discovery/pattern_generator_3D.py 
--data_dir dataset/  
--multi_res

```

### 3. Pre-train TransVW 
```bash
python -W ignore keras/train.py
--data_dir self_discovery/TransVW_data
```
Your pre-trained TransVW will be saved at `./keras/Checkpoints/TransVW_chest_ct.h5`.

## Citation
If you use our source code and/or refer to the baseline results published in the paper, please cite our [paper]() by using the following BibTex entry:
```
@misc{
}
```

## Acknowledgement
We thank [Fatemeh Haghighi](https://github.com/fhaghighi) and [Mohammad Reza Hosseinzadeh Taher](https://github.com/MR-HosseinzadehTaher) for their implementation of TransVW in Keras. Credit to [Models Genesis](https://github.com/MrGiovanni/ModelsGenesis) by [Zongwei Zhou](https://github.com/MrGiovanni). We build 3D U-Net architecture by referring to the released code at [ellisdg/3DUnetCNN](https://github.com/ellisdg/3DUnetCNN). This is a patent-pending technology.
