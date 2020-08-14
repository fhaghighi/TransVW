# Transferable Visual Words

we conceive a new idea that the sophisticated, recurrent patterns in medical images are _anatomical visual words_, which can be automatically discovered from unlabeled medical image data, serving as strong yet free supervision signals for deep convolutional neural networks (DCNN) to learn disentangled representations, via self-supervised learning.
we train deep models to learn semantically enriched visual representation by self-discovery, self-classification, and self-restoration of the anatomical _visual words_, resulting in semantics-enriched, general-purpose, pre-trained 3D models, which we call <b>TransVW (transferable visual words)</b> for their transferable and generalizable capabilities to target tasks.

We envision that TransVW can be considered as an add-on, which can be added to and boost existing self-supervised learning methods; and more importantly, TransVW is an annotation-efficient solution to medical image analysis since it achieves superior performance, accelerates the convergence speed, and reduces the annotation efforts against all its 3D counterparts by a large margin.
\
![Image of framework](https://github.com/fhaghighi/TransVW/blob/master/images/framework.png)


## Paper
<b>Transferable Visual Words</b> <br/>

[Fatemeh Haghighi](https://github.com/fhaghighi)<sup>1</sup>, [Mohammad Reza Hosseinzadeh Taher](https://github.com/MR-HosseinzadehTaher)<sup>1</sup>,[Zongwei Zhou](https://github.com/MrGiovanni)<sup>1</sup>,[Michael B. Gotway](https://www.mayoclinic.org/biographies/gotway-michael-b-m-d/bio-20055566)<sup>2</sup>, [Jianming Liang](https://chs.asu.edu/jianming-liang)<sup>1</sup><br/>
Arizona State University<sup>1</sup>, </sup>Mayo Clinic, <sup>2</sup><br/>
Submitted to  IEEE Transactions on Medical Imaging (TMI) for Special Issue on Annotation-Efficient Deep Learning for Medical Imaging 2020

## Available implementation
<a href="https://keras.io/" target="_blank">
<img alt="Keras" src="https://github.com/fhaghighi/SemanticGenesis/blob/master/images/keras_logo.png" width="200" height="55"> </a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://pytorch.org/" target="_blank"><img alt="Keras" src="https://github.com/fhaghighi/SemanticGenesis/blob/master/images/pytorch_logo.png" width="200" height="48"></a>  

## Citation
If you use our source code and/or refer to the baseline results published in the paper, please cite our [paper](https://github.com/fhaghighi/SemanticGenesis) by using the following BibTex entry:
```
@article{haghighi2020transvw,
  author="Haghighi, Fatemeh and Hosseinzadeh Taher, Mohammad Reza and Zhou, Zongwei and Gotway, Michael B. and Liang, Jianming",
  title="Transferable Visual Words",
  journal="",
  year="2020",
  url=""
}
```


## Acknowledgement
This research has been supported partially by ASU and Mayo Clinic through a Seed Grant and an Innovation Grant, and partially by the National Institutes of Health (NIH) under Award Number R01HL128785. The content is solely the responsibility of the authors and does not necessarily represent the official views of the NIH. This work has utilized the GPUs provided partially by the ASU Research Computing and partially by the Extreme Science and Engineering Discovery Environment (XSEDE) funded by the National Science Foundation (NSF) under grant number ACI-1548562. This is a patent-pending technology.

## License

Released under the [ASU GitHub Project License](https://github.com/fhaghighi/TransVW/blob/master/LICENSE).
