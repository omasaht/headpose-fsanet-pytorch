# headpose-fsanet-pytorch
Pytorch implementation of FSA-Net: Learning Fine-Grained Structure Aggregation for Head Pose Estimation from a Single Image<sup>[2](#references)</sup>.

## Demo
![demo](extras/headpose-demo.gif?raw=true)

Video file or a camera index can be provided to demo script. If no argument is provided, default camera index is used.

### Video File Usage

For any video format that OpenCV supported (`mp4`, `avi` etc.):

```bash
python3 demo.py --video /path/to/video.mp4
```

### Camera Usage

```bash
python3 demo.py --cam 0
``` 

## Results

| Model | Dataset Type | Yaw (MAE) | Pitch (MAE) | Roll (MAE) |
| --- | --- | --- | --- | --- |
| FSA-Caps (1x1) | 1  | 4.85 | 6.27 | 4.96 |
| FSA-Caps (Var)  | 1  | 5.06 | 6.46 | 5.00 |
| FSA-Caps (1x1 + Var) | 1 | **4.64** | **6.10** | **4.79** |

**Note:** My results are slightly worse than original author's results. For best results, please refer to official repository<sup>[1](#acknowledgements)</sup>.


## Dependencies

```
Name                      Version 
python                    3.7.6
numpy                     1.18.5
opencv                    4.2.0
scipy                     1.5.0
matplotlib-base           3.2.2
pytorch                   1.5.1
torchvision               0.6.1
onnx                      1.7.0
onnxruntime               1.2.0
```


Installation with pip
```bash
pip3 install -r requirements.txt
```


You may also need to install jupyter to access notebooks (.ipynb). It is recommended that you use Anaconda to install packages.

Code has been tested on Ubuntu 18.04

## Important Files Overview

- **src/dataset.py:** Our pytorch dataset class is defined here
- **src/model.py:** Pytorch FSA-Net model is defined here
- **src/transforms.py:** Augmentation Transforms are defined here
- **src/1-Explore Dataset.ipynb:** To explore training data, refer to this notebook
- **src/2-Train Model.ipynb:** For model training, refer to this notebook
- **src/3-Test Model.ipynb:** For model testing, refer to this notebook
- **src/4-Export to Onnx.ipynb:** For exporting model, refer to this notebook
- **src/demo.py:** Demo script is defined here

## Download Dataset
For model training and testing, download the preprocessed dataset from author's official git repository<sup>[1](#acknowledgements)</sup> and place them inside data/ directory. I am only using type1 data for training and testing. Your dataset hierarchy should look like:

```
data/
  type1/
    test/
      AFLW2000.npz
    train/
      AFW.npz
      AFW_Flip.npz
      HELEN.npz
      HELEN_Flip.npz
      IBUG.npz
      IBUG_Flip.npz
      LFPW.npz
      LFPW_Flip.npz
```

## License
Copyright (c) 2020, Omar Hassan. (MIT License)

## Acknowledgements
Special thanks to Mr. Tsun-Yi Yang for providing an excellent code to his paper. Please refer to the official repository to see detailed information and best results regarding the model:

\[1] T. Yang, FSA-Net, (2019), [GitHub repository](https://github.com/shamangary/FSA-Net)

The models are trained and tested with various public datasets which have their own licenses. Please refer to them before using the code

- 300W-LP: http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm
- LFPW: https://neerajkumar.org/databases/lfpw/
- HELEN: http://www.ifp.illinois.edu/~vuongle2/helen/
- AFW: https://www.ics.uci.edu/~xzhu/face/
- IBUG: https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/
- AFW2000: http://cvlab.cse.msu.edu/lfw-and-aflw2000-datasets.html

## References
\[2] T. Yang, Y. Chen, Y. Lin and Y. Chuang, "FSA-Net: Learning Fine-Grained Structure Aggregation for Head Pose Estimation From a Single Image," 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Long Beach, CA, USA, 2019, pp. 1087-1096, doi: 10.1109/CVPR.2019.00118. [IEEE-Xplore link](https://ieeexplore.ieee.org/document/8954346)

\[3] Tal Hassner, Shai Harel, Eran Paz, and Roee Enbar. Effective face frontalization in unconstrained images. In CVPR, 2015

\[4] Xiangyu Zhu, Zhen Lei, Junjie Yan, Dong Yi, and Stan Z. Li. High-fidelity pose and expression normalization for face recognition in the wild. In CVPR, 2015.
