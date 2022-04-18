**A simple Unet Benchmark on Carvana for Neural Network Verification**

We propose a new set of benchmarks of simple unet on Carvana for neural network verification in this repository.

**Motivation**

Currently, most networks evaluated in the literature are focusing on image classfication. However, in most practical scenarios, eg. autonomous driving, people pay more attention to object detection or semantic segmentation. Considering the complexity of the object detection, we propose a new simple Unet(four Conv2d layers followed with BN and ReLu). We advocate that tools should handle more practical architectures, and the simplified Unet is the first step towards this goal.

**Model details**

The ONNX format networks are available in the *[onnx] (network/onnx/)* folder, and the pytorch models can be found in the *[pytorch] (network.pytorch/)*  folder. And the inference script(`evaluate_network.py`) of onnx model can be found in the *[src] (src/)* folder.

**Data Format**

The input images should be normalized to the [0, 1] range and the ground truth masks should be normalized to 0 or 1. We also provide `img_process` and  `gt_mask_process` function in `evaluate_network.py`.

**Data Selection**

The Carvana dataset consists of 5088 images covering 318 cars, which means each car has 16 images. We choose one image for each car, 318 images in total, as a testset. And the remaining 4700 images are used for training. There are 43 images whose 98.7 percent pixes can be prectied correctly. We propose to randomly select 40 images from 43 iamges for verification.

**More details**

- There are two inputs for the model, including image and ground truth mask.
- The model has two outputs, including logits and the number of correctly predicted pixes by the model.  And the participant should focus on the latter which is defined in the vnnlib file. The logits is intermediate features for backpropagation.
- The image and mask size are both 47x31 to alleviate computational burden.
- The .vnnlib and .csv files were created with `generate_properties.py` script, which can be found in  *[src] (src/)*  folder.
- The Carvana-unet_instances.csv containts the full list of benchmark instances, one per line: onnx_file, vnn_lib_file, timeout_secs.