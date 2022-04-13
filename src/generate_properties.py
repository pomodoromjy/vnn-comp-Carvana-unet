# @Time     :2022/3/31  15:15
# @Author   :majinyan
# @Software : PyCharm
import argparse
import numpy as np
import os
import re
import PIL.Image as Image
import sys
import torch
import numpy.random as random

def get_random_images(path,random,length,seed):
    list = []
    for filename in os.listdir(path):
        list.append(filename)
    random_sel_list = []
    if random:
        np.random.seed(seed)
        random_sel_list = np.random.choice(list,length,replace=False)
    else:
        for index in range(len(list)):
            while len(random_sel_list) < length:
                random_sel_list.append(list[index])
    return random_sel_list

def gt_mask_process(img_ndarray):
    for i in range(len(img_ndarray)):
        for j in range(len(img_ndarray[0])):
            if img_ndarray[i][j] != 0:
                img_ndarray[i][j] = 1
    return img_ndarray

def write_vnn_spec(img_pre, gt_mask_pre, imagename, epslion, dir_path, prefix="spec", data_lb=0, data_ub=1, n_class=1, mean=0.0, std=1.0, negate_spec=False,csv=''):
    for eps in epslion:
        x = Image.open(img_pre + imagename)
        x = np.array(x) / 255
        x_lb = np.clip(x - eps, data_lb, data_ub)
        x_lb = ((x_lb-mean)/std).reshape(-1)
        x_ub = np.clip(x + eps, data_lb, data_ub)
        x_ub = ((x_ub - mean) / std).reshape(-1)

        maskname = imagename.split('.')[0] + '_mask.gif'
        gt_mask = Image.open(gt_mask_pre + maskname)
        gt_mask = np.asarray(gt_mask)
        gt_mask = gt_mask_process(gt_mask)
        gt_mask = gt_mask.reshape(-1)

        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        spec_name = f"{prefix}_idx_{imagename}_eps_{eps:.5f}.vnnlib"
        spec_path = os.path.join(dir_path, spec_name)

        with open(spec_path, "w") as f:
            f.write(f"; Spec for sample id {imagename} and epsilon {eps:.5f}\n")

            f.write(f"\n; Definition of input variables(image)\n")
            for i in range(len(x_ub)):
                f.write(f"(declare-const X_{i} Real)\n")

            f.write(f"\n; Definition of input variables(ground truth)\n")
            for i in range(len(gt_mask)):
                f.write(f"(declare-const GT_{i} Real)\n")

            f.write(f"\n; Definition of output variables\n")
            for i in range(n_class):
                f.write(f"(declare-const Y_{i} Real)\n")

            f.write(f"\n; Definition of input constraints(image)\n")
            for i in range(len(x_ub)):
                f.write(f"(assert (<= X_{i} {x_ub[i]:.8f}))\n")
                f.write(f"(assert (>= X_{i} {x_lb[i]:.8f}))\n")

            f.write(f"\n; Definition of input constraints(ground truth)\n")
            for i in range(len(gt_mask)):
                f.write(f"(assert (= GT_{i} {gt_mask[i]:.8f}))\n")

            f.write(f"\n; Definition of output constraints\n")
            if negate_spec:
                for i in range(n_class-1):
                    f.write(f"(assert (<= Y_1 1166))\n")
            else:
                f.write(f"(assert (or\n")
                for i in range(n_class):
                    f.write(f"\t(and (>= Y_1 1166))\n")
                f.write(f"))\n")
    csv = csv
    if not os.path.exists(csv):
        os.system(r"touch {}".format(csv))
    csvFile = open(csv, "w")
    network_path = '../net/onnx/'
    vnnlib_path = '../specs/vnnlib/'
    timeout = 300
    for network in os.listdir(network_path):
        network = os.path.join(network_path, network)
        for vnnLibFile in os.listdir(vnnlib_path):
            print(f"{network},{vnnLibFile},{timeout}", file=csvFile)
    csvFile.close()
    return spec_name

def main():
    seed = int(sys.argv[1])
    mean = 0.0
    std = 1.0
    epsilon = [0.012,0.015]
    csv = "../Carvana-unet_instances.csv"

    '''get the list of success images'''
    sucess_images_path = '../dataset/succeeds_mask'
    list = get_random_images(sucess_images_path,random=True,length=40,seed=seed)
    for index in range(len(list)):
        img_file_pre = r'../dataset/test_images/'
        gt_mask_pre = '../dataset/test_masks/'
        mean = np.array(mean).reshape((1, -1, 1, 1)).astype(np.float32)
        std = np.array(std).reshape((1, -1, 1, 1)).astype(np.float32)
        #open image and normalize
        write_vnn_spec(img_file_pre,gt_mask_pre, list[index], epsilon, dir_path='../specs/vnnlib', prefix='spec', data_lb=0,
                        data_ub=1, n_class=2, mean=mean, std=std, negate_spec=True,csv=csv)


if __name__ == "__main__":
    main()