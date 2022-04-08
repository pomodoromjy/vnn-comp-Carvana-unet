# @Time     :2022/3/31  15:15
# @Author   :majinyan
# @Software : PyCharm
import argparse
import numpy as np
import os
import re
import PIL.Image as Image

def get_sucess_images(path):
    list = []
    for filename in os.listdir(path):
        list.append(filename)
    return list

def img_preprocess(pil_img):
    img_ndarray = np.asarray(pil_img) / 255
    return img_ndarray

def write_vnn_spec(img_pre, imagename, epslion, dir_path, prefix="spec", data_lb=0, data_ub=1, n_class=1, mean=0.0, std=1.0, negate_spec=False,csv=''):
    for eps in epslion:
        x = Image.open(img_pre + imagename)
        x = np.array(x) / 255
        x_lb = np.clip(x - eps, data_lb, data_ub)
        x_lb = ((x_lb-mean)/std).reshape(-1)
        x_ub = np.clip(x + eps, data_lb, data_ub)
        x_ub = ((x_ub - mean) / std).reshape(-1)

        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        if np.all(mean==0.) and np.all(std==1.):
            spec_name = f"{prefix}_idx_{imagename}_eps_{eps:.5f}.vnnlib"
        else:
            existing_specs = os.listdir(dir_path)
            competing_norm_ids = [int(re.match(f"{prefix}_idx_{imagename}_eps_{eps:.5f}_n([0-9]+).vnnlib",spec).group(1)) for spec in existing_specs if spec.startswith(f"{prefix}_idx_{index}_eps_{eps:.5f}_n")]
            norm_id = 1 if len(competing_norm_ids) == 0 else max(competing_norm_ids)+1
            spec_name = f"{prefix}_idx_{imagename}_eps_{eps:.5f}_n{norm_id}.vnnlib"


        spec_path = os.path.join(dir_path, spec_name)

        with open(spec_path, "w") as f:
            f.write(f"; Spec for sample id {imagename} and epsilon {eps:.5f}\n")

            f.write(f"\n; Definition of input variables\n")
            for i in range(len(x_ub)):
                f.write(f"(declare-const X_{i} Real)\n")

            f.write(f"\n; Definition of output variables\n")
            for i in range(n_class):
                f.write(f"(declare-const Y_{i} Real)\n")

            f.write(f"\n; Definition of input constraints\n")
            for i in range(len(x_ub)):
                f.write(f"(assert (<= X_{i} {x_ub[i]:.8f}))\n")
                f.write(f"(assert (>= X_{i} {x_lb[i]:.8f}))\n")

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
    parser = argparse.ArgumentParser(description='VNN spec generator',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, help='random seed.',required=True)
    parser.add_argument('--epsilon', type=float, default=[0.012,0.015], help='The epsilon for L_infinity perturbation')
    parser.add_argument('--mean', nargs='+', type=float, default=0.0, help='the mean used to normalize the data with')
    parser.add_argument('--std', nargs='+', type=float, default=1.0,
                        help='the standard deviation used to normalize the data with')
    parser.add_argument('--csv', type=str, default="../Carvana-unet_instances.csv", help='csv file to write to')


    args = parser.parse_args()

    '''get the list of success images'''
    sucess_images_path = '../dataset/succeeds_mask'
    list = get_sucess_images(sucess_images_path)
    for index in range(len(list)):
        img_file_pre = r'../dataset/test_images/'
        mean = np.array(args.mean).reshape((1, -1, 1, 1)).astype(np.float32)
        std = np.array(args.std).reshape((1, -1, 1, 1)).astype(np.float32)
        #open image and normalize
        write_vnn_spec(img_file_pre, list[index], args.epsilon, dir_path='../specs/vnnlib', prefix='spec', data_lb=0,
                        data_ub=1, n_class=2, mean=mean, std=std, negate_spec=True,csv=args.csv)


if __name__ == "__main__":
    main()