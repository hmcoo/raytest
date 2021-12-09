import argparse
import train
import torchvision
import torch
import torch_v


def set_parser():
    parser =argparse.ArgumentParser()

    # compare ray and other libraries' multiprocessing
    parser.add_argument('--ray_multi', type=bool, default=False)
    parser.add_argument('--pytorch_multi', type=bool, default=True)
    parser.add_argument('--mxnet_multi', type=bool, default=False)

    # compare ray and single processing
    parser.add_argument('--ray_single', type=bool, default=True)
    parser.add_argument('--pytorch', type=bool, default=True)
    parser.add_argument('--mxnet', type=bool, default=False)

    # compare ray and cpu
    parser.add_argument('--ray_cpu', type=bool, default=False)
    parser.add_argument('--cpu', type=bool, default=False)

    # test simple deep learning models
    parser.add_argument('--resnet', type=bool, default=False)
    parser.add_argument('--inception', type=bool, default=False)
    parser.add_argument('--vgg', type=bool, default=False)

    # rl algorithms
    parser.add_argument('--taxi', type=bool, default=False)
    parser.add_argument('--frozenlake', type=bool, default=False)
    parser.add_argument('--cartpole', type=bool, default=False)
    parser.add_argument('--mountaincar', type=bool, default=False)

    #hyperparameter tuning

    #seed
    parser.add_argument('--seed', type=int, default=1)

    #training
    parser.add_argument('--gpu', default='0,1')
    parser.add_argument('--workers', type=int, default=4)


    return parser.parse_args()


if __name__ == '__main__':
    parser = set_parser()
    resnet = torch_v.resnet()
    # torchvision.datasets.CIFAR10()
    train.train_simple_model(resnet, parser.ray_single, parser.pytorch_multi, parser.pytorch)
    print("Done")