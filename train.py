import ray
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
# from ray import tune
# from ray.tune import CLIReporter
# from ray.tune.schedulers import ASHAScheduler
from ray.util.sgd.torch import TorchTrainer
import torchvision.transforms as T

import torchvision

from ray.util.sgd.torch.resnet import ResNet50
import timeit
import numpy as np
from torch_v import ModelParallelResNet50, PipelineParallelResNet50
import matplotlib.pyplot as plt
num_classes = 1000

def train_simple_model(model, ray1, multi_gpu, single_gpu):
    k = torch.cuda.device_count()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # train_loader = torch.utils.data.DataLoader(train_set, bat/ch_size=32, shuffle=True)
    stmt = "train_parallel(model)"
    num_repeat = 10

    if ray1:

        ray.init()
        def optimizer_creator(model, config):
            """Returns an optimizer (or multiple)"""
            return torch.optim.SGD(model.parameters(), lr=config["lr"])

        trainer = TorchTrainer(
            model_creator=ResNet50,  # A function that returns a nn.Module
            data_creator=data_creator,  # A function that returns dataloaders
            optimizer_creator=optimizer_creator,  # A function that returns an optimizer
            loss_creator=torch.nn.CrossEntropyLoss,  # A loss function
            config={"lr": 0.001, "batch": 64},  # parameters
            num_workers=2,  # amount of parallelism
            use_gpu=torch.cuda.is_available(),
            use_tqdm=True)
        stats = trainer.train()
        print(trainer.validate())
        torch.save(trainer.state_dict(), "checkpoint.pt")
        trainer.shutdown()
        print("success!")
    if single_gpu:
        # criterion = nn.CrossEntropyLoss().cuda()
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        # model = model.cuda()
        # model.train()
        setup = "import torchvision.models as models;" + \
                "model = models.resnet50(num_classes=num_classes).to('cuda:0')"
        rn_run_times = timeit.repeat(
            stmt, setup, number=1, repeat=num_repeat, globals=globals())
        rn_mean = np.mean(rn_run_times)
        rn_std = np.std(rn_run_times)
        print('single'+str(rn_mean))
        # with torch.profiler.profile(
        #         schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        #         on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/resnet18'),
        #         record_shapes=True,
        #         profile_memory=True,
        #         with_stack=True
        # ) as prof:
        #     for step, batch_data in enumerate(train_loader):
        #         if step >= (1 + 1 + 3) * 2:
        #             break
        #         train(criterion, optimizer, device, model, batch_data)
        #         prof.step()


    if multi_gpu:
        setup = "model = ModelParallelResNet50()"
        mp_run_times = timeit.repeat(
            stmt, setup, number=1, repeat=num_repeat, globals=globals())
        mp_std = np.std(mp_run_times)
        setup = "model = PipelineParallelResNet50()"
        pp_run_times = timeit.repeat(
            stmt, setup, number=1, repeat=num_repeat, globals=globals())
        pp_mean, mp_mean = np.mean(pp_run_times), np.mean(mp_run_times)
        pp_std= np.std(pp_run_times)
        print('parallel'+str(mp_mean))
        print('parallel+pipeline'+str(pp_mean))

    plot([mp_mean, rn_mean, pp_mean],
         [mp_std, rn_std, pp_std],
         ['Model Parallel', 'Single GPU', 'Pipelining Parallel'],
         'compare_c1.png')
# def train_rl(ray):
#     NotImplemented

# @ray.remote
# def train_ray():
#     ray.util.


def data_creator(config):

    transform = T.Compose(
        [T.Resize(224),
         T.ToTensor(),
         T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config["batch"], shuffle=True)
    validation_loader = torch.utils.data.DataLoader(train_set, batch_size=config["batch"], shuffle=True)
    return train_loader, validation_loader

def train_parallel(model):
    num_batches = 3
    batch_size = 64
    image_w = 128
    image_h = 128
    num_classes = 1000
    model.train(True)
    config = {"lr": 0.001, "batch": 64}
    # train_loader = torch.utils.data.DataLoader(train_set, bat / ch_size = 32, shuffle = True)
    tl = data_creator(config)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    one_hot_indices = torch.LongTensor(batch_size) \
                           .random_(0, num_classes) \
                           .view(batch_size, 1)
    dataloader = tl[0]
    # for step, data in dataloader:
    for _ in range(num_batches):
        inputs = torch.randn(batch_size, 3, image_w, image_h)
        labels = torch.zeros(batch_size, num_classes) \
                      .scatter_(1, one_hot_indices, 1)
        # inputs, labels = data[0].unsqueeze(-1), data[1].unsqueeze(-1)
        optimizer.zero_grad()
        outputs = model(inputs.to('cuda:0'))
        labels = labels.to(outputs.device)
        loss_fn(outputs, labels).backward()
        optimizer.step()

def train(criterion, optimizer, device, model, data):
    inputs, labels = data[0].to(device=device), data[1].to(device=device)
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss




def plot(means, stds, labels, fig_name):
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(means)), means, yerr=stds,
           align='center', alpha=0.5, ecolor='red', capsize=10, width=0.6)
    ax.set_ylabel('ResNet50 Execution Time (Second)')
    ax.set_xticks(np.arange(len(means)))
    ax.set_xticklabels(labels)
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close(fig)

# @ray.remote
# def
