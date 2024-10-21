#首先安装ray[tune]
#pip install "ray[tune]" torch torchvision

#设置Pytorch模型以进行调整：导入一些PyTorch和TorchVision模块来帮助我们创建模型并对其进行训练。此外，我们还将导入Ray Tune来帮助我们优化模型。
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

from ray import train, tune
from ray.tune.schedulers import ASHAScheduler

#定义一个将要训练的简单PyTorch模型，用__init__建立你的模型，然后实现前向传递。
#使用一个小型卷积神经网络，由一个2D卷积层、一个完全连接层和一个softmax函数组成。
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3)
        self.fc = nn.Linear(192, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 3))
        x = x.view(-1, 192)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


# 修改该参数的值 实现更快或更慢的训练
EPOCH_SIZE = 512
TEST_SIZE = 256


'''def train_func(model, optimizer, train_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx * len(data) > EPOCH_SIZE:
            return
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()'''


'''def test_func(model, data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            if batch_idx * len(data) > TEST_SIZE:
                break
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    return correct / total'''



#用Tune训练设置一个Tuner
#定义了一个函数，用于为多个epochs训练Pytorch模型。该函数将在一个单独的Ray Actor（进程）上执行，因此我们需要将模型的性能传达回Tune（位于主Python进程上）。
#在训练函数中调用train.report（），它将性能值发送回Tune。由于该函数是在单独的进程上执行的，确保该函数可由Ray序列化。
import os
import tempfile

from ray.train import Checkpoint

def train_mnist(config):
    mnist_transforms = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307, ), (0.3081, ))])

    train_loader = DataLoader(
        datasets.MNIST("~/data", train=True, download=True, transform=mnist_transforms),
        batch_size=64,
        shuffle=True)

    test_loader = DataLoader(
        datasets.MNIST("~/data", train=False, transform=mnist_transforms),
        batch_size=64,
        shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ConvNet()
    model.to(device)

    optimizer = optim.SGD(
        model.parameters(), lr=config["lr"], momentum=config["momentum"])
    for i in range(10):
        train_func(model, optimizer, train_loader)
        acc = test_func(model, test_loader)

        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            checkpoint = None
            if (i + 1) % 5 == 0:
                # This saves the model to the trial directory
                torch.save(
                    model.state_dict(),
                    os.path.join(temp_checkpoint_dir, "model.pth")
                )
                checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)

            # Send the current training result back to Tune
            train.report({"mean_accuracy": acc}, checkpoint=checkpoint)



#通过调用Tuner.fit来运行一个trials，并从学习率和动量的均匀分布中随机采样。
#search_space = {
config = {
    "lr": tune.sample_from(lambda spec: 10 ** (-10 * np.random.rand())),
    "momentum": tune.uniform(0.1, 0.9),
}

# Uncomment this to enable distributed execution
# `ray.init(address="auto")`

# Download the dataset first
datasets.MNIST("~/data", train=True, download=True)

tuner = tune.Tuner(
    train_mnist,
    param_space=config,
    #param_space=search_space,
)
results = tuner.fit()

#Tuner.fit返回一个ResultGrid对象。可以使用它来绘制此Trials的性能。
#dfs = {result.path: result.metrics_dataframe for result in results}
#[d.mean_accuracy.plot() for d in dfs.values()]


'''
#--------------------------------ASHRAcheduler---------------------------
#自适应连续减半的提前停止（ASHRAcheduler）
#将提前停止集成到优化过程中。使用ASHA，一种用于原则性早期停止的可扩展算法。
#在高维度上，ASHA终止了不太有希望的trials，并将更多的时间和资源分配给更有希望的trialsls。随着我们的优化过程变得更加高效，通过调整参数num_samples将搜索空间增加5倍。
ASHA在Tune中被实现为“trials调度器”。这些试验时间表可以提前终止不良试验、暂停试验、克隆试验和更改正在运行的trials超参数。

tuner = tune.Tuner(
    train_mnist,
    tune_config=tune.TuneConfig(
        num_samples=20,
        scheduler=ASHAScheduler(metric="mean_accuracy", mode="max"),
    ),
    param_space=search_space,
)
results = tuner.fit()

# Obtain a trial dataframe from all run trials of this `tune.run` call.
dfs = {result.path: result.metrics_dataframe for result in results}

# Plot by epoch
ax = None  # This plots everything on the same plot
for d in dfs.values():
    ax = d.mean_accuracy.plot(ax=ax, legend=False)

#可以使用 TensorBoard可视化结果
#tensorboard --logdir {logdir}

#---------------------------------使用搜索算法---------------------------
#除了TrialScheduler，还可以使用贝叶斯优化等智能搜索技术进一步优化超参数。可以使用Tune搜索算法。搜索算法利用优化算法来智能地导航给定的超参数空间。
from hyperopt import hp
from ray.tune.search.hyperopt import HyperOptSearch

space = {
    "lr": hp.loguniform("lr", -10, -1),
    "momentum": hp.uniform("momentum", 0.1, 0.9),
}

hyperopt_search = HyperOptSearch(space, metric="mean_accuracy", mode="max")

tuner = tune.Tuner(
    train_mnist,
    tune_config=tune.TuneConfig(
        num_samples=10,
        search_alg=hyperopt_search,
    ),
)
results = tuner.fit()

# To enable GPUs, use this instead:
# analysis = tune.run(
#     train_mnist, config=search_space, resources_per_trial={'gpu': 1})

#----------------------------Tune之后评估你的模型--------------------
best_result = results.get_best_result("mean_accuracy", mode="max")
with best_result.checkpoint.as_directory() as checkpoint_dir:
    state_dict = torch.load(os.path.join(checkpoint_dir, "model.pth"))

model = ConvNet()
model.load_state_dict(state_dict)





'''
