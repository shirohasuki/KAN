import os
import torch
import torchvision
import torchvision.transforms as transforms

class DatasetManager:
    def __init__(self, batch_size=128, num_workers=2, device=None):
        """
        初始化 DatasetManager
        :param batch_size: 数据加载器的批大小
        :param num_workers: 数据加载器的线程数
        :param device: 使用的设备 (e.g., 'cuda:0', 'cpu')
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device or torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # 获取当前脚本的目录作为路径基准
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # 使用基准路径定义 CIFAR-10 和 MNIST 数据路径
        self.cifar_path = os.path.join(current_dir, 'CIFAR10')
        self.mnist_path = os.path.join(current_dir, 'mnist')

        print(f"CIFAR-10 path: {self.cifar_path}")
        print(f"MNIST path: {self.mnist_path}")

    def mnist_dataset(self):
        """
        加载 MNIST 数据集
        :return: trainloader, testloader
        """
        transform = transforms.Compose([
            transforms.RandomRotation(15),  # 随机旋转
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize((0.5,), (0.5,))  # 单通道归一化
        ])

        trainset = torchvision.datasets.MNIST(
            root=self.mnist_path, train=True, download=True, transform=transform
        )
        testset = torchvision.datasets.MNIST(
            root=self.mnist_path, train=False, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        )

        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )

        return trainloader, testloader

    def cifar_dataset(self):
        """
        加载 CIFAR-10 数据集
        :return: trainloader, testloader
        """
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),  # 随机裁剪，带有填充
            transforms.RandomHorizontalFlip(),  # 随机水平翻转
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # 颜色抖动
            transforms.RandomRotation(15),  # 随机旋转
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        trainset = torchvision.datasets.CIFAR10(
            root=self.cifar_path, train=True, download=True, transform=transform_train
        )
        testset = torchvision.datasets.CIFAR10(
            root=self.cifar_path, train=False, download=True, transform=transform_test
        )

        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )

        return trainloader, testloader


# 示例用法
if __name__ == "__main__":
    dataset_manager = DatasetManager(batch_size=128)

    # 加载 MNIST 数据集
    mnist_trainloader, mnist_testloader = dataset_manager.mnist_dataset()
    print("MNIST Dataset Loaded: ")
    print(f"Train batches: {len(mnist_trainloader)}, Test batches: {len(mnist_testloader)}")

    # 加载 CIFAR-10 数据集
    cifar_trainloader, cifar_testloader = dataset_manager.cifar_dataset()
    print("CIFAR-10 Dataset Loaded: ")
    print(f"Train batches: {len(cifar_trainloader)}, Test batches: {len(cifar_testloader)}")

    # 将一个批次数据加载到指定设备
    dataiter = iter(mnist_trainloader)
    images, labels = next(dataiter)
    images, labels = images.to(dataset_manager.device), labels.to(dataset_manager.device)
    print(f"Images device: {images.device}, Labels device: {labels.device}")
