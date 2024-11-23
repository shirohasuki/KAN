import os
import torch

class WeightManager:
    @staticmethod
    def save_model(model, optimizer, epoch, dir_name, file_name):
        """
        保存模型及优化器状态
        :param model: 模型对象
        :param optimizer: 优化器对象
        :param epoch: 当前训练的 epoch 数
        :param path: 保存路径
        """
        base_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(base_dir, dir_name)
        # 创建文件夹（如果不存在）
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, file_name)

        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        torch.save(state, save_path)
        print(f"Model saved to {save_path}")

    @staticmethod
    def load_model(model, optimizer, dir_name, file_name, device):
        """
        加载模型及优化器状态
        :param model: 模型对象
        :param optimizer: 优化器对象
        :param path: 加载路径
        :param device: 目标设备 (e.g., 'cuda' or 'cpu')
        :return: 加载后的模型, 优化器, 起始 epoch
        """
        # 获取当前文件所在目录
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # 目标文件夹路径
        load_path = os.path.join(base_dir, dir_name, file_name)

        checkpoint = torch.load(load_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Model loaded from {load_path}, starting from epoch {start_epoch}")
        return model, optimizer, start_epoch
    
    @staticmethod
    def list_pth_files(dir_name):
        """
        列出给定文件夹下的所有 .pth 文件
        :param name: 文件夹名称
        :return: .pth 文件列表
        """
        # 获取当前文件所在目录
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # 目标文件夹路径
        dir_name = os.path.join(base_dir, dir_name)

        # 检查文件夹是否存在
        if not os.path.exists(dir_name):
            print(f"Folder '{dir_name}' does not exist.")
            return []

        # 遍历文件夹，寻找 .pth 文件
        pth_files = [f for f in os.listdir(dir_name) if f.endswith('.pth')]

        if pth_files:
            print(f"Found {len(pth_files)} .pth file(s) in '{dir_name}':")
            for file in pth_files:
                print(f" - {file}")
        else:
            print(f"No .pth files found in '{dir_name}'.")

        return pth_files
