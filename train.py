import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
from multiprocessing import freeze_support
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import numpy as np
from Loss import CtdetLoss
from torch.utils.data import DataLoader
from dataset import ctDataset
import time, subprocess
import torch.nn as nn
# 设置随机种子
# seed = 42
# torch.manual_seed(seed)  # 设置PyTorch随机种子
# np.random.seed(seed)     # 设置NumPy随机种子
# torch.backends.cudnn.deterministic = True  # 如果使用GPU，确保cuDNN在确定性计算方面的一致
from backbone.resnet import ResNet
# from resnet_dcn import ResNet
# from backbone.dlanet import DlaNet
# from backbone.dlanet import DlaNet
from backbone.dlanet_dcn import DlaNet


def get_gpu_memory_usage(device_id=0):
    """Returns GPU memory used in MB for the specified device using nvidia-smi."""
    try:
        # Run nvidia-smi to query memory usage
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'])
        # Parse the output and extract memory usage (in MB)
        gpu_memory = [int(x) for x in result.strip().split(b'\n')][device_id]
        return gpu_memory
    except subprocess.CalledProcessError as e:
        print(f"Failed to execute nvidia-smi: {e}")
        return None
    except IndexError:
        print(f"Invalid device ID: {device_id}")
        return None
    except Exception as e:
        print(f"Error while fetching GPU memory: {e}")
        return None

def check_cuda_memory(device_id=0, threshold=5000):
    """Continuously checks if GPU memory usage is below a threshold."""
    try:
        while True:
            gpu_memory = get_gpu_memory_usage(device_id)
            if gpu_memory is not None:
                print(f"GPU Memory Used: {gpu_memory} MB")

                if gpu_memory < threshold:
                    print("GPU Memory is below the threshold. Exiting...")
                    break

            time.sleep(20)  # Wait for 20 seconds before the next check

    except KeyboardInterrupt:
        print("Detection stopped by user.")

def run():
    check_cuda_memory(device_id=0, threshold=5000)
    loss_weight = {'hm_weight': 1, 'wh_weight': 0.1, 'ang_weight': 1, 'reg_weight': 0.5}
    device = torch.device("cuda")
    gpus = [0, 1, 2,3]
    learning_rate = 0.0002
    num_epochs = 500
    model = DlaNet(34)
    model= nn.DataParallel(model.cuda(), device_ids=gpus, output_device=gpus[0])
    dataset_name='MPD'
    checkpoint_path = './best_train_roi_model.pth'
    if os.path.exists(checkpoint_path):
        print("model '{}' exists.".format(checkpoint_path))
        # 注意，以相同的顺序加载保证环境一致
        checkpoint = torch.load(checkpoint_path)
        loaded_keys = set(model.state_dict().keys())
        checkpoint_keys = set(checkpoint.keys())
        if loaded_keys == checkpoint_keys:
            model.load_state_dict(checkpoint)
            print("Model loaded successfully.")
        else:
            print("Warning: Model keys do not match exactly.")
    else:
        print("model '{}' does not exist. Start form scratch.".format(checkpoint_path))
    criterion = CtdetLoss(loss_weight)
    model.train()
    params=[]
    params_dict = dict(model.named_parameters())
    for key,value in params_dict.items():
        params += [{'params':[value],'lr':learning_rate}]
    # print(params)
    # 如果你需要更高的训练稳定性，或者在训练特定的大规模深度学习模型时 SGD
    # 如果你希望获得更快的收敛速度，并且对超参数调整不那么敏感 Adam
    # 初期尝试: 使用Adam开始，观察其收敛情况。如果训练过程快速且稳定，Adam可能是一个很好的选择。
    # 精细调优: 如果模型训练表现良好但精度尚未达标，尝试使用SGD进行更精细的调整，尤其是在模型已经较大且学习率调优困难时。
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    # optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=1e-4)
    if(dataset_name=='Tongji'):
        patience=30
        batch_size = 8
    if(dataset_name=='MPD'):
        patience=10
        batch_size=16
    if (dataset_name == 'CASIA'):
        patience = 30
        batch_size = 16
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience,min_lr=1e-6)
    checkpoint_path = './{}_optimizer_scheduler.pth'.format(dataset_name)
    start_epoch=0
    if os.path.exists(checkpoint_path):
        print("Checkpoint file '{}' exists.".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch=checkpoint['epoch']
    else:
        print("Checkpoint file '{}' does not exist. Start form scratch.".format(checkpoint_path))
    train_dataset = ctDataset(data_name=dataset_name, split='palmprint')
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=4)  # num_workers是加载数据（batch）的线程数目
    test_dataset = ctDataset(data_name=dataset_name,split='val')
    test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=True,num_workers=4)
    print('the dataset has %d images' % (len(train_dataset)))
    best_test_loss = np.inf
    best_train_loss = np.inf
    for epoch in range(start_epoch,num_epochs):
        # 会启用诸如dropout和batchnormalization的训练特性。
        model.train()
        # 打印当前学习率 对于多学习率
        # for param_group in optimizer.param_groups:
        #     print(f"Epoch {epoch + 1}, Learning Rate: {param_group['lr']}")
        print(f"Epoch [{epoch+1}/{num_epochs}], Current learning Rate: {optimizer.param_groups[0]['lr']}")
        # if epoch == 90:
        #     learning_rate= learning_rate * 0.1
        # if epoch == 120:
        #     learning_rate= learning_rate * (0.1 ** 2)
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = learning_rate
        total_loss = 0.
        # 注意： Iter [80/80] 是迭代次数 total_num/batch_size
        for i, sample in enumerate(train_loader):
            for k in sample:
                sample[k] = sample[k].to(device=device, non_blocking=True)#更快；自动分发GPU
            pred = model(sample['input'])
            loss = criterion(pred, sample)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % 5 == 0:
                print ('Epoch [%d/%d], Iter [%d/%d] Current Loss: %.4f, Epoch average_loss: %.4f'
                %(epoch+1, num_epochs, i+1, len(train_loader), loss.data, total_loss / (i+1)))
        if(best_train_loss>((total_loss)/ len(train_loader))):
            best_train_loss=(total_loss)/ len(train_loader)
            torch.save(model.state_dict(), 'best_train_roi_model.pth')
        validation_loss = 0.0
        model.eval()
        for i, sample in enumerate(test_loader):
            for k in sample:
                # 使得数据转移操作更加高效，尤其是在使用异步数据传输时。
                sample[k] = sample[k].to(device=device, non_blocking=True)
            pred = model(sample['input'])
            loss = criterion(pred, sample)
            validation_loss += loss.item()
        print('Epoch [%d/%d], Total Test Loss: %.4f, average_loss: %.4f'
              % (epoch + 1, num_epochs,validation_loss, validation_loss /len(test_loader)))

        validation_loss /= len(test_loader)
        scheduler.step(validation_loss)
        # 保存优化器和调度器状态
        torch.save({
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, './{}_optimizer_scheduler.pth'.format(dataset_name))
        if best_test_loss > validation_loss:
            best_test_loss = validation_loss
            print('get best test loss %.5f' % best_test_loss)
            torch.save(model.state_dict(),'best_roi_model.pth')
    torch.save(model.state_dict(),'last_roi_model.pth')
if __name__ == '__main__':
    freeze_support()  # 这行在Windows系统是必要的运行多进程代码
    run()  # 或者是你用来开始训练的函数