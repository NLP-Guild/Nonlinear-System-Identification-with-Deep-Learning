from models__ import TransformerForRegression
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

class myDataset(data.Dataset):
    def __init__(self, timestep:int):
        super(myDataset,self).__init__()
        raw_dataset_np = np.load('small_dataset.npz')['arr_0']
        preprocessed_dataset = []

        for i in raw_dataset_np:
            '''
            result: [[f1,x1,f2,x2,...,ft,xt], [kt,at]]
            '''
            for j in range(0, len(i) - timestep, timestep):
                input = i[j:j + timestep, 1:3].flatten()
                label = i[j + timestep - 1][-2:]
                sample = [input, label]

                preprocessed_dataset.append(sample)

        self.dataset = np.array(preprocessed_dataset)
    def __getitem__(self,idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

# collate function
def myfunc(batch_data):
    '''
    batch_data: Nx2
    '''
    resData = []
    resLabel = []
    for i in batch_data:
        resData.append(i[0])
        resLabel.append(i[1])
    resData = np.array(resData)
    resLabel = np.array(resLabel)
    return torch.tensor(resData,dtype=torch.float),torch.tensor(resLabel,dtype=torch.float)


if __name__ == '__main__':
    dataset = myDataset(int(0.05/0.001))
    # 划分训练集和测试集
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # Dataloader
    trainloader = data.DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=myfunc, drop_last=True)

    testloader = data.DataLoader(test_dataset, batch_size=32, shuffle=True, collate_fn=myfunc, drop_last=True)

    # for i in trainloader:
    #     trData, trLabel = i
    #     print(type(trData))
    #     print(type(trLabel))
    #     print(trData.size())
    #     print(trLabel.size())
    #     break

    # 创建网络模型
    model = TransformerForRegression(100)

    # 损失函数
    loss_fn = nn.MSELoss()

    # 优化器
    learning_rate = 1e-2
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # 训练的轮数
    epoch = 10

    train_epoch_loss = []
    valid_epoch_loss = []

    for i in range(epoch):
        print("-------第 {} 轮训练开始-------".format(i + 1))

        # 训练步骤开始
        model.train()
        train_step_loss = []
        step = 0
        for data in trainloader:
            input, targets = data
            outputs = model(input)  # 求模型的输出
            optimizer.zero_grad()  # 梯度清零
            loss = loss_fn(outputs, targets)  # 求loss
            train_step_loss.append(loss.item())
            step += 1
            if (step % 100 == 0):
                print(f'第{i + 1}轮第{step}训练step时的loss: {loss.item()}')

            # 优化器优化模型
            loss.backward()  # 求梯度
            optimizer.step()  # 更新参数
        train_epoch_loss.append(np.average(train_step_loss))

        # 测试步骤开始
        model.eval()
        valid_step_loss = []
        total_accuracy = 0  # 每一轮总的精确度
        with torch.no_grad():  # 不求梯度，不更新参数
            for data in testloader:
                input, targets = data
                outputs = model(input)
                loss = loss_fn(outputs, targets)
                valid_step_loss.append(loss.item())
                # accuracy = (outputs.argmax(1) == targets).sum()
                # total_accuracy = total_accuracy + accuracy

        print(f"第{i + 1}轮整体训练集上的Loss: {train_epoch_loss[-1]}")
        valid_epoch_loss.append(np.average(valid_step_loss))
        print(f"第{i + 1}轮整体测试集上的Loss: {valid_epoch_loss[-1]}")
        print(f"第{i + 1}轮整体测试集上的正确率: {(total_accuracy / test_size)}")

        # torch.save(model, "model_{}_epoch.pth".format(i))
        # print("模型已保存")

