#多层全连接神经网络
import torch
from torch import nn,optim
from torch.autograd import Variable
import net

#超参数
learn_rate=1e-2
epoch_size=700


# 获得训练数据 - train.csv
import csv
with open('./data/train.csv') as f :
    lines = csv.reader(f)
    label, attr = [], []
    for line in lines :
        if lines.line_num == 1 :
            continue
        label.append(int(line[0]))
        attr.append([float(j) for j in line[1:]])
    # 将数据分为 60(epoches) * 700(rows) 的数据集
    epoches = []
    for i in range(0, len(label), epoch_size):
        torch_attr = torch.FloatTensor(attr[i: i + epoch_size])
        torch_label = torch.LongTensor(label[i: i + epoch_size])
        epoches.append((torch_attr, torch_label))


model=net.simpleNet(28*28,300,100,10)

criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(),lr=learn_rate)

#开始训练
def train():
    epoch_num, loss_sum, cort_num_sum = 0, 0.0, 0
    for epoch in epoches :
            epoch_num += 1
            inputs = Variable(epoch[0])
            target = Variable(epoch[1])

            output = model(inputs)
            loss = criterion(output, target)
            # reset gradients
            optimizer.zero_grad()
            # backward pass
            loss.backward()
            # update parameters
            optimizer.step()

            # get training infomation
            loss_sum += loss.data[0]
            _, pred = torch.max(output.data, 1)


            num_correct = torch.eq(pred, epoch[1]).sum()
            cort_num_sum += num_correct

    loss_avg = loss_sum /float(epoch_num)
    cort_num_avg = cort_num_sum / float(epoch_num) /float( epoch_size)
    return loss_avg,cort_num_avg

# 对所有数据跑300遍模型
loss, correct = [], []
training_time = 300
for i in range(1, training_time + 1) :
    loss_avg, correct_num_avg = train()
    loss.append(loss_avg)
    if i< 20 or i % 20 == 0 :
        print('--- train time {} ---'.format(i))
        print('average loss = {:.4f}'.format(loss_avg))
        print('average correct number = {:.4f}'.format(correct_num_avg))
    correct.append(correct_num_avg)

