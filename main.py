def main():
    import torch.nn as nn
    import torch.nn.functional as F
    import torch
    from torch.autograd import Variable
    import torch.optim as optim
    
    class MyFirstNet(nn.Module):
        def __init__(self):
            super(MyFirstNet, self).__init__()
            self.conv1 = nn.Conv2d(1, 6, 5)   # 1 input image channels, 6 output channels
            self.conv2 = nn.Conv2d(6, 16, 5)   # 6 is the number of kernels, 5 is the size of kernel
            self.fc1 = nn.Linear(16*5*5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
            x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
            x = x.view(-1, self.num_flat_features(x)) # caculate the number of features
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
        def num_flat_features(self, x):
            size = x.size()[1:]
            num_features = 1
            for i in size:
                num_features = num_features * i
            return num_features

    net = MyFirstNet()
    input = Variable(torch.randn(1, 1, 32, 32))   # 1*1*32*32tensor
    output = net(input)
    # target = torch.randn(10)  # a dummy target, for example
    # target = target.view(1, -1)
    target = torch.unsqueeze(Variable(torch.arange(1, 11)),0)  # use unsqueeze function to add a new dim
    target = target.type(torch.float32)   # transform the datatype of target
    criterion = nn.MSELoss()
    loss = criterion(output, target)
    #print(net.conv1.bias.grad)
    net.zero_grad()

    # training procedure
    optimizer = optim.Adam(net.parameters(), lr = 0.001)
    loss.backward()  # pay attention to the dtype of "target",dtype = float32, can not be int
    print(net.conv1.bias)
    learning_rate = 0.0005
    for i in range(1000):
        for f in net.parameters():
            f.data.sub_(f.grad.data * learning_rate)
        output = net(input)
        loss = criterion(output, target)
        print(i, "-th ""loss = ", loss)
        if(loss < 5):
            break
    print(list(net.parameters()))




if __name__ == "__main__":
    main()