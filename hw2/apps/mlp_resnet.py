import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
     return nn.Sequential(
        nn.Residual(
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                norm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(drop_prob),
                nn.Linear(hidden_dim, dim),
                norm(dim),
            )
        ),
        nn.ReLU(),
    )
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    return nn.Sequential(
        nn.Linear(dim, hidden_dim),
        nn.ReLU(),
        *[ResidualBlock(dim=hidden_dim,hidden_dim=hidden_dim//2,norm=norm,drop_prob=drop_prob) for i in range(num_blocks)],
        nn.Linear(hidden_dim, num_classes),
    ) 
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    loss_func= nn.SoftmaxLoss()
    error=0
    loss=0
    if opt is not None:
        model.train()
    else:
        model.eval()
    for x,y in dataloader:
        y_pred=model(x)
        batch_loss=loss_func(y_pred,y)
        if opt is not None:
            opt.reset_grad()
            batch_loss.backward()
            opt.step()
        loss+=batch_loss.numpy()*x.shape[0]
        y=y.numpy()
        y_pred=y_pred.numpy()
        y_pred=np.argmax(y_pred,axis=1)
        error+=np.sum(y_pred!=y)
    return error/len(dataloader.dataset),loss/len(dataloader.dataset)
    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    train_dataset = ndl.data.MNISTDataset(data_dir+"/train-images-idx3-ubyte.gz", data_dir+"/train-labels-idx1-ubyte.gz")
    test_dataset = ndl.data.MNISTDataset(data_dir+"/t10k-images-idx3-ubyte.gz", data_dir+"/t10k-labels-idx1-ubyte.gz")
    train_dataloader=ndl.data.DataLoader(train_dataset,batch_size,True)
    test_dataloader=ndl.data.DataLoader(test_dataset,batch_size)
    model=MLPResNet(784, hidden_dim)
    opt=optimizer(model.parameters(),lr=lr,weight_decay=weight_decay)
    for i in range(epochs):
        train_error,train_loss=epoch(train_dataloader,model,opt)
        test_error, test_loss = epoch(test_dataloader, model)
    return train_error,train_loss,test_error,test_loss
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
