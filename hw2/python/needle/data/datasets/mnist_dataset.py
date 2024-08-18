from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import struct,gzip

def parse_mnist(image_filename, label_filename):
    img_file=gzip.open(image_filename,'rb')
    label_file=gzip.open(label_filename,'rb')
    img_file.read(4)
    label_file.read(8)
    cnt=struct.unpack('>I',img_file.read(4))[0]
    r,w=struct.unpack('>II',img_file.read(8))
    X=[]
    Y=[]
    for i in range(cnt):
        x=[]
        for j in range(r):
            for k in range(w):
                x.append(struct.unpack('>B',img_file.read(1)))
        X.append(x)
        Y.append(struct.unpack('>B',label_file.read(1)))
    img_file.close()
    label_file.close()
    X=np.array(X,np.uint8).reshape(-1,r*w).astype(np.float32)
    X=X/255.0
    Y=np.array(Y,np.uint8).reshape(cnt)
    return X,Y

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        self.transforms=transforms
        self.X,self.Y=parse_mnist(image_filename,label_filename)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        x=self.X[index].reshape(28,28,-1)
        if self.transforms !=None:
            x=self.apply_transforms(x)
        return x.reshape(-1,28*28),self.Y[index]
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.X.shape[0]
        ### END YOUR SOLUTION