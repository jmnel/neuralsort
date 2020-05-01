import os.path
import torch
import pickle

DATA_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                         '..', 'data', 'mnist_stitched.pkl')


class StitchedMNIST(torch.utils.data.Dataset):

    def __init__(self, file_path, transform=None):
        super().__init__()

#        pickle.dump(dataset, open(os.path.join(
#            DATA_PATH, 'mnist_stitched.pkl'), 'wb'))
        self.transform = transform
        self.data = pickle.load(open(file_path, 'rb'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = self.data[idx][0].reshape(112, 28)
        label = self.data[idx][1]

        s = str(label)

        s = tuple(int(x) for x in s)

        s = torch.FloatTensor(s)
#        for k in range(4):
#            for l in range(10):
#                onehot[k*10 + l] = 1. if l == int(s[0]) else 0.
        #        onehot[0] = 1 >> int(s[0])
        #        onehot[1] = 1 >> int(s[1])
        #        onehot[2] = 1 >> int(s[2])
        #        onehot[3] = 1 >> int(s[3])

        #        onehot = torch.zeros(36)

        #        s = str(label)
        #        for k in range(4):
        #            onehot[k + int(s[k])] = 1.0

        if self.transform:
            img = self.transform(img)

        sample = (img, s)
        return sample
