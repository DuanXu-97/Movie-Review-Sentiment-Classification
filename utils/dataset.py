from torch.utils import data
import torch as t
import numpy as np

class MovieReviewDataset(data.Dataset):

    def __init__(self, root_path, config):

        self.config = config
        self.dataset = []
        self.labels = []

        with open(root_path, encoding='utf-8') as f:

            for line in f.readlines():
                try:
                    sp = line.strip().split()
                    _data = [config.word2id[word] for word in sp[1:]]
                    _data = _data + [0] * (config.max_seq_len - len(_data))
                    self.dataset.append(t.LongTensor(_data))
                    self.labels.append(t.LongTensor([1., 0.] if sp[0] == '1' else [0., 1.]))

                except Exception as e:
                    print(e)

    def __getitem__(self, index):
        return self.dataset[index], self.labels[index]

    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    from models.configs import AttTextCNNConfig
    dataset = MovieReviewDataset("../data/Dataset/train.txt", AttTextCNNConfig())
    print(dataset[0])







