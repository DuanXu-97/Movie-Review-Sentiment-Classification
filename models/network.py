import torch as t
from torch import nn
from torch.nn import functional as F


class BasicModule(nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()

    def load(self, path):
        self.load_state_dict(t.load(path))

    def save(self, path):
        t.save(self.state_dict(), path)


class TextCNN(BasicModule):

    def __init__(self, config):
        super(TextCNN, self).__init__()

        self.config = config
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False, padding_idx=0)
        else:
            self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=0)

        self.conv_layers = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, self.embedding.weight.size(1))) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)

        out = []
        for conv in self.conv_layers:
            _x = F.relu(conv(x)).squeeze(3)
            _x = F.max_pool1d(_x, _x.size(2)).squeeze(2)
            out.append(_x)

        x = t.cat(out, 1)
        x = self.dropout(x)
        logits = self.fc(x)
        output = F.softmax(logits, dim=1)

        return logits, output


class AttTextCNN(BasicModule):

    def __init__(self, config):
        super(AttTextCNN, self).__init__()

        self.config = config
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False, padding_idx=0)
        else:
            self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=0)

        self.conv_layers = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, self.embedding.weight.size(1))) for k in config.filter_sizes])

        self.attention_w = nn.Parameter(t.empty(config.num_filters * len(config.filter_sizes), config.attention_size))
        self.attention_b = nn.Parameter(t.zeros(config.attention_size))
        self.attention_uw = nn.Parameter(t.empty(config.attention_size, 1))
        nn.init.normal_(self._parameters['attention_w'], mean=0.0, std=0.1)
        nn.init.normal_(self._parameters['attention_uw'], mean=0.0, std=0.1)

        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def attention(self, x):
        u_t = t.tanh(t.mm(x, self.attention_w) + self.attention_b)
        z_t = t.mm(u_t, self.attention_uw)
        alpha = F.softmax(z_t, dim=-1)
        out = x * alpha
        return out

    def forward(self, x):
        x = self.embedding(x)
        x = x.clone().detach()
        x = x.unsqueeze(1)
        out = []
        for conv in self.conv_layers:
            _x = F.relu(conv(x)).squeeze(3)
            _x = F.max_pool1d(_x, _x.size(2)).squeeze(2)
            out.append(_x)

        x = t.cat(out, 1)
        x = self.attention(x)
        x = self.dropout(x)
        logits = self.fc(x)
        output = F.softmax(logits, dim=1)

        return logits, output


if __name__ == '__main__':
    from models.configs import AttTextCNNConfig
    from utils.dataset import MovieReviewDataset
    from torch.utils.data import DataLoader
    from utils.preprocess import *
    from torch.autograd import Variable

    config = AttTextCNNConfig()
    config.word2id = build_word2id([config.train_path, config.validation_path, config.test_path])
    config.embedding_pretrained = t.from_numpy(build_word2vec(config.embedding_pretrained_path, config.word2id))
    config.max_seq_len = get_max_len([config.train_path, config.validation_path, config.test_path])

    train_set = MovieReviewDataset(root_path=config.train_path, config=config)
    train_dataloader = DataLoader(train_set, config.batch_size,
                                  shuffle=False,
                                  num_workers=config.num_workers)
    model = AttTextCNN(config)

    for _iter, (train_data, train_target) in enumerate(train_dataloader):
        train_data = np.array([data.numpy() for data in train_data])
        input = Variable(t.from_numpy(train_data))
        logits, output = model(input)
        break




