import sys
sys.path.append('../')
import os
import re
import time
import argparse
import torch as t
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchnet import meter
from models import configs
from models import network
from utils.dataset import MovieReviewDataset
from utils.visualize import Visualizer
from utils.preprocess import *
from torch.autograd import Variable


def train(args):
    vis = Visualizer()

    config = getattr(configs, args.model + 'Config')()

    config.word2id = build_word2id([config.train_path, config.validation_path, config.test_path])
    config.embedding_pretrained = t.from_numpy(build_word2vec(config.embedding_pretrained_path, config.word2id))
    config.max_seq_len = get_max_len([config.train_path, config.validation_path, config.test_path])

    train_set = MovieReviewDataset(root_path=config.train_path, config=config)
    validation_set = MovieReviewDataset(root_path=config.validation_path, config=config)

    train_dataloader = DataLoader(train_set, config.batch_size,
                                  shuffle=True,
                                  num_workers=config.num_workers)
    validation_dataloader = DataLoader(validation_set, config.batch_size,
                                       shuffle=False,
                                       num_workers=config.num_workers)

    model = getattr(network, args.model)(config).eval()

    if args.load_model_path:
        model.load(args.load_model_path)
    if args.use_gpu:
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=1e-5)

    train_loss_meter, validation_loss_meter = meter.AverageValueMeter(), meter.AverageValueMeter()
    train_confusion_matrix, validation_confusion_matrix = meter.ConfusionMeter(config.num_classes), meter.ConfusionMeter(config.num_classes)

    best_validation_loss = 1e5
    best_epoch = 0
    dist_to_best = 0

    time_begin = time.clock()

    for epoch in range(config.epoch):

        # train
        model.train()
        train_loss_meter.reset()
        train_confusion_matrix.reset()

        for _iter, (train_data, train_target) in enumerate(train_dataloader):

            optimizer.zero_grad()
            train_data = t.from_numpy(np.array([data.numpy() for data in train_data]))
            train_target = t.max(train_target, 1)[1]

            if args.use_gpu:
                train_data = train_data.cuda()
                train_target = train_target.cuda()

            train_data, train_target = Variable(train_data), Variable(train_target)
            train_logits, train_output = model(train_data)
            train_loss = criterion(train_logits, train_target)
            train_loss.backward()
            optimizer.step()

            train_loss_meter.add(train_loss.item())
            train_confusion_matrix.add(train_logits.data, train_target.data)

            if _iter % config.print_freq == 0:
                vis.plot('train_loss', train_loss_meter.value()[0])
        model.save(path=os.path.join(args.ckpts_dir, 'model_{0}.pth'.format(str(epoch))))

        # validation
        model.eval()
        validation_loss_meter.reset()
        validation_confusion_matrix.reset()

        for _iter, (validation_data, validation_target) in enumerate(validation_dataloader):

            validation_data = t.from_numpy(np.array([data.numpy() for data in validation_data]))
            validation_target = t.max(validation_target, 1)[1]

            if args.use_gpu:
                validation_data = validation_data.cuda()
                validation_target = validation_target.cuda()

            validation_data, validation_target = Variable(validation_data), Variable(validation_target)

            validation_logits, validation_output = model(validation_data)
            validation_loss = criterion(validation_logits, validation_target)

            validation_loss_meter.add(validation_loss.item())
            validation_confusion_matrix.add(validation_logits.detach().squeeze(), validation_target.type(t.LongTensor))

        validation_cm = validation_confusion_matrix.value()
        validation_accuracy = 100. * (validation_cm.diagonal().sum()) / (validation_cm.sum())

        vis.plot('validation_accuracy', validation_accuracy)

        vis.log("epoch:{epoch}, train_loss:{train_loss}, train_cm:{train_cm}, validation_loss:{validation_loss}, validation_cm:{validation_cm}, validation_accuracy:{validation_accuracy}".format(
            epoch=epoch,
            train_loss=train_loss_meter.value()[0],
            train_cm=str(train_confusion_matrix.value()),
            validation_loss=validation_loss_meter.value()[0],
            validation_cm=str(validation_cm),
            validation_accuracy=validation_accuracy
        ))
        print("epoch:{epoch}, train_loss:{train_loss}, validation_loss:{validation_loss}, validation_accuracy:{validation_accuracy}".format(
            epoch=epoch,
            train_loss=train_loss_meter.value()[0],
            validation_loss=validation_loss_meter.value()[0],
            validation_accuracy=validation_accuracy
        ))
        print("train_cm:\n{train_cm}\n\nvalidation_cm:\n{validation_cm}".format(
            train_cm=str(train_confusion_matrix.value()),
            validation_cm=str(validation_cm),
        ))

        # early stop
        if validation_loss_meter.value()[0] < best_validation_loss:
            best_epoch = epoch
            best_validation_loss = validation_loss_meter.value()[0]
            dist_to_best = 0

        dist_to_best += 1
        if dist_to_best > 4:
            break

    model.save(path=os.path.join(args.ckpts_dir, 'model.pth'))
    vis.save()
    print("save model successfully")
    print("best epoch: ", best_epoch)
    print("best valid loss: ", best_validation_loss)
    time_end = time.clock()
    print('time cost: %.2f' % (time_end - time_begin))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='TextCNN', help="model to be used")
    parser.add_argument('--use_gpu', action='store_true', help="whether use gpu")
    parser.add_argument('--load_model_path', type=str, default=None, help="Path of pre-trained model")
    parser.add_argument('--ckpts_dir', type=str, default=None, help="Dir to store checkpoints")

    args = parser.parse_args()

    if not os.path.exists(args.ckpts_dir):
        os.makedirs(args.ckpts_dir)

    train(args)





