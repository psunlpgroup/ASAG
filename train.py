import os
import sys
import argparse
import time
import random
import numpy as np

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim


from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from DataModule import LRDataset
from LR_model import LRClassifier
from tqdm import tqdm, trange
from utils import sort_batch
import ml_metrics as metrics
import gc

gc.collect()
torch.cuda.empty_cache()

lr = 5e-6
lr_max = 5e-4
lr_gamma = 2
lr_step = 20
clip_norm = 50
weight_decay = 1e-4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', nargs='+', default=['./example/new_format/train/aug_answer.csv'],
                        help='data_directory')
    parser.add_argument('--test_dir', nargs='+',
                        default=['./example/new_format/test/aug_answer.csv'],
                        help='data_directory')
    parser.add_argument('--emb_path', type=str, default='./model/glove.6B.100d.txt',
                        help='data_directory')
    parser.add_argument('--hidden_dim', type=int, default=32,
                        help='LSTM hidden dimensions')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='size for each minibatch')
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='maximum number of epochs')
    parser.add_argument('--char_dim', type=int, default=100,
                        help='character embedding dimensions')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight_decay rate')
    parser.add_argument('--max_length', type=int, default=30,
                        help='max_length')
    parser.add_argument('--gpu', type=str, default='cuda',
                        help='gpu name')
    parser.add_argument('--ckp_name', type=str, default='ckp_name',
                        help='ckp_name')
    parser.add_argument('--data_parallel', type=bool, default=True,
                        help='data_parallel')
    args = parser.parse_args()
    train(args)


def loadEMbeddingMatrix(EMBEDDING_FILE, typeToLoad, vocab, embed_size=100):
    if typeToLoad == 'glove' or typeToLoad == 'fasttext':
        embedding_index = dict()
        f = open(EMBEDDING_FILE)
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = coefs
        f.close()
        print('Load %s word vectors.' % len(embedding_index))
        embedding_matrix = np.random.normal(0, 1, (len(vocab), embed_size))
        embeddedCount = 0
        for word, i in vocab.items():  # 词
            embedding_vector = embedding_index.get(word)
            # print('word: ', word)
            # print('i: ', i)
            # print('embedding_vector: ', embedding_vector)

            if embedding_vector is not None:
                # print("++++")
                # print(embedding_vector)
                embedding_matrix[i] = embedding_vector
                embeddedCount += 1
        print('total_embedded:', embeddedCount, 'commen words')
        # print('embedding matrix: ', embedding_matrix)

        return embedding_matrix


def apply(model, criterion, batch, targets, lengths):
    pred = model(torch.autograd.Variable(batch), lengths)
    # print("pred {}".format(pred))
    # print("targets {}".format(targets))
    loss = criterion(pred, torch.autograd.Variable(targets))
    return pred, loss


def train_model(ckp_name, best_acc, device, model, optimizer, scheduler, train_loader, dev_loader, max_epochs, train_len, dev_len):
    # criterion = nn.NLLLoss(size_average=False)
    criterion = nn.CrossEntropyLoss()
    best_ckp_path = ''
    start = time.time()
    # Training loop
    for epoch in range(max_epochs):
        print('Epoch:', epoch)
        if scheduler.get_lr()[0] < lr_max:
            scheduler.step()
        print("The learning rate is: {}".format(scheduler.get_lr()[0]))
        train_iterator = tqdm(train_loader, desc="Train Iteration")
        y_true = list()
        y_pred = list()
        total_loss = 0
        for step, batch in enumerate(train_iterator):
            #print("++++++++++++++++")
            padded_seq_tensor = batch[0]
            #print(padded_seq_tensor.size())
            targets, lengths = batch[1].squeeze(), batch[2]
            batch, targets, lengths = sort_batch(padded_seq_tensor, targets, lengths)
            model.zero_grad()
            pred, loss = apply(model, criterion,
                            batch.to(device), targets.to(device), lengths.squeeze().to(device))
            # print("loss: {}".format(loss))
            # nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            loss.backward()
            optimizer.step()

            pred_idx = torch.max(pred, 1)[1]
            y_true += list(targets.data.cpu().numpy())
            y_pred += list(pred_idx.data.cpu().numpy())
            # print("y_true {}".format(y_true))
            # print("y_pred {}".format(y_pred))
            total_loss += loss

        acc = accuracy_score(y_true, y_pred)
        # long running
        # do something other
        end = time.time()
        print("The training time :{}".format(str(end - start)))
        val_loss, val_acc, best_ckp_path, best_acc = evaluate_validation_set(best_ckp_path, best_acc, epoch, ckp_name,
                                                                             device, model, dev_loader, criterion, dev_len)
        print("Train loss: {} - acc: {} \nValidation loss: {} - acc: {}".format(total_loss.data.float() / train_len, acc,
                                                                              val_loss, val_acc))
        torch.cuda.empty_cache()
    return model, best_ckp_path, best_acc


def evaluate_validation_set(best_ckp_path, best_acc, epoch, ckp_name, device, model, dev_loader, criterion, dev_len):
    print("start evaluation")
    torch.cuda.empty_cache()
    y_true = list()
    y_pred = list()
    total_loss = 0
    start = time.time()
    dev_iterator = tqdm(dev_loader, desc="Dev Iteration")
    for step, batch in enumerate(dev_iterator):
        padded_seq_tensor = batch[0]
        # print(padded_seq_tensor.size())
        targets, lengths = batch[1].squeeze(), batch[2]
        batch, targets, lengths = sort_batch(padded_seq_tensor, targets, lengths)
        pred, loss = apply(model, criterion,
                           batch.to(device), targets.to(device), lengths.squeeze().to(device))
        pred_idx = torch.max(pred, 1)[1]
        y_true += list(targets.data.cpu().numpy())
        y_pred += list(pred_idx.data.cpu().numpy())
        total_loss += loss
    acc = accuracy_score(y_true, y_pred)
    if acc > best_acc:
        best_acc = acc
        with open(
                'checkpoint/checkpoint_{}_at_epoch{}.model'.format(str(ckp_name), str(epoch)), 'wb'
        ) as f:
            torch.save(model.state_dict(), f)
        best_ckp_path = 'checkpoint/checkpoint_{}_at_epoch{}.model'.format(str(ckp_name), str(epoch))
    end = time.time()
    print("The testing time :{}".format(str(end - start)))
    # print(best_ckp_path)
    return total_loss.data.float() / dev_len, acc, best_ckp_path, best_acc


def evaluate_test_set(best_ckp_path, device, model, test_loader):
    y_true = list()
    y_pred = list()

    # print(test)
    print("start to test at {} ".format(best_ckp_path))
    model.load_state_dict(torch.load('./' + best_ckp_path))
    model.eval()

    test_iterator = tqdm(test_loader, desc="Test Iteration")
    for step, batch in enumerate(test_iterator):
        padded_seq_tensor = batch[0]
        # print(padded_seq_tensor.size())
        targets, lengths = batch[1].squeeze(), batch[2]
        batch, targets, lengths = sort_batch(padded_seq_tensor, targets, lengths)
        pred = model(batch.to(device), lengths.squeeze().to(device))
        pred_idx = torch.max(pred, 1)[1]
        y_true += list(targets.data.cpu().numpy())
        y_pred += list(pred_idx.data.cpu().numpy())

    print(len(y_true), len(y_pred))
    acc = accuracy_score(y_true, y_pred)
    print("Quadratic Weighted Kappa is {}".format(metrics.quadratic_weighted_kappa(y_true, y_pred)))
    print("Test acc is {} ".format(acc))
    print(classification_report(y_true, y_pred))
    # print(confusion_matrix(y_true, y_pred))


def train(args):
    random.seed(123)
    device = torch.device(args.gpu if torch.cuda.is_available() else 'cpu')
    train_dataset = LRDataset(args.train_dir, args.max_length)
    test_dataset = LRDataset(args.test_dir, args.max_length)

    validation_split = 0.2
    dataset_size = len(train_dataset)
    testset_size = len(test_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    shuffle_dataset = True

    if shuffle_dataset:
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=validation_sampler)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)
    char_vocab = train_dataset.token2id
    char_vocab_size = len(char_vocab)
    tag_vocab = train_dataset.tag2id
    print('Training Set Size {}, Validation Set Size {}'.format(len(train_indices), len(val_indices)))
    print('Testing Set Size {}'.format(testset_size))

    embedding_matrix = loadEMbeddingMatrix(args.emb_path, "glove", char_vocab, args.char_dim)
    print("------------------------------------")
    print(embedding_matrix.shape)
    model = LRClassifier(char_vocab_size, args.char_dim, args.hidden_dim, len(tag_vocab), embedding_matrix)
    if args.data_parallel:
        # model = nn.DataParallel(model)
        model = model.to(device)
    # optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)
    best_acc = 0
    model, best_ckp_path, best_acc = train_model(args.ckp_name, best_acc, device, model, optimizer,
                                                 scheduler, train_loader, val_loader,
                                                 args.num_epochs, len(train_indices), len(val_indices))
    print(best_ckp_path)
    evaluate_test_set(best_ckp_path, device, model, test_loader)


if __name__ == '__main__':
    main()