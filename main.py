import argparse
import wandb
import random
from tqdm.auto import tqdm
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader, SubsetRandomSampler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, AutoConfig
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.metrics import cohen_kappa_score
from torch.optim import AdamW
from Constants import *
from DataModules import SequenceDataset
from SFRNModel import SFRNModel


def train(args):
    wandb.init(project="SFRN", entity="zhaohuilee", config=config_dictionary)
    random.seed(hyperparameters['random_seed'])
    # model
    best_acc, best_f1 = 0, 0
    best_ckp_path = ''
    DEVICE = args.device
    print(DEVICE)
    model_name = hyperparameters['model_name']
    print(model_name)
    print(hyperparameters)
    # Initialize BERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Load Train dataset and split it into Train and Validation dataset
    train_dataset = SequenceDataset(TRAIN_FILE_PATH, tokenizer, DEVICE)
    test_dataset = SequenceDataset(TEST_FILE_PATH, tokenizer, DEVICE)
    test_dataset.tag2id = train_dataset.tag2id
    trainset_size = len(train_dataset)
    testset_size = len(test_dataset)
    shuffle_dataset = True
    validation_split = hyperparameters['data_split']
    indices = list(range(trainset_size))
    split = int(np.floor(validation_split * trainset_size))
    if shuffle_dataset:
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1,
                                               sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1,
                                             sampler=validation_sampler)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)


    training_acc_list, validation_acc_list = [], []

    model = SFRNModel()
    model.to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=hyperparameters['lr'], weight_decay=hyperparameters['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    #scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    num_training_steps = len(train_loader) * hyperparameters['epochs']
    warmup_steps = int(hyperparameters['WARMUP_STEPS'] * num_training_steps)  # 10% of total training steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=num_training_steps)
    model.train()
    # Training Loop
    for epoch in range(hyperparameters['epochs']):
        train_loss = 0.0
        y_true = list()
        y_pred = list()
        train_iterator = tqdm(train_loader, desc="Train Iteration")
        for step, batch in enumerate(train_iterator):
            #optimizer.zero_grad()
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            logits = model(input_ids, attention_mask=attention_mask)
            #logits, loss = outputs.logits, outputs.loss
            loss = criterion(logits, labels) #/ GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            loss = loss.data.cpu().numpy()
            train_loss += loss
            pred_idx = torch.max(logits, 1)[1]
            y_true += list(labels.data.cpu().numpy())
            y_pred += list(pred_idx.data.cpu().numpy())
            nn.utils.clip_grad_norm_(model.parameters(), hyperparameters['max_norm'])
            if (step + 1) % hyperparameters['GRADIENT_ACCUMULATION_STEPS'] == 0:
                #scheduler.step()
                optimizer.step()
                model.zero_grad()
                scheduler.step()
#             optimizer.step()
#             scheduler.step()
        train_f1 = f1_score(y_true, y_pred, average='macro')
        train_qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
        train_acc = accuracy_score(y_true, y_pred)
        print('Epoch {} - Loss {}'.format(epoch + 1, train_loss))
        #print('Epoch {} - Loss {:.2f}'.format(epoch + 1, epoch_loss / len(train_indices)))

        # Validation Loop
        with torch.no_grad():
            model.eval()
            val_loss = 0
            val_y_true = list()
            val_y_pred = list()
            val_iterator = tqdm(val_loader, desc="Validation Iteration")
            for step, batch in enumerate(val_iterator):
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["label"].to(DEVICE)
                logits = model(input_ids, attention_mask=attention_mask)
                #logits, loss = outputs.logits, outputs.loss
                loss = criterion(logits, labels) / hyperparameters['GRADIENT_ACCUMULATION_STEPS']
                val_loss += loss.data.cpu().numpy()
                _, predicted = torch.max(logits.data, 1)
                val_y_pred += list(predicted.data.cpu().numpy())
                val_y_true += list(labels.data.cpu().numpy())
                #break
            val_acc = accuracy_score(val_y_true, val_y_pred)
            val_f1 = f1_score(val_y_true, val_y_pred, average='macro')
            val_qwk = cohen_kappa_score(val_y_true, val_y_pred, weights='quadratic')
            print('Training Accuracy {} - Validation Accurracy {}'.format(
                train_acc, val_acc))
            print('Training F1 {} - Validation F1 {}'.format(
                train_f1, val_f1))
            print('Training loss {} - Validation Loss {}'.format(
                train_loss, val_loss))
            if (val_acc > best_acc) and (val_f1 > best_f1):
                best_acc = val_acc
                best_f1 = val_f1
                with open(
                        'checkpoint/checkpoint_{}_at_epoch{}.model'.format(str(args.ckp_name), str(epoch)), 'wb'
                ) as f:
                    torch.save(model.state_dict(), f)
                best_ckp_path = 'checkpoint/checkpoint_{}_at_epoch{}.model'.format(str(args.ckp_name), str(epoch))

        with torch.no_grad():
            model.eval() 
            y_true = list()
            y_pred = list()
            test_iterator = tqdm(test_loader, desc="Test Iteration")
            for step, batch in enumerate(test_iterator):
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["label"].to(DEVICE)
                logits = model(input_ids, attention_mask=attention_mask)
                pred_idx = torch.max(logits, 1)[1]
                y_true += list(labels.data.cpu().numpy())
                y_pred += list(pred_idx.data.cpu().numpy())
                # break
            acc = accuracy_score(y_true, y_pred)
            qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
            f1 = f1_score(y_true, y_pred, average='macro')
            print("Test acc is {} ".format(acc))
            print("Test qwk is {} ".format(qwk))
            print("Test f1 is {} ".format(f1))
            wandb.log({"Train loss": train_loss, "Val loss": val_loss, "Train f1": train_f1, "Val f1": val_f1,
                   "Train Acc": train_acc, "Val Acc": val_acc, "Train QWK": train_qwk, "Val QWK": val_qwk, "test Acc": acc,"test F1":f1, "test qwk":qwk})
    print('Real Test: \n')
    with torch.no_grad():
        test_correct_total = 0
        print("start to test at {} ".format(best_ckp_path))
        model.load_state_dict(torch.load('./' + best_ckp_path))
        model.eval()
        y_true = list()
        y_pred = list()
        test_iterator = tqdm(test_loader, desc="Test Iteration")
        for step, batch in enumerate(test_iterator):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            logits = model(input_ids, attention_mask=attention_mask)
            pred_idx = torch.max(logits, 1)[1]
            y_true += list(labels.data.cpu().numpy())
            y_pred += list(pred_idx.data.cpu().numpy())
            # break
        acc = accuracy_score(y_true, y_pred)
        qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
        f1 = f1_score(y_true, y_pred, average='macro')
        print("Test acc is {} ".format(acc))
        print("Test f1 is {} ".format(f1))
        print("Quadratic Weighted Kappa is {}".format(qwk))
        print(classification_report(y_true, y_pred))
        print(confusion_matrix(y_true, y_pred))
        wandb.log({"final_test Acc": acc})
        wandb.log({"final_test QWK": qwk})
        wandb.log({"final_test f1": f1})
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckp_name', type=str, default='debug_cpt',
                        help='ckp_name')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")')
    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()
