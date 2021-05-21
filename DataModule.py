import csv
import spacy
import torch
from torch.utils.data import Dataset

# Try running python -m spacy download en and it should install the model to the correct directory.

class LRDataset(Dataset):
    def __init__(self, answer_file_path, max_length):
        # Read JSON file and assign to headlines variable (list of strings)
        self.data_dict = []
        self.lable_set = set()
        self.token2id = {}
        self.max_length = max_length
        # prepare data
        self.token_set, datas = self.load_data(answer_file_path)
        self.data_dict = self.process_data(datas)
        self.token2id = self.set2id(self.token_set, 'PAD', 'UNK')
        self.tag2id = self.set2id(set((datas).keys()))
        # create_dataset



    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, index):
        line, label = self.data_dict[index]

        seq_vector = self.vectorized_data(line, self.token2id)
        #print("seq_vector : {}".format(seqs_tensor.size()))
        seq_lengths = torch.LongTensor([len(seq_vector)])
        # padding
        padding_length = self.max_length - len(seq_vector)
        senten_vec = seq_vector + [0] * padding_length
        seqs_tensor = torch.LongTensor(senten_vec)
        # label
        label_tensor = torch.LongTensor([self.tag2id[label]])

        # print("seqs_tensor : {}".format(seqs_tensor.size()))
        # print(seqs_tensor)
        #
        # print("seq_lengths : {}".format(seq_lengths.size()))
        # print(seq_lengths)
        #
        # print("label_tensor : {}".format(label_tensor.size()))
        # print(label_tensor)
        #assert 1 == 0

        return seqs_tensor, label_tensor, seq_lengths

    def set2id(self, item_set, pad=None, unk=None):
        item2id = {}
        if pad is not None:
            item2id[pad] = 0
        if unk is not None:
            item2id[unk] = 1

        for item in item_set:
            item2id[item] = len(item2id)

        return item2id

    def load_data(self, data_file):
        tok = spacy.load('en')
        token_set = set()

        datas = {}
        # Train Set
        file_data = []
        for file in data_file:
            print(file)
            with open(file) as csvfile:
                csv_reader = csv.reader(csvfile)
                file_header = next(csv_reader)
                for row in csv_reader:
                    file_data.append(row)

        for row in file_data:
            cat = row[3]
            q_id = row[2]
            data = []

            line = row[1]
            #print(line)
            line = [token.text for token in tok.tokenizer(line)]
            for token in line:
                token_set.add(token)
            if len(line) > (self.max_length - 2):
                print("Exceed the max length")
                line = line[0:self.max_length - 3]
            data.append(line)
            if cat in datas.keys():
                datas[cat].append(data)
            else:
                datas[cat] = []
                datas[cat].append(data)

        return token_set, datas

    def process_data(self, datas):
        """
        Split data into train, dev, and test (currently use 80%/10%/10%)
        It is more make sense to split based on category, but currently it hurts performance
        """
        print('Data statistics: ')
        all_data = []
        for cat in datas:
            cat_data = datas[cat]
            print(cat, len(datas[cat]))
            # for dat in cat_data:
            #   print((dat, cat))
            all_data += [(dat, cat) for dat in cat_data]

        train_cat = set()
        for item, cat in all_data:
            train_cat.add(cat)
        print('Categories:', sorted(list(train_cat)))
        return all_data

    def vectorized_data(self, data, item2id):
        data_vec = []
        for seq in data:
            #print(seq)
            senten_vec = []
            # text
            for token in seq:
                if token in item2id:
                    senten_vec.append(item2id[token])
                else:
                    senten_vec.append(item2id['UNK'])
            data_vec.append(senten_vec)

        #print(data_vec)
        return data_vec[0]


