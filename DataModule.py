import torch
import csv
from torch.utils.data import Dataset

from Constants import *

class SequenceDataset(Dataset):
    def __init__(self, dataset_file_path, tokenizer, device):
        # Read JSON file and assign to headlines variable (list of strings)
        self.data_dict = []
        self.device = device
        self.lable_set = set()
        file_data = []
        for file in dataset_file_path:
            with open(file) as csvfile:
                csv_reader = csv.reader(csvfile)
                file_header = next(csv_reader)
                for row in csv_reader:
                    file_data.append(row)
                for row in file_data:
                    #col-stat
                    cat = row[-1]
                    q_id = row[2]
                    #print(row[0])
                    q_text = q_text_dict[q_id]
                    # print(refs)
                    data_list = []
                    ref_list = []
                    if len(q_rubric_dict[q_id])>1:
                        ref_list = q_rubric_dict[q_id]
                    else:
                        ref_list.append(q_rubric_dict[q_id][0])
                        ref_list.append(q_rubric_dict[q_id][0])
                    for t in ref_list:
                        line = CLS_TOKEN + row[1] + SEP_TOKEN + t + SEP_TOKEN + q_text
                        #line = CLS_TOKEN + q_text + SEP_TOKEN + t + SEP_TOKEN + row[1]
                        #line = q_text + SEP_TOKEN + t + SEP_TOKEN + row[1] # test5
                        #line = t + SEP_TOKEN + row[1] # test6
                        #line = row[1] # test7
                        data_list.append(line)
                    # assert 0 == 1
                    data = []
                    self.lable_set.add(cat)
                    data.append(cat)
                    data.append(data_list)
                    self.data_dict.append(data)
        self.tokenizer = tokenizer
        self.tag2id = self.set2id(self.lable_set)
        #self.tag2id = {'0': 0, '1': 1, '2': 2, '3': 3}
        print(self.tag2id)
        print(self.get_category_distribution())

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, index):
        DEVICE = self.device
        label, lines = self.data_dict[index]
        label = self.tag2id[label]
        input_ids, attention_masks = [], []
        #print(lines)
        for line in lines:
            #tokenized_data = self.tokenizer(line, padding="max_length", truncation=True, return_tensors="pt")
            tokenized_data = self.tokenizer(line, padding="max_length", truncation=True, max_length=hyperparameters['max_length'])
            input_id = tokenized_data["input_ids"]
            attention_mask = tokenized_data["attention_mask"]
            input_ids.append(input_id)
            attention_masks.append(attention_mask)
        return {
            "input_ids":  torch.tensor(input_ids, dtype=torch.long, device=self.device),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long, device=self.device),
            "label": label,
        }


    def set2id(self, item_set, pad=None, unk=None):
        item2id = {}
        if pad is not None:
            item2id[pad] = 0
        if unk is not None:
            item2id[unk] = 1

        for item in item_set:
            item2id[item] = len(item2id)

        return item2id
    def get_category_distribution(self):
        cat_count = {}
        for data in self.data_dict:
            cat = data[0]  # Assuming category is the first element in the list
            #print(cat)
            if cat not in cat_count:
                cat_count[cat] = 0
            cat_count[cat] += 1
        #print(cat_count)
        return cat_count
