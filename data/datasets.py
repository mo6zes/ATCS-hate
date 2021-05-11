import torch
import numpy as np
import json
import csv
import os
import re
from string import punctuation
from torch.utils.data import Dataset, DataLoader, Sampler


class DataTwitterDavidson(Dataset):
    """
    Dataset class for Twitter data by Davidson
    """
    def __init__(self, csv_file_dir: str="./raw_datasets/davidsonTwitterData.csv"):
        self.text = []
        self.labels = []
        with open(os.path.join(os.path.dirname(__file__), csv_file_dir), mode='r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self.text.append(row['tweet'])
                self.labels.append(torch.tensor(int(row['class']), dtype=torch.long))

        assert len(self.text) == len(self.labels)
        self.n_classes = torch.numel(torch.unique(torch.tensor(self.labels)))
        self.task_name = csv_file_dir.split("/")[-1]

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        return self.text[idx], self.labels[idx]


class DataFoxNews(Dataset):
    """
    Dataset Class for the fox news dataset
    """
    def __init__(self, json_file_dir:str="./raw_datasets/fox-news-comments.json"):
        self.text = []
        self.labels = []

        with open(os.path.join(os.path.dirname(__file__), json_file_dir), mode='r') as jsonfile:
            content = json.load(jsonfile)

            for data in content:
                self.text.append(data['text'])
                self.labels.append(torch.tensor(int(data['label']), dtype=torch.long))

        assert len(self.text) == len(self.labels)
        self.n_classes = torch.numel(torch.unique(torch.tensor(self.labels)))
        self.task_name = json_file_dir.split('/')[-1]

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        return self.text[idx], self.labels[idx]


class DeGilbertStormFront(Dataset):
    """
    Dataset for DeGilbert Dataset based on Storm Front forum
    """
    def __init__(self, csv_file_dir:str="./raw_datasets/deGilbertStormfront.csv"):
        self.tweets = []
        self.labels = []
        self.label_dict = {'noHate': 0, 'hate': 1}

        with open(os.path.join(os.path.dirname(__file__), csv_file_dir), mode='r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self.tweets.append(row['text'])
                self.labels.append(torch.tensor(int(self.label_dict[row['label']]), dtype=torch.long))

        assert len(self.tweets) == len(self.labels)
        self.n_classes = torch.numel(torch.unique(torch.tensor(self.labels)))
        self.task_name = csv_file_dir.split("/")[-1]

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, idx):
        return self.tweets[idx], self.labels[idx]

    @staticmethod
    def create_data_csv_file(csv_annotations_dir='../data/deGilbert/annotations_metadata.csv'):

        with open(csv_annotations_dir, mode='r') as csvfile:
            reader = csv.DictReader(csvfile)
            data = []

            for row in reader:
                label = row['label']
                file_id = row['file_id']

                with open('../data/deGilbert/all_files/' + file_id + '.txt', mode='r', encoding="utf8") as txt_file:
                    text = txt_file.read()

                data.append([file_id, label, text])

        # write to new csv file
        new_header = ['file_id', 'label', 'text']
        new_file_name = '../data/deGilbert/deGilbertStormfront.csv'
        with open(new_file_name, 'a', newline='', encoding="utf8") as new_file:
            wr = csv.writer(new_file)
            if(os.stat(new_file_name).st_size == 0):
                wr.writerow(new_header)
            wr.writerows(data)

class QuianData(Dataset):
    """
    Quian Dataset which is the same for Gab and Reddit which can be indicated with flag
    """
    def __init__(self, csv_file_dir: str="./raw_datasets/gabQuian.csv", raw_csv_file_dir: str=None,
                 save_new_csv_dir: str=None):
        self.tweets = []
        self.labels = []

        self.label_dict = {'noHate': 0, 'hate': 1}

        self.data_flag = None  # TODO

        self.raw_csv = raw_csv_file_dir
        self.save_dir = save_new_csv_dir

        with open(os.path.join(os.path.dirname(__file__), csv_file_dir), mode='r', encoding="utf8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self.tweets.append(row['text'])
                self.labels.append(torch.tensor(int(self.label_dict[row['label']]), dtype=torch.long))

        assert len(self.tweets) == len(self.labels)
        self.n_classes = torch.numel(torch.unique(torch.tensor(self.labels)))
        self.task_name = csv_file_dir.split("/")[-1]

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, idx):
        return self.tweets[idx], self.labels[idx]

    def create_data_csv_file(self):

        with open(self.raw_csv, mode='r', encoding="utf8") as csvfile:
            reader = csv.DictReader(csvfile)
            data = []

            for row in reader:

                if row['hate_speech_idx'] == 'n/a':
                    continue

                text = row['text'].split('\n')[:-1]  # don't need last split ob break that yields empty string

                hate_idx = row['hate_speech_idx'].replace('[', '').replace(']', '').replace(',', '')
                hate_idx = list(hate_idx.split(" "))
                # turn new_k into ints
                hate_idx = [int(i) - 1 for i in hate_idx]

                for idx, txt in enumerate(text):

                    txt = txt[2:].lstrip(punctuation).lstrip()  # remove numbering and leading white space/tab/punctuation

                    if txt == '':  # skip empty text
                        continue
                    # if idx in hate_idx then this part is hate speech
                    if idx in hate_idx:
                        data.append(['hate', txt])
                    # otherwise no hate label
                    else:
                        data.append(['noHate', txt])

        # write to new csv file
        new_header = ['label', 'text']
        with open(self.save_dir, 'a', newline='', encoding="utf8") as new_file:
            wr = csv.writer(new_file)
            if os.stat(new_file_name).st_size == 0:  # TODO; new_file_name does not exist
                wr.writerow(new_header)
            wr.writerows(data)
            
class RezvanHarrassment(Dataset):
    def __init__(self, csv_file_dir: str="./raw_datasets/rezvanData.csv"):
        self.tweets = []
        self.labels = []

        self.label_dict = {
            'no': 0, 
            'appearance': 1,
            'intelligence': 2,
            'political': 3,
            'racial': 4,
            'sexual': 5}
        
        with open(os.path.join(os.path.dirname(__file__), csv_file_dir), mode='r') as csvfile:
            reader = csv.DictReader(csvfile)

            for row in reader:
                self.tweets.append(row['Tweets'])
                self.labels.append(torch.tensor(int(self.label_dict[row['Decision']]), dtype=torch.long))

        assert len(self.tweets) == len(self.labels)
        self.n_classes = torch.numel(torch.unique(torch.tensor(self.labels)))
        self.task_name = csv_file_dir.split("/")[-1]

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, idx):
        return self.tweets[idx], self.labels[idx]

class FountaDataset(Dataset):
    def __init__(self, csv_file_dir: str="./raw_datasets/fountaCombined.csv"):
        self.tweets = []
        self.labels = []

        self.label_dict = {
            'normal': 0, 
            'abusive': 1,
            'spam': 2,
            'hateful': 3
        }
        
        with open(os.path.join(os.path.dirname(__file__), csv_file_dir), mode='r', encoding="utf8") as csvfile:
            reader = csv.reader(csvfile)
            
            for row in reader:
                s = ' '
                data = s.join(row)
                data_split = re.split(r'\t+', data)
                
                tweet = data_split[0]
                label = data_split[1]
    
                self.tweets.append(tweet)
                self.labels.append(torch.tensor(int(self.label_dict[label]), dtype=torch.long))

        assert len(self.tweets) == len(self.labels)
        self.n_classes = torch.numel(torch.unique(torch.tensor(self.labels)))
        self.task_name = csv_file_dir.split("/")[-1]

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, idx):
        return self.tweets[idx], self.labels[idx]
            
class BalancedSampler(Sampler):
    """
    Sample from dataset in a balanced manner. No guarantee all the samples are seen during training.
    """
    def __init__(self, labels):
        self.labels = labels
        self.num_classes = torch.numel(torch.unique(torch.stack(self.labels)))

        self.class_indices = []
        for i in range(self.num_classes):
            lst = torch.unbind(torch.nonzero(torch.stack(self.labels) == i).squeeze(-1))
            lst = [i.item() for i in lst]
            self.class_indices.append(lst)
            
        self.counts = [0] * self.num_classes
        self.current_class = 0

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count >= len(self.labels):
            raise StopIteration
        self.count += 1
        return self.sample()

    def sample(self):
        chosen_class = self.get_class()
        class_indices = self.class_indices[chosen_class]
        chosen_index = np.random.choice(class_indices)
        self.counts[chosen_class] += 1
        return chosen_index

    def get_class(self):
        min_count = self.counts[0]
        min_classes = [0]
        for i in range(1, self.num_classes):
            if self.counts[i] < min_count:
                min_count = self.counts[i]
                min_classes = [i]
            if self.counts[i] == min_count:
                min_classes.append(i)
        chosen_class = np.random.choice(min_classes)
        return chosen_class

    def __len__(self):
        return len(self.labels)


ALL_DATASETS = {
    'twitter_davidson': DataTwitterDavidson,
    'fox_news': DataFoxNews,
    'degilbert_storefront': DeGilbertStormFront,
    'quian': QuianData
}

# if __name__ == "__main__":
#     data_dir = './data/gabQuian.csv'
#     dataset = DataGabQuian(data_dir)
