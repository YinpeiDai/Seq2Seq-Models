import random
import copy
import json
import re,pprint


class ToyDataset:
    def __init__(self):
        with open('data.json') as f:
            self.data = json.load(f)

        self.train_data = self.data[1000:]
        self.test_data = self.data[:1000]
        self.batch_id = 0


    def get_batch(self, batch_size=20):
        if self.batch_id+batch_size>=len(self.train_data):
            self.batch_id = 0
            random.shuffle(self.train_data)
        batch_data = self.train_data[self.batch_id:self.batch_id + batch_size]
        batch_source = copy.deepcopy(batch_data)
        batch_target = copy.deepcopy(batch_data)
        self.batch_id = self.batch_id + batch_size
        return batch_source, batch_target

    def get_test_data(self):
        batch_data = self.test_data
        batch_source = copy.deepcopy(batch_data)
        batch_target = copy.deepcopy(batch_data)
        return batch_source, batch_target












if __name__ == '__main__':
    dataset = ToyDataset()
    pprint.pprint(dataset.get_batch())
