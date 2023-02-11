import pprint
class StreamDataLoader:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset.events['pulls']
        self.batch_size = batch_size
        self.currentIdx = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.currentIdx + self.batch_size -1 >= len(self.dataset):
            raise StopIteration
        else:
            # return 10 items as first tuple and next one as the second tuple
            training_pulls = self.dataset.iloc[self.currentIdx: self.currentIdx+ self.batch_size -1].to_dict('records')
            test_pull = self.dataset.iloc[self.currentIdx+self.batch_size].to_dict()
            self.currentIdx += 1
            return (training_pulls, test_pull)

    def __len__(self):
        return len(self.dataset)
    
    def reset(self):
        self.batch = 0
    
    def get_batch_size(self):
        return self.batch_size
    
    def get_dataset(self):
        return self.dataset
    
    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
    
    def set_dataset(self, dataset):
        self.dataset = dataset
    
    def get_batch(self):
        return self.batch
    
    def set_batch(self, batch):
        self.batch = batch

    