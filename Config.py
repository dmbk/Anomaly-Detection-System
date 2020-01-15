
from os.path import join

class Config:
    def __init__(self, data_dir):
        self.DATASET_PATH = join(data_dir,"UCSDped1/Train")
        self.SINGLE_TEST_PATH = join(data_dir,"UCSDped1/Test/Test032")
        self.BATCH_SIZE = 4
        self.EPOCHS = 3
        self.MODEL_PATH = join("model.hdf5")
