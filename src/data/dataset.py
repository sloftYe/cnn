class Dataset:
    def __init__(self, data_dir, image_size=(224, 224), batch_size=32):
        self.data_dir = data_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def load_data(self):
        # Load the dataset from the specified directory
        # This method should implement the logic to load images and labels
        pass

    def preprocess_data(self):
        # Implement preprocessing steps such as normalization and augmentation
        pass

    def get_train_data(self):
        return self.train_data

    def get_val_data(self):
        return self.val_data

    def get_test_data(self):
        return self.test_data