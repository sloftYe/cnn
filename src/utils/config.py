# Contents of the file: /fashion-classification-cnn/fashion-classification-cnn/src/utils/config.py

import os

class Config:
    def __init__(self):
        # Dataset paths
        self.dataset_path = '/content/drive/MyDrive/fashion_dataset'  # Update this path as needed
        self.train_data_path = os.path.join(self.dataset_path, 'train')
        self.test_data_path = os.path.join(self.dataset_path, 'test')

        # Training parameters
        self.batch_size = 32
        self.epochs = 50
        self.learning_rate = 0.001

        # Model parameters
        self.input_shape = (128, 128, 3)  # Example input shape for CNN
        self.num_classes = 10  # Update this based on the dataset

        # Checkpoint settings
        self.checkpoint_dir = './checkpoints'
        self.checkpoint_filepath = os.path.join(self.checkpoint_dir, 'model.h5')

        # Visualization settings
        self.visualization_dir = './visualizations'