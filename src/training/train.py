from tensorflow import keras
from tensorflow.keras import layers
import os
import yaml
from src.data.dataset import FashionDataset
from src.models.cnn_model import CNNModel
from src.utils.config import load_config

def train_model():
    # Load configuration
    config = load_config('configs/config.yaml')
    
    # Load dataset
    train_dataset = FashionDataset(config['dataset']['train_data_path'])
    val_dataset = FashionDataset(config['dataset']['val_data_path'])
    
    # Create model
    model = CNNModel(input_shape=config['model']['input_shape'], num_classes=config['model']['num_classes'])
    
    # Compile model
    model.compile(optimizer=config['model']['optimizer'], 
                  loss=config['model']['loss'], 
                  metrics=config['model']['metrics'])
    
    # Train model
    history = model.fit(train_dataset, 
                        validation_data=val_dataset, 
                        epochs=config['training']['epochs'], 
                        batch_size=config['training']['batch_size'])
    
    # Save the trained model
    model.save(os.path.join(config['model']['save_path'], 'fashion_model.h5'))
    
    return history

if __name__ == "__main__":
    train_model()