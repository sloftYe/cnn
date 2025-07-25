def save_model(model, filepath):
    """Saves the model weights to the specified filepath."""
    model.save_weights(filepath)

def load_model(model, filepath):
    """Loads the model weights from the specified filepath."""
    model.load_weights(filepath)

def get_model_summary(model):
    """Returns a summary of the model architecture."""
    return model.summary()