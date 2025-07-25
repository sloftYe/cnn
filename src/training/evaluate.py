def evaluate_model(model, test_data, test_labels):
    """
    Evaluate the trained model on the test dataset.

    Parameters:
    model: The trained model to evaluate.
    test_data: The data to evaluate the model on.
    test_labels: The true labels for the test data.

    Returns:
    loss: The loss value of the model on the test dataset.
    accuracy: The accuracy of the model on the test dataset.
    """
    loss, accuracy = model.evaluate(test_data, test_labels, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    return loss, accuracy


if __name__ == "__main__":
    import sys
    from src.utils.config import load_config
    from src.models.cnn_model import CNNModel
    from src.data.dataset import load_data

    # Load configuration
    config = load_config()

    # Load test data
    test_data, test_labels = load_data(config['data']['test_data_path'])

    # Initialize and load the model
    model = CNNModel(input_shape=config['model']['input_shape'], num_classes=config['model']['num_classes'])
    model.load_weights(config['model']['weights_path'])

    # Evaluate the model
    evaluate_model(model, test_data, test_labels)