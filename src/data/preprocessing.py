def normalize_image(image):
    return image / 255.0

def augment_image(image):
    # Example augmentation: flipping the image horizontally
    return image[:, ::-1, :]

def preprocess_data(images):
    normalized_images = [normalize_image(img) for img in images]
    augmented_images = [augment_image(img) for img in normalized_images]
    return augmented_images

def load_and_preprocess_data(dataset):
    images = dataset.load_images()  # Assuming dataset has a method to load images
    return preprocess_data(images)