# Project Overview

This project implements a convolutional neural network (CNN) for fashion classification using a dataset of fashion items. The project is structured to facilitate data loading, preprocessing, model training, and evaluation. It includes Jupyter notebooks for data exploration, model training, and evaluation, as well as a modular codebase for better organization and maintainability.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fashion-classification-cnn.git
   cd fashion-classification-cnn
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Upload Dataset**: Upload the fashion dataset to your Google Drive.
2. **Load Dataset**: Modify the dataset loading path in `src/data/dataset.py` to point to your Google Drive.
3. **Data Exploration**: Use the `notebooks/data_exploration.ipynb` to explore the dataset.
4. **Model Training**: Train the CNN model using `notebooks/model_training.ipynb`.
5. **Model Evaluation**: Evaluate the trained model with `notebooks/evaluation.ipynb`.

## File Structure

```
fashion-classification-cnn
├── src
│   ├── data
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   └── preprocessing.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── cnn_model.py
│   │   └── utils.py
│   ├── training
│   │   ├── __init__.py
│   │   ├── train.py
│   │   └── evaluate.py
│   └── utils
│       ├── __init__.py
│       ├── config.py
│       └── visualization.py
├── notebooks
│   ├── data_exploration.ipynb
│   ├── model_training.ipynb
│   └── evaluation.ipynb
├── configs
│   └── config.yaml
├── requirements.txt
├── setup.py
└── README.md
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.