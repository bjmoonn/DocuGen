# Enhanced Documentation

# Installation

#setup
[roadmap]: https://github.com/Trusted-AI/adversarial-robustness-toolbox/wiki/Roadmap
[citing]: https://github.com/Trusted-AI/adversarial-robustness-toolbox/wiki/Contributing#citing-art

The library is under continuous development. Feedback, bug reports and contributions are very welcome!



# Usage

## Quick Start
```python
import adversarial_robustness_toolbox

# Initialize model
model = adversarial_robustness_toolbox.Model()

# Train model
model.train(data)

# Make predictions
predictions = model.predict(test_data)
```

## Examples

### Basic Training
```python
from adversarial_robustness_toolbox import Model, Dataset

# Load data
dataset = Dataset.from_csv('data.csv')

# Initialize and train
model = Model(
    input_size=dataset.input_size,
    hidden_size=128
)

# Train
model.train(
    dataset,
    epochs=10,
    batch_size=32
)
```

### Making Predictions
```python
# Load trained model
model = Model.load('path/to/saved/model')

# Make predictions
predictions = model.predict(test_data)
```

See the `examples/` directory for more detailed examples.


# API Reference

## Core Classes

### Model
The main model class for adversarial-robustness-toolbox.

#### Methods
- `__init__(input_size, hidden_size=128, **kwargs)`: Initialize model
  - `input_size`: Dimension of input features
  - `hidden_size`: Size of hidden layers
  - `**kwargs`: Additional model configuration

- `train(data, epochs=10, batch_size=32)`: Train the model
  - `data`: Training dataset
  - `epochs`: Number of training epochs
  - `batch_size`: Batch size for training

- `predict(data)`: Make predictions
  - `data`: Input data for predictions
  - Returns: Model predictions

- `save(path)`: Save model to disk
  - `path`: Path to save model

- `load(path)`: Load model from disk (class method)
  - `path`: Path to saved model
  - Returns: Loaded model instance

### Dataset
Dataset handling class.

#### Methods
- `from_csv(path)`: Create dataset from CSV file
- `from_numpy(data)`: Create dataset from numpy array


# Requirements

## System Requirements
- Python 3.7 or later
- CUDA compatible GPU (optional, for GPU acceleration)

## Python Dependencies
```
numpy>=1.19.2
pandas>=1.2.0
torch>=1.7.0
scikit-learn>=0.24.0
tqdm>=4.50.0
```

## Installation
Install all dependencies:
```bash
pip install -r requirements.txt
```

## Optional Dependencies
For development:
```bash
pip install -r requirements-dev.txt
```

## GPU Support
To enable GPU acceleration, install PyTorch with CUDA support:
```bash
pip install torch --extra-index-url https://download.pytorch.org/whl/cu116
```


