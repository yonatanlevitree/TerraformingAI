# Installation Guide

This guide will help you set up the environment needed to run the Terrain Optimization tools.

## Requirements

- Python 3.8 or higher
- pip (Python package installer)
- Virtual environment (recommended)

## Installation Steps

### 1. Clone the Repository

If you haven't already, clone the repository containing the code:

```bash
git clone <repository-url>
cd Terraforming-AI/Newest_Attempt/Stage_Two
```

### 2. Create and Activate a Virtual Environment (recommended)

#### On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

#### On macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

To verify that everything is installed correctly, run:

```bash
python -c "import numpy; import torch; import matplotlib; import scipy; import plotly; print('All packages imported successfully!')"
```

If no errors appear, you're ready to use the Terrain Optimization tools.

## Running the Demonstration

After successful installation, you can run the demonstration script:

```bash
python run_terrain_optimization.py
```

This will generate several visualization files that show the terrain optimization process.

## Troubleshooting

### PyTorch Installation Issues

If you encounter issues with the PyTorch installation, you might need to install it separately with specific configuration for your system. Visit [PyTorch's official installation page](https://pytorch.org/get-started/locally/) and follow the instructions for your operating system and CUDA version (if applicable).

### Plotly Rendering Issues

If you have trouble with Plotly visualizations:

1. Ensure you have ipywidgets installed:
   ```bash
   pip install ipywidgets
   ```

2. If using Jupyter Notebook, install the Jupyter Plotly extension:
   ```bash
   jupyter labextension install jupyterlab-plotly
   ```

### Matplotlib 3D Visualization Issues

If 3D plots in Matplotlib don't display properly, try updating your backend:

```python
import matplotlib
matplotlib.use('TkAgg')  # or try 'Qt5Agg' if TkAgg doesn't work
import matplotlib.pyplot as plt
```

## Additional Resources

- [NumPy Documentation](https://numpy.org/doc/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [Plotly Documentation](https://plotly.com/python/) 