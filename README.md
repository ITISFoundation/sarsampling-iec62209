[![Pytest](.github/badges/tests.svg) ![Coverage](.github/badges/coverage.svg)](
    ./coverage.txt
)

# SAR Sampling

A Python package for generating structured samples for SAR (Specific Absorption Rate) testing scenarios using Latin Hypercube sampling with domain-specific constraints and mappings.

## Overview

SAR Sampling is designed to create realistic test scenarios for SAR testing by combining Latin Hypercube sampling with domain-specific knowledge about antenna configurations, frequency/radius mappings, and modulation schemes. The package processes input tables containing antenna data and generates structured samples that maintain the statistical properties of the original data while providing comprehensive coverage of the test space.

## Features

- **Latin Hypercube Sampling**: Implements various LHS methods including classic, centered, maximin, and centermaximin
- **Domain-Specific Constraints**: Handles SAR-specific requirements for antenna configurations, frequencies, distances, and modulation schemes
- **Voronoi Diagram Integration**: Uses spatial distribution analysis for frequency-radius mappings
- **Flexible Input Formats**: Supports both pandas DataFrames and CSV files
- **Configurable Sampling**: Customizable domains, weights, and sampling methods

## Installation

### Prerequisites

The package requires Python 3.10+ and the following dependencies:
- numpy
- pandas
- scipy
- shapely

### Install from source

```bash
git clone <repository-url>
pip install <project-root-dir>
```

## Quick Start

### Input Data Format

The package expects input data in a specific format. Here's an example:

```csv
antenna,frequency,2mm,5mm,10mm,21mm,M1,M2,M3,M4
D750,750,,,,15.4,21.5,1,1,1,0
D835,835,4.6,,15.0,20.8,1,1,0,0
D900,900,4.7,,14.6,20.4,1,1,0,0
D1450,1450,5.4,9.4,,18.3,1,1,0,0
```

**Column requirements:**
- `antenna`: Unique identifier for each antenna configuration
- `frequency`: Frequency value in MHz
- Distance columns: Must end with 'mm' (e.g., '2mm', '5mm', '10mm')
- Modulation columns: Must start with 'M' (e.g., 'M1', 'M2', 'M3')

### Basic Run

```python
from sar_sampling import SarSampler
import pandas as pd

# Create or load input data from csv
input_data = {
    'antenna': ['D750', 'D835', 'D900', 'D1450'],
    'frequency': [750, 835, 900, 1450],
    '10mm': [15.4, 15.0, 14.6, None],
    '21mm': [21.5, 20.8, 20.4, 18.3],
    'M1': [1, 1, 1, 1],
    'M2': [1, 1, 1, 1],
}
input_df = pd.DataFrame(input_data)

# Initialize sampler
sampler = SarSampler(input_df)

# Generate samples
sample = sampler(n_samples=100, method='maximin', seed=42)

# Access results
print(sample)

# save to file
sample.to_csv('sample.csv')
```

## Sampling Methods

The package supports several Latin Hypercube sampling methods:

- **'classic'**: Standard Latin Hypercube sampling
- **'center'/'c'**: Centered Latin Hypercube sampling
- **'maximin'/'m'**: Maximin distance optimization
- **'centermaximin'/'cm'**: Centered with maximin optimization

## Core Components

### SarSampler

The main class for SAR-specific sampling:

```python
from sar_sampling import SarSampler

# Initialize with input table
sampler = SarSampler(input_table, config={})

# Generate samples
samples = sampler(n_samples=100, method='maximin', seed=42)
```

**Parameters:**
- `input_table`: DataFrame or CSV path containing antenna configurations
- `config`: Optional configuration overrides

**Required input columns:**
- `antenna`: Antenna identifiers
- `frequency`: Frequency values
- Distance columns ending with 'mm' (e.g., '10mm', '21mm')
- Modulation columns starting with 'M' (e.g., 'M1', 'M2')

### LhSampler

General-purpose Latin Hypercube sampler:

```python
from sar_sampling import LhSampler

# Define domains
domains = {
    'a': [0, 10],      # Continuous uniform on [0, 10]
    'b': [0, 20],      # Continuous uniform on [0, 20]
    'c': [1, 2, 3]  # Discrete values
}

# Initialize sampler
sampler = LhSampler(domains)

# Generate samples
samples = sampler(100, method='maximin')
```

### lhs Function

Direct Latin Hypercube sampling function:

```python
from sar_sampling import lhs
import numpy as np

# Generate 100 samples in 3 dimensions
samples = lhs(dim=3, size=100, method='maximin', seed=42)
```

### Sample Class

Container for structured data with variable classification:

```python
from sar_sampling import Sample

# Create sample with variable classification
sample = Sample(df, xvar=['x', 'y'], zvar='output')

# Access data
x_data = sample.xdata()  # Independent variables
z_data = sample.zdata()  # Dependent variables
```

## Examples

### Basic Usage

See `examples/example.py` for a complete working example:

```python
from sar_sampling import SarSampler
import pandas as pd

# Load sample data
input_df = pd.read_csv('data/input_table.csv')

# Create sampler
sampler = SarSampler(input_df)

# Generate samples
samples = sampler(1000, method='maximin', seed=123)

# Analyze results
print(f"Generated {samples.size()} samples")
print(f"X range: {samples.data['x'].min():.1f} to {samples.data['x'].max():.1f}")
print(f"Frequency range: {samples.data['frequency'].min():.0f} to {samples.data['frequency'].max():.0f}")

# Save results
samples.to_csv('output_samples.csv')
```
