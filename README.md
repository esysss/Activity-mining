# Global Trace Segmentation for Activity Mining

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## About The Project

Global Trace Segmentation is a Python implementation of the algorithm proposed by Günther, Rozinat, and van der Aalst in their research paper ["Activity Mining by Global Trace Segmentation"]. This approach addresses the challenge of extracting meaningful high-level activities from low-level event logs recorded by IT systems.

Many real-world event logs capture events at a very granular level, making process discovery and analysis difficult. This algorithm groups and clusters related low-level events into higher-level activities by analyzing global co-occurrence patterns within process traces. The result is a hierarchical abstraction of event logs that enables better process mining, trace discovery, and process model simplification.

## Built With

- Python 3.8+
- NumPy
- pandas

## Features

- Scans event logs to build a global correlation matrix of event classes
- Builds a hierarchical cluster tree using agglomerative clustering with complete linkage
- Allows adaptive simplification of event logs to user-specified abstraction levels
- Efficient with linear complexity in log size
- Includes utilities to inspect cluster composition and correlation matrix

## Getting Started

### Prerequisites

Install Python dependencies:

pip install numpy pandas


### Installation

Clone the repository and copy `global_trace_segmentation.py` into your working directory.

### Usage

Basic example:

from global_trace_segmentation import GlobalTraceSegmentation

Example event log with list of traces
traces = [
['A', 'B', 'A', 'X', 'Y', 'Z', 'C', 'D', 'E'],
['A', 'A', 'B', 'X', 'Z', 'Y', 'C', 'E', 'D'],
['B', 'A', 'X', 'Y', 'Y', 'Z', 'C', 'D', 'D', 'E']
]

Initialize and fit the model
model = GlobalTraceSegmentation(window_size=6, attenuation_factor=0.8)
model.fit(traces)

Transform logs to abstraction level 3
transformed_traces = model.transform(traces, abstraction_level=3)

Inspect clusters at this level
print(model.get_cluster_info(3))

text

### Parameters

- `window_size` (default=6): Look-back window size for correlation scanning.
- `attenuation_factor` (default=0.8): Controls decay of correlation with distance.
- `abstraction_level`: Hierarchy level to use for abstraction (0 = no abstraction).

## Folder Structure

├── global_trace_segmentation.py # Core implementation
├── examples/ # Example scripts and logs
├── README.md # This file
├── requirements.txt # Python dependencies

text

## Contribution

Contributions are welcome! Please open issues or pull requests to improve the project or fix bugs.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to the original authors Christian W. Günther, Anne Rozinat, and Wil M.P. van der Aalst for the foundational work.
- Inspired by the Process Mining field and the ProM framework.

## References

- Günther, C.W., Rozinat, A., van der Aalst, W.M.P. "Activity Mining by Global Trace Segmentation", BPM Workshops 2010.
