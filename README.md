# Global Trace Segmentation for Activity Mining

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## About The Project

Global Trace Segmentation is a Python implementation of the algorithm proposed by Günther, Rozinat, and van der Aalst in their research paper ["Activity Mining by Global Trace Segmentation"](https://link.springer.com/chapter/10.1007/978-3-642-11590-2_12). This approach addresses the challenge of extracting meaningful high-level activities from low-level event logs recorded by IT systems.

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
