# HybridIoT — Cooperative Multi-Agent System for Missing Sensor Data Estimation

**Python implementation and experimental evaluation of a cooperative multi-agent system for estimating missing environmental sensor measurements.**

This repository contains work carried out during a research internship at the **Institut de Recherche en Informatique de Toulouse (IRIT)**, within the **SMAC (Systèmes Multi-Agents Coopératifs)** research team, as part of the **neOCampus** project.

HybridIoT aims to reconstruct missing environmental measurements using historical observations and cooperation between heterogeneous sensors, potentially reducing the need for additional physical sensing infrastructure.

The internship focused on developing a modular Python implementation of existing HybridIoT estimation methods, designing a generic data representation, and evaluating the system using real meteorological measurements from Toulouse, France.

## Overview

Urban sensor networks face several limitations, including incomplete geographical coverage, hardware failures, and high deployment and maintenance costs.

HybridIoT addresses these challenges through a cooperative multi-agent approach. Each sensing agent represents a sensor measuring a specific physical quantity, such as temperature, humidity, or atmospheric pressure.

When a measurement is unavailable, an agent can estimate it using:

1. **Historical estimation:** identifying similar patterns in its own past observations.
2. **Cooperative estimation:** exploiting information from other agents measuring different physical quantities.

The HybridIoT research model also defines a third approach based on confidence zones and cooperation between homogeneous sensors. This method is not implemented in the present version.

## Main Contributions

The work carried out during the internship includes:

- **Data acquisition and preprocessing:** developed a configurable Python script for retrieving, aggregating, and standardizing meteorological measurements from Toulouse Métropole Open Data.
- **Generic data representation:** designed reusable dataset abstractions using PyTorch tensors, supporting multiple sensor types, timestamps, missing values, and temporal queries.
- **Multi-agent architecture:** developed and reorganized the `Agent`, `Environment`, and `Estimator` components, separating data management, inter-agent communication, and numerical estimation.
- **Estimation methods:** implemented two previously proposed HybridIoT methods in Python: history-based endogenous estimation and heterogeneous cooperative exogenous estimation.
- **Experimental evaluation:** evaluated the methods against a persistence baseline on real meteorological data, investigated the influence of collaborator selection, and identified limitations of heterogeneous cooperation.

The estimation principles are based on earlier HybridIoT research, particularly the work of Davide Andrea Guastella and collaborators. The contributions described above concern their implementation, integration, adaptation, and experimental evaluation during this internship.

## System Architecture

The implementation separates the system into three main components:

| Component | Responsibility |
|---|---|
| `Agent` | Maintains local sensor observations, detects invalid measurements, constructs temporal context windows, and requests estimations |
| `Environment` | Registers agents, manages their interactions, and selects collaborators |
| `Estimator` | Performs numerical calculations independently of agent and environment objects |

Agents do not communicate directly with one another. Requests are mediated by the `Environment`, which selects collaborators and manages the exchange of information.

The architecture also includes:

- `Sensor` and `Position` for sensor metadata and geographical positioning.
- `HybridIoTDataset`, an abstract representation of multidimensional sensor data.
- `MeteoDataset`, a specialized implementation for meteorological measurements.
- `DataWindow`, a structure for extracting and exchanging timestamped sensor data.

Measurements are represented internally using PyTorch tensors, allowing indexing by data type, timestamp, and sensor.

The implementation supports multiple `Environment` instances, providing the architectural foundation for future confidence-zone mechanisms. Those mechanisms are not yet implemented.

## Estimation Methods

### 1. History-Based Endogenous Estimation

The first method estimates a missing measurement using the agent's own historical observations.

The agent:

1. Extracts its current temporal context window.
2. Searches its historical observations for similar windows.
3. Selects up to 10 of the most similar candidates.
4. Calculates a similarity-weighted estimate of the expected variation.
5. Adds this variation to the last observed value.

The estimation follows:

\[
\hat{x}_t = x_{t-1} + w_t
\]

where \(w_t\) represents the estimated variation derived from historical patterns.

In the evaluated implementation, context windows contain eight observations, corresponding to two hours of measurements at the dataset's 15-minute sampling frequency.

### 2. Heterogeneous Cooperative Exogenous Estimation

The second method extends historical estimation through cooperation with agents measuring different physical quantities.

The process consists of three stages:

1. The requesting agent identifies historical periods resembling its current situation.
2. Collaborating agents analyze their own measurements at the corresponding timestamps and return estimated variations.
3. The requesting agent combines the received contributions using a histogram-based procedure and its endogenous estimate.

Collaborators exchange timestamps and estimated variations rather than raw sensor measurements during this estimation process.

The default collaboration ratio is 30% of the eligible agents. Collaborators are selected randomly with probabilities influenced by geographical proximity.

## Experimental Evaluation

### Dataset and Protocol

The evaluation uses meteorological measurements from the [Toulouse Métropole Open Data portal](https://data.toulouse-metropole.fr/).

| Property | Description |
|---|---|
| Period | 2020 |
| Weather stations | 6 |
| Measurement types | Temperature, humidity, atmospheric pressure |
| Sampling frequency | 15 minutes |
| Agents | 18 (6 stations × 3 measurement types) |
| Evaluated predictions | 148,651 |

The experiments use a hold-out evaluation protocol: each known measurement is estimated using only preceding observations, and the prediction is compared with its actual value.

The evaluation compares:

- **Persistence:** the last available measurement, used as a baseline.
- **History-based estimation:** prediction using previous observations from the same agent.
- **Exogenous estimation:** prediction combining endogenous estimation with contributions from heterogeneous agents.

Performance is measured using Mean Absolute Error (MAE) and normalized MAE (nMAE). Normalization by the standard deviation of the reference measurements enables comparisons across physical quantities with different units.

### Results

The following table reports normalized MAE, aggregated by measurement type. **Lower values indicate better performance.**

| Measurement type | Persistence | History-based | Exogenous |
|---|---:|---:|---:|
| Temperature | 0.0387 | **0.0316** | 0.0708 |
| Humidity | **0.0738** | 0.0765 | 0.0831 |
| Pressure | **0.0158** | 0.0292 | 0.0226 |

The main observations are:

- **Temperature:** history-based estimation achieves approximately 18% lower normalized MAE than persistence.
- **Humidity:** persistence performs slightly better than the two estimation methods.
- **Pressure:** persistence remains the most accurate approach, largely because pressure measurements frequently remain unchanged between consecutive samples.
- **Heterogeneous cooperation:** exogenous estimation does not consistently improve upon the history-based method.

Further analysis showed that the cooperative component alone did not outperform the history-based estimator in the aggregated comparisons.

### Analysis of Cooperative Estimation

A central finding concerns the **incompatibility of the numerical scales of the exchanged variations**.

Collaborating agents measure different physical quantities, but their calculated variations retain their original units.

For example, a pressure sensor can produce variations expressed in pascals, while a temperature sensor operates in degrees Celsius. Directly transferring such variations can introduce significant errors, even when the collaborating agents have similar temporal patterns.

The histogram-based combination can reject some inconsistent contributions, but it cannot guarantee a meaningful estimate when all available contributions have incompatible scales.

Experiments with collaboration ratios of 10%, 30%, 50%, and 100% also showed that using more collaborators does not necessarily provide substantial improvements beyond a certain point.

These findings suggest that **normalizing exchanged variations according to each sensor's characteristic scale** could improve heterogeneous cooperation. This remains a proposed improvement and has not been implemented or experimentally validated in this repository.

### Limitations and Future Work

The main directions identified during the evaluation include:

- Normalizing variations exchanged between heterogeneous agents.
- Evaluating estimation performance during prolonged sensor failures rather than isolated missing measurements.
- Investigating adaptive context-window sizes.
- Comparing alternative collaborator-selection strategies.
- Implementing confidence-zone-based homogeneous cooperation.
- Evaluating virtual sensor deployment strategies.

## Repository Structure

```text
.
├── hybridiot/                   # Core Python implementation
│   ├── agent.py
│   ├── environment.py
│   ├── estimator.py
│   ├── sensor.py
│   ├── position.py
│   ├── base_dataset.py
│   ├── meteo_dataset.py
│   ├── data_window.py
│   └── __init__.py
├── notebooks/                   # Experimental evaluation and validation
│   ├── analyse_finale.ipynb
├── scripts/
│   └── download_toulouse_meteo.py
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```


## Installation

**Requirements:** Python and the dependencies listed in `requirements.txt`.

Clone the repository:

```bash
git clone https://github.com/guides22/HybridIoT.git
cd HybridIoT
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows, activate the environment with `.venv\Scripts\activate`.

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

The main dependencies include PyTorch, NumPy, pandas, Requests, Matplotlib, and Jupyter.

## Reproducing the Experiments

The meteorological datasets and generated experimental results are not included in the repository because of their size.

The following procedure explains how to download the source data and rerun the evaluation.

### Step 1 — Download the meteorological data

From the repository root, create the data directory:

```bash
mkdir -p data
```

Then execute:

```bash
python scripts/download_toulouse_meteo.py --config "archive_2020(35.000)"
```

The script uses the Toulouse Métropole Open Data API to retrieve measurements from the configured stations, standardize their columns, associate geographical metadata, and aggregate the measurements into a single CSV file:

```text
data/toulouse_meteo_2020(35.000).csv
```

The script also supports dataset discovery, column inspection, and additional predefined configurations:

```bash
python scripts/download_toulouse_meteo.py --list
python scripts/download_toulouse_meteo.py --search "archive-2020"
```

The data remain the property of their respective publishers and are subject to the terms of the original data source.

### Step 2 — Run the evaluation notebook

Start Jupyter from the `notebooks` directory:

```bash
cd notebooks
jupyter notebook analyse_finale.ipynb
```

Execute the notebook cells in order.

The notebook:

1. Loads the meteorological dataset and initializes 18 agents.
2. Evaluates the history-based and exogenous estimation methods alongside persistence.
3. Saves intermediate computations to avoid repeating expensive calculations.
4. Computes performance metrics and generates comparative visualizations.
5. Analyzes the influence of collaborator selection and collaboration ratios.

Generated datasets, results, and pickle caches are stored locally in `data/` and excluded from Git versioning.

**Computational cost:** the complete experimental evaluation required approximately nine hours in the original development environment. Actual execution time depends on the hardware and software environment.

**Reproducibility note:** collaborator selection includes random sampling without a fixed seed in the original experiments. Consequently, rerunning the evaluation may produce results that differ from those reported above.

The numerical results presented in this README correspond to the original internship evaluation.

## Research Context and Acknowledgments

This repository contains work developed by **Guillaume Andrea Desaphy** during a Master 1 research internship at **IRIT**, within the **SMAC research team**, from May to July 2026.

The internship was supervised by:

- **Valérie Camps** — IRIT, SMAC research team.
- **Davide Andrea Guastella** — technical supervision and previous research on HybridIoT.

HybridIoT is an existing research project developed within the neOCampus research context. Its underlying concepts and estimation methodologies originate from earlier work by Davide Andrea Guastella and collaborators, including Valérie Camps and Marie-Pierre Gleizes.

The Python implementation, data handling, software adaptations, and experimental evaluation documented here were carried out as part of the internship, building upon this prior research and pre-existing project code.

The project acknowledges IRIT, the SMAC research team, neOCampus, the University of Toulouse, and the institutions and researchers involved in the original HybridIoT work.

### Related Research

1. D. Guastella, *Dynamic Learning of the Environment for Eco-Citizen Behavior*, PhD thesis, 2020.
2. D. A. Guastella, V. Camps, and M.-P. Gleizes, *A Cooperative Multi-Agent System for Crowd Sensing Based Estimation in Smart Cities*, IEEE Access, 2020.
3. D. A. Guastella and E. Pournaras, *Cooperative Multi-Agent Traffic Monitoring Can Reduce Camera Surveillance*, IEEE Access, 2023.
4. D. A. Guastella, V. Camps, and M.-P. Gleizes, *Estimation d'informations environnementales avec le système HybridIoT : un cas d'étude sur la ville de Toulouse*, Revue Ouverte d'Intelligence Artificielle, 2024.

## License

The source code is distributed under the **GNU General Public License v3.0 (GPL-3.0)**. See [LICENSE](LICENSE) for details.

Publication of the internship code was authorized by the relevant IRIT representatives subject to open-source licensing, appropriate attribution, and verification that no confidential or proprietary third-party materials are included.
