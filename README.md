# Advanced Machine Learning

A set of labs for Advanced Machine Learning discipline at NURE.

## Project structure

Repository consists of the following components:

- `README.md`: This file.
- `labs/`: Directory containing all labs.

## Dependencies

All dependencies are listed in `pyproject.toml` file and could be installed using `uv`.

## Setup

Clone the repository:

```bash
git clone git@github.com:ivanchukhran/Advanced-Machine-Learning.git
cd Advanced-Machine-Learning/
```

Install uv if not installed:

```bash
pip install uv
```

Create virtual environment and install dependencies:

```bash
uv venv
source .venv/bin/activate
uv sync
```

## Usage

Run one of the labs by executing the following command:

```bash
python labs/<lab_name>/main.py
```

where `<lab_name>` is the name of the lab you want to run.
