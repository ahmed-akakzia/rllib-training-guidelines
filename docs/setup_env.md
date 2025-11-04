# Setting up the virtual environment with uv

This first guide aims at setting up your local virtual environment with uv. Follow these steps once you clone the present repository. 

1. Execute the following

```bash
uv venv --python=3.12
source .venv/bin/activate
uv sync --all-groups
```

2. Install `gymnasium-robotics`: We separate the installation of this package to give you a hint of how you could, if needed, enforce dependencies: 

```bash
uv pip install "gymnasium-robotics==1.4.0" --no-deps
```

Actually, we want to work with this specific version of `gymnasium-robotics` and with a lower version of `gymnasium` for the safe of the tutorial. 

