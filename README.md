# Trabalho Prático de Grafos

## Prerequisites

This project uses `uv` for dependency management and `git-lfs` for handling large dataset files.

### 1. Install uv

**Linux (Arch):**
```bash
sudo pacman -S uv
# or
yay -S uv
```

**Linux (Ubuntu/Debian) & macOS:**
```bash
# Using the official installer script
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**macOS (Homebrew):**
```bash
brew install uv
```

**Windows:**
```powershell
# Using PowerShell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or using Winget
winget install --id=astral-sh.uv  -e
```

For other systems, check [uv documentation](https://docs.astral.sh/uv/getting-started/installation/).

### 2. Configure Git LFS

The output data (JSON files) can be large, so we use [Git LFS](https://git-lfs.com/).

```bash
# Install Git LFS hooks
git lfs install

# Track large files (if not already tracked)
git lfs track "*.json"
```

## Setup

Initialize the project and pre-commit hooks:

```bash
uv sync
uvx pre-commit install
```

## Usage

The main script supports two commands: `fetch` to extract data from GitHub API and `build` to construct the graph from JSON data.

### Fetch Data from GitHub API

Fetch data from the configured GitHub repository and save it to a JSON file:

```bash
uv run main.py fetch
```

This will save the data to `data/<repo-name>.json`.

### Build Graph from JSON Data

Build a graph from a JSON file and export it to Gephi format:

```bash
# Using default paths (data/node.json → tables/)
uv run main.py build

# Using custom input and output paths
uv run main.py build -i data/starship.json -o output/
```

Options:
- `-i, --input`: Input JSON file path (default: `data/node.json`)
- `-o, --output`: Output directory for Gephi files (default: `tables/`)

### Help

View available commands and options:

```bash
uv run main.py --help
uv run main.py fetch --help
uv run main.py build --help
```
