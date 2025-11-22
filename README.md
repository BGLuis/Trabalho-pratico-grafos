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
# Using default paths and graph type (data/node.json → tables/ using integrated graph)
uv run main.py build

# Using custom input and output paths
uv run main.py build -i data/starship.json -o output/

# Using a specific graph type
uv run main.py build -t comments
uv run main.py build -t reviews
uv run main.py build -t closed
uv run main.py build -t integrated  # default
```

Options:
- `-i, --input`: Input JSON file path (default: `data/node.json`)
- `-o, --output`: Output directory for Gephi files (default: `tables/`)
- `-t, --type`: Graph type to build (default: `integrated`)
  - `integrated`: Weighted graph combining all interactions with different weights
  - `comments`: Graph based on PR and issue comments
  - `reviews`: Graph based on PR reviews and merges
  - `closed`: Graph based on issue closures

#### Graph Types Explained

**Integrated (default)**: Combines all interaction types with weighted edges:
- Comment on issue/PR: weight 2
- Issue opened and commented: weight 3
- PR review/approval: weight 4
- PR merge: weight 5

**Comments**: Focuses on comment interactions between users on issues and pull requests.

**Reviews**: Focuses on code review and merge interactions on pull requests.

**Closed**: Focuses on who closes issues opened by others.

### Analyze Graph Metrics

Calculate centrality, structure, and community metrics from generated graphs:

```bash
# Analyze graph from tables directory
uv run main.py analyze -i tables/node/integrated

# Analyze with custom output directory
uv run main.py analyze -i tables/node/integrated -o my_statistics
```

Options:
- `-i, --input`: Input graph directory or base path (required)
- `-o, --output`: Output directory for statistics (default: `statistics`)

#### Metrics Calculated

**Centrality Metrics:**
- Degree Centrality: Number of direct connections (participation level)
- In-Degree Centrality: Who receives interactions
- Out-Degree Centrality: Who initiates interactions
- Betweenness Centrality: Bridge nodes between groups
- Closeness Centrality: Proximity to all other nodes
- PageRank: Influence measure (weighted by connection importance)
- Eigenvector Centrality: Influence based on connections

**Structure and Cohesion Metrics:**
- Density: Proportion of existing vs. possible connections
- Clustering Coefficient: Tendency to form clusters
- Assortativity: Whether highly connected nodes connect with each other

**Community Metrics:**
- Community Detection: Identify groups working together
- Modularity: Quality of community structure
- Bridging Nodes: Users connecting different communities

#### Using Metrics in Gephi

The generated CSV file (`statistics/{graph_name}_metrics.csv`) can be imported into Gephi:

1. Open your graph in Gephi
2. Go to Data Laboratory
3. Click "Import Spreadsheet"
4. Select the metrics CSV file
5. Choose "Append to existing workspace" and "Nodes table"
6. Map the "Id" column to node IDs

All metrics will be added as node attributes, ready for visualization and analysis!

### Help

View available commands and options:

```bash
uv run main.py --help
uv run main.py fetch --help
uv run main.py build --help
uv run main.py analyze --help
```
