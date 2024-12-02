# DocuGen – Automated Documentation Analysis & Enhancement
Byeongjun Moon, 2024, [bj.moon@usc.edu](mailto:bj.moon@usc.edu)

A machine learning-powered tool that analyzes GitHub repository documentation quality and suggests improvements using state-of-the-art language models.

## Features

- Analyzes README quality using DistilBERT for semantic understanding
- Evaluates code-documentation alignment using CodeBERT
- Generates enhanced documentation using OPT-125m (optimized for Apple Silicon)
- Provides detailed section-by-section analysis with quality scores
- Suggests actionable improvements based on best practices
- Web interface for easy interaction

## Project Structure

```
src/
├── models/                 # Core ML models and analyzers
│   ├── code_documentation_analyzer.py  # CodeBERT-based code analysis
│   ├── quality_enhancer.py            # OPT-125m-based enhancement
│   ├── unified_scorer.py              # Combined scoring system
├── trainers/              # Model training scripts
└── main.py               # Application entry point
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/docugen.git
cd docugen
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your GitHub token
```

## Usage

1. Start the web interface:
```bash
python src/main.py
```

2. Open your browser and navigate to:
```
http://127.0.0.1:7862
```

3. Enter a GitHub repository URL to analyze

## Models

### DistilBERT & CodeBERT
- Used for semantic analysis of documentation content
- Evaluates code-documentation alignment
- Measures documentation completeness and quality
- Analyzes docstrings and code comments

### OPT-125m Enhancer
- Lightweight model optimized for Apple Silicon
- Generates contextual documentation improvements
- Memory-efficient operation
- Enhanced performance on M1/M2 chips

### ReadmeQualityModel
- Custom model for quality scoring
- Fine-tuned on high-quality documentation examples
- Evaluates clarity, completeness, and structure
- Provides section-specific quality metrics

## Development

### Training the Quality Model

1. Add your GitHub token to .env

2. Run the data collection script:
```zsh
python src/models/getting_data.py
```

3. Train the README quality model:
```zsh
python src/trainers/train_readme_model.py
```

4. Run the main script to start the web interface:
```zsh
python src/main.py
```
