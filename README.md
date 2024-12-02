# DocuGen – Automated Documentation Generation for Machine Learning Papers
Byeongjun Moon, 2024, [bj.moon@usc.edu](mailto:bj.moon@usc.edu)
A machine learning-powered tool that analyzes GitHub repository documentation quality using state-of-the-art language models.

## Features

- Analyzes README quality using DistilBERT for semantic understanding
- Evaluates code-documentation alignment using CodeBERT
- Generates enhanced documentation using DistilGPT2
- Provides detailed section-by-section analysis
- Suggests improvements based on best practices
- Web interface for easy interaction

## Project Structure

```
src/
├── app/                    # Web application components
├── data/                   # Dataset and data handling
├── models/                 # Core ML models and analyzers
│   ├── code_documentation_analyzer.py  # CodeBERT-based code analysis
│   ├── documentation_analyzer.py       # Documentation quality analysis
│   ├── quality_enhancer.py            # DistilGPT2-based enhancement
│   ├── readme_quality_model.py        # DistilBERT-based quality scoring
│   └── unified_scorer.py              # Combined scoring system
├── trainers/              # Model training scripts
├── utils/                 # Utility functions
└── main.py               # Application entry point
```

## Installation

1. Clone the repository:
```bash
git clone [your-repo-url]
cd [repo-name]
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

### DistilBERT Quality Model
- Used for semantic analysis of README content
- Fine-tuned on high-quality documentation examples
- Evaluates clarity, completeness, and structure

### CodeBERT Analyzer
- Analyzes alignment between code and documentation
- Evaluates docstrings and code comments
- Measures documentation coverage

### DistilGPT2 Enhancer
- Generates improvement suggestions
- Enhances existing documentation sections
- Maintains project-specific context

## Development

### Training Models

1. Prepare training data:
```bash
python src/utils/data_preprocessor.py
```

2. Train the README quality model:
```bash
python src/trainers/train_readme_model.py
```