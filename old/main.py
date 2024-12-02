import sys
sys.path.append('src')

import json
from pathlib import Path
from models.documentation_analyzer import DocumentationAnalyzer
from models.quality_enhancer import QualityEnhancer
import gradio as gr

def load_dataset(file_path="ml_repos_dataset/dataset.json"):
    """Load the dataset from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def analyze_single_repo(repo_data):
    """Analyze a single repository"""
    analyzer = DocumentationAnalyzer()
    enhancer = QualityEnhancer()
    
    # Run analysis
    analysis_results = analyzer.analyze_documentation_quality(repo_data)
    enhanced_docs = enhancer.enhance_documentation(repo_data)
    
    return {
        'analysis': analysis_results,
        'enhanced': enhanced_docs
    }

def gradio_interface(github_url, readme_content):
    """Interface for Gradio web app"""
    repo_data = {
        'name': github_url.split('/')[-1],
        'url': github_url,
        'readme_content': readme_content
    }
    
    results = analyze_single_repo(repo_data)
    
    # Format results for display
    quality_score = f"{results['analysis']['metrics']['completeness']['overall_score']*100:.1f}%"
    suggestions = '\n'.join(results['analysis']['suggestions'])
    
    enhanced_docs = "# Enhanced Documentation\n\n"
    for section, content in results['enhanced']['sections'].items():
        enhanced_docs += f"## {section.title()}\n{content}\n\n"
    
    return quality_score, suggestions, enhanced_docs

def analyze_all_repos():
    """Analyze all repositories in the dataset"""
    dataset = load_dataset()
    results = []
    
    for repo in dataset:
        print(f"Analyzing {repo['name']}...")
        try:
            result = analyze_single_repo(repo)
            results.append({
                'repo_name': repo['name'],
                'analysis': result['analysis'],
                'enhanced': result['enhanced']
            })
        except Exception as e:
            print(f"Error analyzing {repo['name']}: {str(e)}")
    
    # Save results
    with open('analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Analyzed {len(results)} repositories. Results saved to analysis_results.json")

def launch_gradio():
    """Launch the Gradio interface"""
    interface = gr.Interface(
        fn=gradio_interface,
        inputs=[
            gr.Textbox(label="GitHub Repository URL"),
            gr.Textbox(label="README Content", lines=10)
        ],
        outputs=[
            gr.Textbox(label="Documentation Quality Score"),
            gr.Textbox(label="Improvement Suggestions", lines=5),
            gr.Markdown(label="Enhanced Documentation")
        ],
        title="DocuGen - ML Repository Documentation Analyzer",
        description="""
        Analyze and enhance documentation for machine learning repositories.
        Paste a GitHub repository URL and its README content to get:
        1. Documentation quality assessment
        2. Specific improvement suggestions
        3. Enhanced documentation with better structure and clarity
        """
    )
    interface.launch()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run DocuGen analysis')
    parser.add_argument('--mode', choices=['analyze', 'serve'], default='serve',
                      help='Run mode: analyze all repos or serve Gradio interface')
    
    args = parser.parse_args()
    
    if args.mode == 'analyze':
        analyze_all_repos()
    else:
        launch_gradio()