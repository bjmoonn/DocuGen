"""
main.py: main entry point for the documentation analysis and enhancement system.

key features:
- provides gradio web interface for easy interaction
- analyzes github repository documentation quality
- generates enhanced documentation suggestions
- handles resource cleanup and error management

dependencies:
- gradio: for web interface
- torch: for ml model operations
- github: for repository access
- transformers: for nlp models

usage:
    python src/main.py

the application will start a local server at http://127.0.0.1:7862
"""

from dotenv import load_dotenv
load_dotenv()

from models.unified_scorer import UnifiedScorer
from models.quality_enhancer import QualityEnhancer
import gradio as gr
import base64
from github import Github
import os
import torch
import gc
import logging

# set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# create a single scorer instance to be reused
scorer = None

def initialize_scorer():
    """initialize the unified scorer singleton"""
    global scorer
    try:
        logger.info("initializing scorer...")
        if scorer is None:
            scorer = UnifiedScorer()
        logger.info("scorer initialized successfully")
        return scorer
    except Exception as e:
        logger.error(f"error initializing scorer: {str(e)}")
        raise

def format_score(score):
    """
    format score as percentage with color indicator
    
    args:
        score (float): score value between 0 and 1
        
    returns:
        str: formatted score with color indicator
    """
    if score >= 0.7:
        return f"🟢 {score*100:.1f}%"
    elif score >= 0.4:
        return f"🟡 {score*100:.1f}%"
    else:
        return f"🔴 {score*100:.1f}%"

def analyze_repo(github_url):
    """
    analyze github repository documentation
    
    args:
        github_url (str): github repository url
        
    returns:
        tuple: (score, report, enhanced_docs)
    """
    try:
        logger.info(f"starting analysis for: {github_url}")
        
        if not github_url.startswith("https://github.com/"):
            logger.warning("invalid github url format")
            return 0.0, "invalid github url format", ""
        
        # get readme content
        logger.info("fetching readme content...")
        g = Github(os.getenv('GITHUB_TOKEN'))
        _, _, _, owner, repo_name = github_url.rstrip('/').split('/')
        repo = g.get_repo(f"{owner}/{repo_name}")
        readme = repo.get_readme()
        readme_content = base64.b64decode(readme.content).decode('utf-8')
        logger.info("readme content fetched successfully")
        
        # get scores and generate report
        logger.info("running analysis...")
        scorer = initialize_scorer()
        results = scorer.get_unified_score(github_url, readme_content)
        logger.info("analysis completed")
        
        # generate enhanced documentation if needed
        logger.info("generating enhanced documentation...")
        enhancer = QualityEnhancer()
        repo_data = {
            'name': repo_name,
            'url': github_url,
            'readme_content': readme_content
        }
        enhanced_docs = enhancer.enhance_documentation(repo_data, results)
        logger.info("enhanced documentation generated")
        
        # format the analysis report
        report = f"""
# documentation analysis report

## overall quality score: {format_score(results['overall_score'])}

## detailed scores:
{chr(10).join([f'- {aspect}: {format_score(score)}' for aspect, score in results['aspect_scores'].items()])}

## areas for improvement:
{chr(10).join([f'- {s}' for s in results['suggestions']]) if results['suggestions'] else '- no improvements needed'}

## analysis details:
- documentation completeness: {results['completeness']}
- code examples: {'present' if results.get('has_code_samples') else 'missing'}
- api documentation: {'present' if results.get('has_api_docs') else 'missing'}
"""
        
        return (
            results['overall_score'],
            report.strip(),
            enhanced_docs if enhanced_docs else "no documentation enhancements needed."
        )
        
    except Exception as e:
        logger.error(f"error during analysis: {str(e)}")
        cleanup()
        return 0.0, f"error: {str(e)}", ""

def cleanup():
    """cleanup resources and memory"""
    global scorer
    try:
        logger.info("running cleanup...")
        if scorer is not None:
            del scorer
            scorer = None
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        logger.info("cleanup completed")
    except Exception as e:
        logger.error(f"error during cleanup: {str(e)}")

def create_interface():
    """create and configure gradio interface"""
    logger.info("creating gradio interface...")
    return gr.Interface(
        fn=analyze_repo,
        inputs=gr.Textbox(
            label="GitHub repository url",
            placeholder="https://github.com/username/repo"
        ),
        outputs=[
            gr.Number(label="Quality score (0-1)"),
            gr.Markdown(label="Analysis report"),
            gr.Markdown(label="Enahnced documentation")
        ],
        title="DocuGen – Automated Documentation Analysis & Enhancement",
        description="""
Analyze Github repository documentation using AI:
- DistilBERT for semantic documentation analysis
- CodeBERT for code-documentation alignment
- Automatic documentation enhancement suggestions
""",
        allow_flagging="never"
    )

if __name__ == "__main__":
    try:
        logger.info("starting application...")
        iface = create_interface()
        logger.info("launching gradio interface...")
        iface.launch(
            server_name="127.0.0.1",
            server_port=7862,
            share=False
        )
    except Exception as e:
        logger.error(f"error launching application: {str(e)}")
    finally:
        cleanup()