import gradio as gr
import json
from pathlib import Path
from models.documentation_analyzer import DocumentationAnalyzer
from models.quality_enhancer import QualityEnhancer

class DocuGenApp:
    def __init__(self):
        self.doc_analyzer = DocumentationAnalyzer()
        self.quality_enhancer = QualityEnhancer()
        
    def analyze_repository(self, github_url, readme_content):
        """Analyze repository documentation and structure"""
        # Process input
        repo_data = {
            'name': github_url.split('/')[-1],
            'url': github_url,
            'readme_content': readme_content
        }
        
        # Run analysis
        doc_analysis = self.doc_analyzer.analyze_documentation_quality(repo_data)
        
        # Generate enhanced documentation
        enhanced_docs = self.quality_enhancer.enhance_documentation(repo_data)
        
        # Format results
        results = {
            'Documentation Quality': f"{doc_analysis['metrics']['completeness']['overall_score']*100:.1f}%",
            'Suggestions': '\n'.join(doc_analysis['suggestions']),
            'Enhanced Documentation': self._format_enhanced_docs(enhanced_docs)
        }
        
        return (
            results['Documentation Quality'],
            results['Suggestions'],
            results['Enhanced Documentation']
        )
    
    def _format_enhanced_docs(self, enhanced_docs):
        """Format enhanced documentation for display"""
        formatted = "# Enhanced Documentation\n\n"
        
        for section, content in enhanced_docs['sections'].items():
            formatted += f"## {section.title()}\n{content}\n\n"
            
        return formatted

# Create Gradio interface
def create_interface():
    app = DocuGenApp()
    
    interface = gr.Interface(
        fn=app.analyze_repository,
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
    
    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch()