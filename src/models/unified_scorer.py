"""
unified_scorer.py: provides a unified scoring system for repository documentation quality.
combines code analysis and documentation assessment into a single quality metric.

key features:
- analyzes repository documentation completeness
- evaluates code-documentation alignment
- generates actionable improvement suggestions
- provides detailed scoring breakdown
"""

from models.code_documentation_analyzer import CodeDocumentationAnalyzer

class UnifiedScorer:
    def __init__(self):
        """
        initializes the unified scorer with required analyzers.
        uses code documentation analyzer for comprehensive analysis.
        """
        self.analyzer = CodeDocumentationAnalyzer()

    def get_unified_score(self, repo_url: str, readme_content: str) -> dict:
        """
        analyzes repository and generates unified quality score.

        args:
            repo_url (str): github repository url
            readme_content (str): repository's readme content

        returns:
            dict: contains scores, suggestions, and analysis details
        """
        try:
            # analyze basic readme structure
            has_installation = 'installation' in readme_content.lower()
            has_usage = 'usage' in readme_content.lower()
            has_api = 'api' in readme_content.lower() or '## functions' in readme_content.lower()
            has_code_samples = '```' in readme_content
            
            # calculate completeness score
            completeness = sum([
                has_installation,
                has_usage,
                has_api,
                has_code_samples
            ]) / 4.0
            
            # get detailed analysis from analyzer
            analyzer_results = self.analyzer.analyze_repository(repo_url)
            
            # combine all scores
            aspect_scores = {
                'documentation_structure': completeness,
                'code_documentation': analyzer_results.get('code_doc_score', 0.5),
                'clarity': analyzer_results.get('clarity_score', 0.5),
                'examples': 1.0 if has_code_samples else 0.0
            }
            
            # generate suggestions
            suggestions = []
            if not has_installation:
                suggestions.append("add installation instructions")
            if not has_usage:
                suggestions.append("add usage examples")
            if not has_api:
                suggestions.append("add api documentation")
            if not has_code_samples:
                suggestions.append("add code examples")
                
            # add analyzer suggestions
            suggestions.extend(analyzer_results.get('suggestions', []))
            
            # calculate overall score
            overall_score = sum(aspect_scores.values()) / len(aspect_scores)
            
            return {
                'overall_score': overall_score,
                'completeness': completeness,
                'has_code_samples': has_code_samples,
                'has_api_docs': has_api,
                'aspect_scores': aspect_scores,
                'suggestions': suggestions,
                'missing_sections': [
                    f"add {section}" for section, present in {
                        'installation': has_installation,
                        'usage': has_usage,
                        'api': has_api
                    }.items() if not present
                ]
            }
            
        except Exception as e:
            print(f"error in scoring: {str(e)}")
            return {
                'overall_score': 0.0,
                'completeness': 0.0,
                'has_code_samples': False,
                'has_api_docs': False,
                'aspect_scores': {
                    'documentation_structure': 0.0,
                    'code_documentation': 0.0,
                    'clarity': 0.0,
                    'examples': 0.0
                },
                'suggestions': ['error analyzing repository'],
                'missing_sections': []
            }