"""
code_documentation_analyzer.py: analyzes code and documentation quality in github repositories.
uses codebert for semantic analysis of code-documentation alignment.

key features:
- evaluates documentation completeness
- analyzes code-documentation alignment
- generates improvement suggestions
- provides detailed quality metrics
"""

from transformers import AutoTokenizer, AutoModel
import torch
from github import Github
import base64
from typing import List, Dict
import re
import os
from dotenv import load_dotenv

class CodeDocumentationAnalyzer:
    def __init__(self, github_token: str = None):
        """
        initializes the analyzer with required models and tokens.

        args:
            github_token (str, optional): github api token. defaults to environment variable.
        """
        load_dotenv()
        
        # initialize model for semantic analysis
        model_name = "microsoft/codebert-base"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # set device (mps for m1/m2, cuda for nvidia, cpu for others)
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 
                                 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # initialize github client
        self.github = Github(github_token or os.getenv('GITHUB_TOKEN'))
        
        # define documentation aspects to analyze
        self.doc_aspects = {
            'setup': ['install', 'requirements', 'prerequisites', 'dependencies'],
            'usage': ['usage', 'getting started', 'quickstart', 'example'],
            'api': ['api', 'reference', 'documentation', 'function', 'method'],
            'project_info': ['about', 'overview', 'features', 'description']
        }

    def analyze_repository(self, repo_url: str) -> Dict:
        """
        analyzes repository documentation quality.

        args:
            repo_url (str): github repository url

        returns:
            dict: analysis results including scores and suggestions
        """
        try:
            # extract owner and repo name from url
            _, _, _, owner, repo = repo_url.rstrip('/').split('/')
            repo = self.github.get_repo(f"{owner}/{repo}")
            
            # get readme content
            readme = repo.get_readme()
            readme_content = base64.b64decode(readme.content).decode('utf-8')
            
            # analyze documentation semantically
            doc_scores = self._analyze_documentation_semantics(readme_content)
            
            # generate suggestions based on analysis
            suggestions = self._generate_semantic_suggestions(doc_scores)
            
            return {
                'overall_score': sum(doc_scores.values()) / len(doc_scores),
                'suggestions': suggestions,
                'aspect_scores': doc_scores
            }
            
        except Exception as e:
            print(f"error analyzing repository: {str(e)}")
            return {
                'overall_score': 0.0,
                'suggestions': ['error analyzing repository'],
                'aspect_scores': {}
            }

    def _analyze_documentation_semantics(self, content: str) -> Dict[str, float]:
        """
        analyzes documentation using semantic understanding.

        args:
            content (str): documentation content to analyze

        returns:
            dict: aspect-wise documentation scores
        """
        scores = {}
        
        for aspect, keywords in self.doc_aspects.items():
            chunks = self._get_relevant_chunks(content, keywords)
            if not chunks:
                scores[aspect] = 0.0
                continue
            
            with torch.no_grad():
                inputs = self.tokenizer(
                    chunks,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                ).to(self.device)
                
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
                scores[aspect] = self._calculate_semantic_score(embeddings, aspect)
        
        return scores

    def _get_relevant_chunks(self, content: str, keywords: List[str]) -> List[str]:
        """
        extracts relevant documentation chunks based on keywords.

        args:
            content (str): documentation content
            keywords (list): relevant keywords to search for

        returns:
            list: relevant documentation chunks
        """
        chunks = []
        lines = content.split('\n')
        current_chunk = []
        
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in keywords):
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
            elif current_chunk:
                current_chunk.append(line)
                if len(current_chunk) > 10:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = []
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
            
        return chunks

    def _calculate_semantic_score(self, embeddings: torch.Tensor, aspect: str) -> float:
        """
        calculates semantic similarity score for documentation aspect.

        args:
            embeddings (torch.tensor): document embeddings
            aspect (str): aspect being analyzed

        returns:
            float: semantic similarity score
        """
        ideal_texts = {
            'setup': "clear installation instructions with all dependencies and requirements listed",
            'usage': "comprehensive usage examples with code snippets and explanations",
            'api': "detailed api documentation with function signatures and descriptions",
            'project_info': "clear project overview with features and purpose explained"
        }
        
        with torch.no_grad():
            ideal_inputs = self.tokenizer(
                ideal_texts[aspect],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.device)
            
            ideal_outputs = self.model(**ideal_inputs)
            ideal_embedding = ideal_outputs.last_hidden_state.mean(dim=1)
            
            similarity = torch.nn.functional.cosine_similarity(
                embeddings.mean(dim=0, keepdim=True),
                ideal_embedding
            )
            
            return similarity.item()

    def _generate_semantic_suggestions(self, scores: Dict[str, float]) -> List[str]:
        """
        generates improvement suggestions based on semantic analysis.

        args:
            scores (dict): aspect-wise documentation scores

        returns:
            list: improvement suggestions
        """
        suggestions = []
        
        for aspect, score in scores.items():
            if score < 0.7:
                if aspect == 'setup':
                    suggestions.append("enhance installation instructions with clear steps and requirements")
                elif aspect == 'usage':
                    suggestions.append("add more comprehensive usage examples with code snippets")
                elif aspect == 'api':
                    suggestions.append("include detailed api documentation with function descriptions")
                elif aspect == 'project_info':
                    suggestions.append("provide a clearer project overview and feature description")
        
        return suggestions

    def __del__(self):
        """cleanup resources when analyzer is destroyed"""
        try:
            del self.model
            del self.tokenizer
            torch.cuda.empty_cache()
        except:
            pass