from pathlib import Path
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import networkx as nx

class StructureAnalyzer:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def analyze_repo_structure(self, repo_path):
        """Analyze repository structure and organization"""
        results = {
            'components': self._identify_components(repo_path),
            'dependency_graph': self._analyze_dependencies(repo_path),
            'modularity_score': 0.0,
            'suggestions': []
        }
        
        # Calculate modularity score
        results['modularity_score'] = self._calculate_modularity(results['dependency_graph'])
        
        # Generate structure-based suggestions
        results['suggestions'] = self._generate_structure_suggestions(results)
        
        return results
    
    def _identify_components(self, repo_path):
        """Identify key ML project components"""
        components = {
            'data_loading': [],
            'model_definition': [],
            'training': [],
            'evaluation': [],
            'inference': [],
            'utils': []
        }
        
        # Analyze files and categorize them
        for file_path in Path(repo_path).rglob('*.py'):
            category = self._categorize_file(file_path)
            if category:
                components[category].append(str(file_path))
                
        return components
    
    def _analyze_dependencies(self, repo_path):
        """Create dependency graph between components"""
        G = nx.DiGraph()
        
        # Analyze imports and create graph
        for file_path in Path(repo_path).rglob('*.py'):
            imports = self._extract_imports(file_path)
            for imp in imports:
                G.add_edge(str(file_path), imp)
                
        return nx.node_link_data(G)
    
    def _calculate_modularity(self, graph_data):
        """Calculate modularity score of the codebase"""
        G = nx.node_link_graph(graph_data)
        communities = nx.community.greedy_modularity_communities(G.to_undirected())
        return nx.community.modularity(G.to_undirected(), communities)
    
    def _generate_structure_suggestions(self, analysis_results):
        """Generate suggestions for improving repository structure"""
        suggestions = []
        
        # Check component coverage
        for component, files in analysis_results['components'].items():
            if not files:
                suggestions.append(f"Add dedicated {component.replace('_', ' ')} module")
                
        # Check modularity
        if analysis_results['modularity_score'] < 0.5:
            suggestions.append("Consider reorganizing code to improve modularity")
            
        return suggestions
    
    def _categorize_file(self, file_path):
        """Categorize Python file based on content and name"""
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Use semantic similarity to categorize
        embeddings = self.model.encode([content])
        # Implementation of categorization logic
        
        return self._determine_category(file_path.name, embeddings)
    
    def _extract_imports(self, file_path):
        """Extract import statements from Python file"""
        imports = []
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith('import') or line.startswith('from'):
                    imports.append(line.strip())
        return imports

# Usage example:
if __name__ == "__main__":
    analyzer = StructureAnalyzer()
    results = analyzer.analyze_repo_structure("repository_path")
    with open("structure_analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)