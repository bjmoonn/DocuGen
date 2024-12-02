import sys
sys.path.append('src')
import json
from models.documentation_analyzer import DocumentationAnalyzer
from models.quality_enhancer import QualityEnhancer
from pathlib import Path

def debug_section_extraction(content, keywords):
    """Helper function to debug section extraction"""
    patterns = [
        (f"#{1,6}\\s*{kw}\\b", f"Markdown header: #{kw}") for kw in keywords
    ] + [
        (f"\\n{kw}\\b.*?\\n[=-]+\\s*\\n", f"Underlined header: {kw}") for kw in keywords
    ] + [
        (f"\\n\\*\\*{kw}\\*\\*", f"Bold header: {kw}") for kw in keywords
    ] + [
        (f"\\n{kw}:", f"Colon header: {kw}") for kw in keywords
    ]
    
    content_lower = content.lower()
    found = []
    
    for pattern, desc in patterns:
        matches = list(re.finditer(pattern, content_lower, re.IGNORECASE | re.MULTILINE))
        for match in matches:
            start = max(0, match.start() - 50)  # Get some context
            end = min(len(content), match.end() + 200)  # Get content after header
            found.append({
                'pattern': desc,
                'content': content[start:end],
                'position': match.start()
            })
    
    return found

def test_single_repo(repo_index=0, verbose=True):
    """Test analysis on a single repository from the dataset"""
    # Load dataset
    with open("src/data/dataset.json", 'r') as f:
        dataset = json.load(f)
    
    # Take first repo as example
    test_repo = dataset[repo_index]
    print(f"Testing with repository: {test_repo['name']}")
    
    if verbose:
        print("\nRepository details:")
        print(f"Stars: {test_repo.get('stars', 'N/A')}")
        print(f"URL: {test_repo.get('url', 'N/A')}")
        print(f"Language: {test_repo.get('language', 'N/A')}")
        print(f"README length: {len(test_repo.get('readme_content', ''))}")
    
    # Debug section extraction
    print("\nDEBUGGING SECTION EXTRACTION:")
    readme_content = test_repo.get('readme_content', '')
    
    sections = {
        'installation': ['installation', 'setup', 'getting started'],
        'usage': ['usage', 'how to use', 'examples'],
        'api': ['api', 'documentation', 'reference'],
        'dependencies': ['requirements', 'dependencies', 'prerequisites']
    }
    
    print("\nSearching for sections in README:")
    print("-" * 60)
    for section, keywords in sections.items():
        print(f"\nLooking for {section.upper()} section:")
        found = debug_section_extraction(readme_content, keywords)
        if found:
            for match in found:
                print(f"\nFound {match['pattern']}:")
                print("Content preview:")
                print("-" * 40)
                print(match['content'].strip())
                print("-" * 40)
        else:
            print("No matches found")
    
    # Initialize analyzers
    doc_analyzer = DocumentationAnalyzer()
    quality_enhancer = QualityEnhancer()
    
    # Run documentation analysis
    print("\nRunning documentation analysis...")
    try:
        analysis_results = doc_analyzer.analyze_documentation_quality(test_repo)
        
        if verbose:
            print("\nDetailed Analysis Results:")
            for section, score in analysis_results['metrics']['completeness']['section_scores'].items():
                print(f"- {section}: {score*100:.1f}%")
        
        print(f"\nOverall Score: {analysis_results['metrics']['completeness']['overall_score']*100:.1f}%")
        print("\nSuggestions:")
        for suggestion in analysis_results['suggestions']:
            print(f"- {suggestion}")
    except Exception as e:
        print(f"Error in documentation analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # Run enhancement
    print("\nGenerating enhanced documentation...")
    try:
        enhanced_docs = quality_enhancer.enhance_documentation(test_repo)
        
        # Save results
        output_dir = Path("test_output")
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / "analysis_results.json", 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        with open(output_dir / "enhanced_docs.json", 'w') as f:
            json.dump(enhanced_docs, f, indent=2)
        
        # Create markdown version
        markdown_content = "# Enhanced Documentation\n\n"
        for section, content in enhanced_docs['sections'].items():
            markdown_content += f"{content}\n\n"
        
        with open(output_dir / "enhanced_docs.md", 'w') as f:
            f.write(markdown_content)
        
        if verbose:
            print("\nEnhanced Documentation Preview:")
            for section in ['installation', 'usage', 'api', 'requirements']:
                if section in enhanced_docs['sections']:
                    content = enhanced_docs['sections'][section]
                    preview = content.split('\n')[0:3]
                    print(f"\n{section.upper()}:")
                    print('\n'.join(preview) + "...")
        
        print("\nResults saved to test_output/")
        print("- analysis_results.json")
        print("- enhanced_docs.json")
        print("- enhanced_docs.md")
        
    except Exception as e:
        print(f"Error in documentation enhancement: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import re  # Required for debug_section_extraction
    repo_index = 1  # Try different repo
    test_single_repo(repo_index, verbose=True)