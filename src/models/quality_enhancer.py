"""
quality_enhancer.py: generates enhanced documentation for gh repos using templates
and code analysis. designed for *efficient* operation on apple silicon and other architectures.

key features:
- analyzes repository structure and code
- extracts classes, functions, and their documentation
- generates structured markdown documentation
- uses lightweight models for silicon compatibility
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re
from typing import Dict
import os
from github import Github
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn


class ReadmeQualityModel(nn.Module):
    """model for predicting readme quality scores"""
    def __init__(self, model_name='microsoft/codebert-base'):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.regressor = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # use [CLS] token
        return self.regressor(pooled_output)
    
class QualityEnhancer:
    def __init__(self):
        """
        initializes the quality enhancer with a lightweight model configuration.
        uses mps (metal performance shaders) when available on apple silicon.
        """
        try:
            # lightweight model selection for better performance
            model_name = "facebook/opt-125m"
            self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
            
            # model initialization with memory optimization
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            ).to(self.device)
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id
            self.model.eval()
            
            # load readme quality model
            weights_path = Path('src/models/weights')
            if (weights_path / 'readme_quality_model.pt').exists():
                self.readme_model = ReadmeQualityModel().to(self.device)
                self.readme_model.load_state_dict(
                    torch.load(
                        weights_path / 'readme_quality_model.pt',
                        weights_only=True
                    )
                )
                self.readme_tokenizer = AutoTokenizer.from_pretrained(
                    weights_path / 'tokenizer'
                )
                self.readme_model.eval()
            else:
                print("warning: readme quality model not found")
                self.readme_model = None
                
        except Exception as e:
            print(f"error initializing quality enhancer: {str(e)}")
            raise
        
        # documentation section templates
        self.section_templates = {
            'installation': """
# installation

here's how to install {package_name}:

## quick install
```bash
{install_command}
```

## from source
```bash
git clone {repo_url}
cd {package_name}
pip install -e .
```
""",
            'usage': """
# usage

## basic example
```python
import {package_name}

{usage_example}
```

for more examples, see the examples/ directory.
""",
            'api': """
# api reference

## main classes and functions

{api_content}

for complete api documentation, visit our documentation site.
""",
            'dependencies': """
# requirements

## system requirements
- python {python_version} or later
{system_requirements}

## python dependencies
```
{python_dependencies}
```
"""
        }

    def _gather_code_context(self, repo_url: str) -> Dict:
        """
        analyzes repository to extract information about classes and functions.
        
        args:
            repo_url (str): github repository url
            
        returns:
            dict: contains lists of classes and functions with their documentation
        """
        try:
            # parse github url
            _, _, _, owner, repo = repo_url.rstrip('/').split('/')
            g = Github(os.getenv('GITHUB_TOKEN'))
            repo = g.get_repo(f"{owner}/{repo}")
            
            classes = []
            functions = []
            files_to_process = []
            
            # recursively collect python files
            contents = repo.get_contents("")
            while contents:
                file_content = contents.pop(0)
                if file_content.type == "dir":
                    contents.extend(repo.get_contents(file_content.path))
                elif file_content.path.endswith(".py"):
                    files_to_process.append(file_content)
            
            # analyze each python file
            for file in files_to_process:
                content = file.decoded_content.decode('utf-8')
                
                # extract classes
                for match in re.finditer(r'class\s+(\w+)\s*(?:\([^)]*\))?\s*:', content):
                    class_name = match.group(1)
                    class_end = content.find('class', match.end())
                    if class_end == -1:
                        class_end = len(content)
                    class_content = content[match.start():class_end]
                    docstring = re.search(r'"""(.*?)"""', class_content, re.DOTALL)
                    
                    classes.append({
                        'name': class_name,
                        'file': file.path,
                        'docstring': docstring.group(1).strip() if docstring else None
                    })
                
                # extract functions (excluding private ones)
                for match in re.finditer(r'def\s+(\w+)\s*\([^)]*\)\s*(?:->.*?)?:', content):
                    func_name = match.group(1)
                    if not func_name.startswith('_'):
                        func_end = content.find('def', match.end())
                        if func_end == -1:
                            func_end = len(content)
                        func_content = content[match.start():func_end]
                        docstring = re.search(r'"""(.*?)"""', func_content, re.DOTALL)
                        
                        functions.append({
                            'name': func_name,
                            'file': file.path,
                            'docstring': docstring.group(1).strip() if docstring else None
                        })
            
            return {'classes': classes, 'functions': functions}
            
        except Exception as e:
            print(f"error gathering code context: {str(e)}")
            return {'classes': [], 'functions': []}

    def _generate_section_content(self, section_type: str, context: Dict) -> str:
        """
        generate content for a specific section using templates.
        
        args:
            section_type (str): type of section to generate
            context (dict): repository context including:
                - name (str): repository name
                - url (str): repository url
                - classes (list): extracted classes
                - functions (list): extracted functions
        
        returns:
            str: generated section content
        """
        try:
            # ensure all required context values exist with defaults
            package_name = context.get('name', 'package').lower()
            repo_url = context.get('url', '')
            classes = context.get('classes', [])
            functions = context.get('functions', [])

            if section_type == 'installation':
                return self.section_templates['installation'].format(
                    package_name=package_name,
                    install_command=f"pip install {package_name}",
                    repo_url=repo_url
                )
            
            elif section_type == 'usage':
                # generate basic usage example based on package name
                usage_example = f"""
# basic example
my_instance = {package_name}.Client()
result = my_instance.process()
print(result)
"""
                return self.section_templates['usage'].format(
                    package_name=package_name,
                    usage_example=usage_example
                )
            
            elif section_type == 'api':
                # generate api documentation from extracted classes and functions
                api_content = ""
                
                if classes:
                    api_content += "### classes\n\n"
                    for cls in classes:
                        api_content += f"#### {cls['name']}\n{cls.get('docstring', 'no documentation available')}\n\n"
                
                if functions:
                    api_content += "### functions\n\n"
                    for func in functions:
                        api_content += f"#### {func['name']}\n{func.get('docstring', 'no documentation available')}\n\n"
                
                if not api_content:
                    api_content = "documentation will be added soon."
                
                return self.section_templates['api'].format(api_content=api_content)
            
            elif section_type == 'dependencies':
                return self.section_templates['dependencies'].format(
                    python_version="3.7",
                    system_requirements="- no special system requirements",
                    python_dependencies="""
requests>=2.25.1
numpy>=1.19.2
pandas>=1.2.0
"""
                )
            
            return ""
                
        except Exception as e:
            print(f"error generating {section_type} section: {str(e)}")
            return f"error generating {section_type} section"

    def predict_readme_quality(self, content: str) -> float:
        """
        predict quality score for readme content
        
        args:
            content (str): readme content to analyze
            
        returns:
            float: predicted quality score
        """
        if not self.readme_model:
            return 0.5
            
        try:
            with torch.no_grad():
                inputs = self.readme_tokenizer(
                    content,
                    max_length=512,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                ).to(self.device)
                
                score = self.readme_model(
                    inputs['input_ids'],
                    inputs['attention_mask']
                ).squeeze().item()
                
                return max(0.0, min(1.0, score))  # clip to 0-1 range
                
        except Exception as e:
            print(f"error predicting readme quality: {str(e)}")
            return 0.5

    def enhance_documentation(self, repo_data: Dict, scores: Dict) -> str:
        """
        enhances repository documentation based on analysis scores.
        
        args:
            repo_data (dict): repository information
            scores (dict): analysis scores and suggestions
            
        returns:
            str: enhanced documentation in markdown format
        """
        try:
            missing_sections = scores.get('missing_sections', [])
            
            if not missing_sections:
                return "no enhancement needed. documentation appears to be complete."
            
            # gather code context for api documentation
            code_context = self._gather_code_context(repo_data['url']) if 'api' in str(missing_sections).lower() else {}
            enhanced_sections = {}
            
            # use model to predict quality and adjust suggestions
            quality_score = self.predict_readme_quality(repo_data.get('readme_content', ''))
            if quality_score < 0.3:
                missing_sections.append("improve overall documentation quality")
            elif quality_score < 0.6:
                missing_sections.append("consider enhancing documentation clarity")
                
            # generate each missing section
            for section in ['installation', 'usage', 'api', 'dependencies']:
                if any(s.lower().startswith(f"add {section}") or 
                      s.lower().startswith(f"enhance {section}") 
                      for s in missing_sections):
                    
                    context = {
                        'name': repo_data.get('name', ''),
                        'url': repo_data.get('url', ''),
                        'classes': code_context.get('classes', []),
                        'functions': code_context.get('functions', [])
                    }
                    
                    section_content = self._generate_section_content(section, context)
                    if section_content:
                        enhanced_sections[section] = section_content
            
            if not enhanced_sections:
                return "error generating enhanced documentation sections."
            
            # combine all enhanced sections
            enhanced_content = "# enhanced readme sections\n\n"
            enhanced_content += "the following sections have been enhanced or added:\n\n"
            
            for section, content in enhanced_sections.items():
                enhanced_content += f"{content}\n\n"
            
            return enhanced_content
            
        except Exception as e:
            print(f"error enhancing documentation: {str(e)}")
            return "error generating enhanced documentation"

    def __del__(self):
        """cleanup resources when the enhancer is destroyed"""
        try:
            del self.model
            del self.tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass