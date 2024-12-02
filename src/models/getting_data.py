import pandas as pd
from github import Github
import json
from pathlib import Path
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# 38 repos is the number of decent quality repos that can be collected with the current github token
def collect_ml_repos(repos_per_query=38):
    """
    Collect machine learning repositories focused on software packages/libraries
    """
    github_token = os.getenv('GITHUB_TOKEN')
    if not github_token:
        raise ValueError("GITHUB_TOKEN not found in environment variables")
        
    g = Github(github_token)

    # Create full directory path
    dataset_dir = Path("src/data")
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # More targeted search queries for software packages
    search_queries = [
        "machine learning library language:python stars:>100",
        "deep learning framework language:python stars:>100", 
        "ML toolkit language:python stars:>100",
        "AI library language:python stars:>100",
        "data science library language:python stars:>100"
    ]

    # Keywords that suggest a paper implementation rather than a library
    implementation_keywords = [
        "implementation", "paper", "reproduction", "pytorch implementation",
        "tensorflow implementation", "keras implementation"
    ]

    # Keywords that suggest a proper software package
    library_keywords = [
        "library", "framework", "toolkit", "package", "sdk", "api"
    ]

    repos_data = []
    
    for query in search_queries:
        print(f"\nSearching for: {query}")
        query_count = 0
        skip_count = 0  # Track consecutive skips
        duplicate_count = 0  # Track consecutive duplicates
        no_setup_count = 0  # Track consecutive no setup.py

        while query_count < repos_per_query:
            try:
                repositories = g.search_repositories(
                    query=query,
                    sort="stars",
                    order="desc"
                )

                for repo in repositories:
                    if query_count >= repos_per_query:
                        break

                    try:
                        # Skip if repo name/description suggests it's a paper implementation
                        if repo.description and any(kw.lower() in repo.description.lower() for kw in implementation_keywords):
                            print(f"  Skipping implementation repo: {repo.name}")
                            skip_count += 1
                            continue

                        # Prioritize repos that look like libraries
                        is_library = repo.description and any(kw.lower() in repo.description.lower() for kw in library_keywords)
                        if not is_library:
                            skip_count += 1
                            continue

                        print(f"Processing {query} repo {query_count+1}: {repo.name}")

                        # Skip if we already have this repo
                        if any(r['name'] == repo.name for r in repos_data):
                            print(f"  Skipping duplicate: {repo.name}")
                            duplicate_count += 1
                            if duplicate_count >= 5:  # If 5 consecutive duplicates
                                print("Too many consecutive duplicates, moving to next query.")
                                query_count = repos_per_query  # Force move to next query
                                break
                            continue
                        duplicate_count = 0  # Reset if not duplicate

                        try:
                            readme = repo.get_readme().decoded_content.decode('utf-8')
                        except:
                            readme = ""

                        # Additional checks for quality
                        if len(readme) < 500:  # Skip repos with very short READMEs
                            print(f"  Skipping {repo.name} - insufficient documentation")
                            skip_count += 1
                            continue

                        has_setup = False
                        try:
                            has_setup = repo.get_contents("setup.py", ref="master")
                        except:
                            try:
                                has_setup = repo.get_contents("setup.py", ref="main")
                            except:
                                print(f"  Skipping {repo.name} - no setup.py found")
                                no_setup_count += 1
                                if no_setup_count >= 5:  # If 5 consecutive no setup.py
                                    print("Too many repositories without setup.py, moving to next query.")
                                    query_count = repos_per_query  # Force move to next query
                                    break
                                continue
                        no_setup_count = 0  # Reset if setup.py found

                        repo_data = {
                            'name': repo.name,
                            'description': repo.description or "",
                            'stars': repo.stargazers_count,
                            'url': repo.html_url,
                            'readme_content': readme,
                            'search_query': query,
                            'language': repo.language,
                            'topics': repo.get_topics(),
                            'has_setup_py': bool(has_setup),
                            'is_library': is_library
                        }

                        repos_data.append(repo_data)
                        query_count += 1
                        skip_count = 0  # Reset skip count on successful addition

                        print(f"  Successfully processed: {repo.name} (Stars: {repo.stargazers_count})")

                        # Save progress
                        with open(dataset_dir / "dataset.json", 'w') as f:
                            json.dump(repos_data, f, indent=2)

                        # Update CSV summary
                        summary_data = [{
                            'name': r['name'],
                            'stars': r['stars'],
                            'language': r['language'],
                            'has_readme': bool(r['readme_content']),
                            'search_query': r['search_query'],
                            'url': r['url'],
                            'is_library': r['is_library']
                        } for r in repos_data]

                        pd.DataFrame(summary_data).to_csv(
                            dataset_dir / "summary.csv",
                            index=False
                        )

                        time.sleep(2)  # Rate limiting

                    except Exception as e:
                        print(f"  Error processing repository: {str(e)}")
                        continue

                # If too many consecutive skips, move to next query
                if skip_count >= 10:
                    print("Too many consecutive skips, moving to next query.")
                    break

            except Exception as e:
                print(f"Error with search query '{query}': {str(e)}")
                break

    print(f"Total repositories collected: {len(repos_data)}")
    return repos_data

if __name__ == "__main__":
    repos = collect_ml_repos(repos_per_query=38) 
    print(f"Total repositories collected: {len(repos)}")