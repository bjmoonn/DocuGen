"""
trains a model to evaluate readme quality using our curated dataset
"""

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import json
from pathlib import Path
import numpy as np

class ReadmeDataset(Dataset):
    def __init__(self, texts, scores, tokenizer, max_length=512):
        """
        dataset for readme content and quality scores
        
        args:
            texts (list): readme contents
            scores (list): normalized quality scores
            tokenizer: huggingface tokenizer
            max_length (int): maximum sequence length
        """
        self.texts = texts
        self.scores = scores
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        score = self.scores[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'score': torch.tensor(score, dtype=torch.float)
        }

class ReadmeQualityModel(nn.Module):
    def __init__(self, model_name='microsoft/codebert-base'):
        """
        model for predicting readme quality scores
        """
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

def train_readme_model(dataset_path: str = 'data/dataset.json', epochs: int = 5):
    """
    train the readme quality model using our dataset
    
    args:
        dataset_path (str): path to dataset.json
        epochs (int): number of training epochs
    """
    # load and prepare data
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    texts = [repo['readme_content'] for repo in data]
    # normalize stars to 0-1 range for scoring
    stars = [repo['stars'] for repo in data]
    scores = (np.array(stars) - min(stars)) / (max(stars) - min(stars))
    
    # initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
    
    # create dataset and dataloader
    dataset = ReadmeDataset(texts, scores, tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # initialize model and training
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = ReadmeQualityModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.MSELoss()
    
    # training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            scores = batch['score'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.squeeze(), scores)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"epoch {epoch+1}, loss: {total_loss/len(dataloader):.4f}")
    
    # save model and tokenizer
    save_path = Path('src/models/weights')
    save_path.mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), save_path / 'readme_quality_model.pt')
    tokenizer.save_pretrained(save_path / 'tokenizer')

if __name__ == "__main__":
    train_readme_model()