import os
import json

def extract():
    with open('corporate_transfer_bundle.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for filepath, content in data.items():
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f'Created {filepath}')
        
    # Create empty required directories
    for d in ['data/raw', 'data/processed', 'models', 'reports', 'indexes', 'logs']:
        os.makedirs(d, exist_ok=True)
    
    print('Project successfully reconstructed! You can now place your CSV in data/raw/incidents.csv')

if __name__ == '__main__':
    extract()
