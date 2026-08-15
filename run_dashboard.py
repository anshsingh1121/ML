import os
import sys

if __name__ == '__main__':
    print('Launching Enterprise AI Incident Intelligence Dashboard...')
    # Use sys.executable to bypass Windows PATH environment issues
    os.system(f'{sys.executable} -m streamlit run src/dashboard/app.py')