import sys
import subprocess

if __name__ == '__main__':
    print('Launching Enterprise AI Incident Intelligence Dashboard...')
    # Use subprocess with a list to automatically handle spaces in file paths safely
    subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'src/dashboard/app.py'])