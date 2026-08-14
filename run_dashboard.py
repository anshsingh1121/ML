import os
import sys

def main():
    """Launch the Streamlit dashboard."""
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    os.system("streamlit run src/dashboard/app.py")

if __name__ == "__main__":
    main()

