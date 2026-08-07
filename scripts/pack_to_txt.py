import pathlib
import base64
import json

def generate_unpack_file():
    root = pathlib.Path(__file__).resolve().parent.parent
    files_to_pack = {}
    
    # Exclude directories that are runtime/generated or environment specific
    exclude_dirs = {
        '.git', '.venv', '__pycache__', '.pytest_cache', '.mypy_cache', 
        '.ruff_cache', 'data', 'models', 'indexes', 'reports', 'logs', 
        'datasets', 'dist', 'build', 'htmlcov', '.gemini', 'scratch'
    }
    
    # Exclude file extensions that are binary/large/runtime
    exclude_exts = {
        '.pyc', '.pyo', '.zip', '.z_', '.pkl', '.index', '.npy', '.log', '.db', '.sqlite3'
    }
    
    exclude_names = {
        'unpack_project.txt', 'unpack_project.py', 'pack_to_txt.py'
    }
    
    for p in sorted(root.rglob('*')):
        if not p.is_file():
            continue
        if any(part in exclude_dirs for part in p.parts):
            continue
        if p.suffix.lower() in exclude_exts:
            continue
        if p.name in exclude_names:
            continue
            
        rel = p.relative_to(root).as_posix()
        try:
            files_to_pack[rel] = base64.b64encode(p.read_bytes()).decode('ascii')
        except Exception as e:
            print(f"[WARNING] Skipping {rel} due to read error: {e}")
            
    header = '''import json, base64, pathlib
print('===================================================================')
print('First Citizens Bank -- Enterprise Incident Intelligence Unpacker')
print('===================================================================')
payload_text = pathlib.Path(__file__).read_text(encoding='utf-8')
json_str = payload_text.split('### PAYLOAD ###\\n')[1]
payload = json.loads(json_str)
for rel_path, b64_data in payload.items():
    p = pathlib.Path(rel_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(base64.b64decode(b64_data.encode('ascii')))
print(f'[SUCCESS] Unpacked {len(payload)} enterprise project files successfully!')
print('Next steps:\\n  1. Double-click setup.bat\\n  2. Double-click run.bat')

### PAYLOAD ###
'''
    out_path = root / 'unpack_project.txt'
    out_path.write_text(header + json.dumps(files_to_pack, indent=2), encoding='utf-8')
    size_kb = out_path.stat().st_size / 1024
    size_mb = size_kb / 1024
    print(f"[SUCCESS] Packed {len(files_to_pack)} project files into {out_path.name}")
    print(f"[INFO] File path: {out_path.resolve()}")
    print(f"[INFO] File size: {size_kb:.2f} KB ({size_mb:.2f} MB)")

if __name__ == '__main__':
    generate_unpack_file()
