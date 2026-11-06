"""Verify release hashes, parse exported sources/JSON and enforce asset exclusions."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    count = 0
    for manifest in ['SOURCE_MANIFEST.json', 'RESEARCH_MANIFEST.json']:
        metadata = json.loads((ROOT/manifest).read_text())
        files = metadata if isinstance(metadata, list) else metadata['files']
        for record in files:
            path = ROOT/record['export']
            raw = path.read_bytes()
            assert hashlib.sha256(raw).hexdigest() == record['export_sha256'], path
            if path.suffix == '.json':
                json.loads(raw)
            if path.suffix == '.py':
                compile(raw, str(path), 'exec')
            count += 1
    denied = {'.key', '.pem', '.pt', '.pth', '.ckpt', '.npy', '.npz'}
    credential_patterns = [r'gh[pousr]_[A-Za-z0-9]{30,}', r'github_pat_[A-Za-z0-9_]{40,}',
                           r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',
                           r'AKIA[0-9A-Z]{16}', r'sk-[A-Za-z0-9_-]{35,}']
    for path in (ROOT/'research').rglob('*'):
        if not path.is_file() or '__pycache__' in path.parts:
            continue
        assert path.suffix not in denied and not path.name.startswith('.env'), path
        text = path.read_text()
        assert '/mnt/bum/jjenchen' not in text and '/home/nagio' not in text, path
        for pattern in credential_patterns:
            assert not re.search(pattern, text), 'Potential credential: ' + str(path)
    print(f'{count} manifest file hashes verified; research sources/JSON parse; asset exclusions pass.')


if __name__ == '__main__':
    main()
