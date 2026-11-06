"""Export project-owned research code and compact evidence, never private assets.

Usage: python scripts/export_research_snapshot.py /path/to/phyre-unc
This updates only research/ and RESEARCH_MANIFEST.json, not the reusable package.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CODE = {'.py', '.sh', '.sbatch', '.yaml', '.yml'}
DENIED_PARTS = {'__pycache__', '.git', '.venv', 'logs', 'observations', 'data',
                'pools', 'cache', 'checkpoints', 'models', 'runs', 'agent',
                'literature-survey'}
SECRET_PATTERNS = [r'gh[pousr]_[A-Za-z0-9]{30,}', r'github_pat_[A-Za-z0-9_]{40,}',
                   r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',
                   r'AKIA[0-9A-Z]{16}', r'sk-[A-Za-z0-9_-]{35,}']


def sha(data):
    return hashlib.sha256(data).hexdigest()


def sanitize(text):
    text = text.replace('下一次工作先讀 [STATE.md](../STATE.md)',
                        '公開程式與資產需求請見 [research README](../README.md)')
    text = text.replace('/mnt/bum/jjenchen/phyre-unc', '/path/to/phyre-unc')
    text = text.replace('/home/nagio/gptSpace/phyre-unc', '/path/to/phyre-unc')
    text = text.replace('/mnt/bum/jjenchen', '/path/to/research-assets')
    text = text.replace('/home/nagio/gptSpace', '/path/to/workspace')
    text = text.replace('/home/nagio', '/path/to/user-home')
    # Other cluster roots are configuration placeholders in this source archive.
    text = re.sub(r'/home/[A-Za-z0-9_.-]+', '/path/to/user-home', text)
    text = re.sub(r'/nas/[^\s\x22\x27`<>]+', '/path/to/external-assets', text)
    text = re.sub(r'\b(?:10|192\.168)\.(?:\d{1,3}\.){1,2}\d{1,3}\b', '127.0.0.1', text)
    return text


def export(source):
    source = source.resolve()
    selected, omitted = [], []
    for area in ['pipeline', 'results', 'docs']:
        for path in sorted((source / area).rglob('*')):
            if not path.is_file() or path.is_symlink():
                continue
            rel = path.relative_to(source)
            if any(part in DENIED_PARTS for part in rel.parts[1:-1]):
                continue
            include = path.suffix in CODE
            if area == 'docs':
                include = path.suffix == '.md' and not any(
                    word in path.name for word in ['SLIDE', 'PRESENTATION', 'DECK', 'MANUSCRIPT', 'NOTES_EXTRACTED'])
            if area == 'results':
                include |= path.suffix == '.csv'
                include |= path.suffix == '.json' and path.stem.isupper() and len(rel.parts) <= 4
                include |= path.name == 'probe_cross_training_scope_20261020.json'
                include |= path.suffix == '.md' and len(rel.parts) <= 3
            if not include:
                continue
            if path.suffix == '.json' and path.stat().st_size > 5_000_000:
                omitted.append({'source': str(rel), 'bytes': path.stat().st_size,
                                'reason': 'Large historical detail; retained in private research archive'})
                continue
            selected.append(path)
    records = []
    for path in selected:
        raw = path.read_bytes()
        text = raw.decode('utf-8')
        for pattern in SECRET_PATTERNS:
            if re.search(pattern, text):
                raise ValueError('Potential credential in selected file: ' + str(path.relative_to(source)))
        sanitized = sanitize(text)
        normalized = sanitized.replace('\r\n', '\n')
        if path.suffix == '.md':
            normalized = normalized.rstrip() + '\n'
        exported = normalized.encode('utf-8')
        relative = path.relative_to(source)
        dest = REPO / 'research' / relative
        syntax_error = None
        if path.suffix == '.py':
            try:
                compile(exported, str(relative), 'exec')
            except SyntaxError as exc:
                # Preserve failed historical attempts as text, never silently repair them.
                compile_original_failed = False
                try:
                    compile(raw, str(relative), 'exec')
                except SyntaxError:
                    compile_original_failed = True
                if not compile_original_failed:
                    raise
                syntax_error = {'line': exc.lineno, 'message': exc.msg}
                dest = dest.with_suffix('.py.txt')
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(exported)
        if path.suffix == '.json':
            json.loads(exported)
        records.append({'source': str(relative), 'export': str(dest.relative_to(REPO)),
                        'source_sha256': sha(raw), 'export_sha256': sha(exported),
                        'bytes': len(exported), 'path_redacted': sanitized != text,
                        'newline_normalized': normalized != sanitized,
                        'historical_syntax_error': syntax_error})
    manifest = {'release_date': '2026-11-06', 'latest_experiment_date': '2026-10-21',
                'source_directory_name': source.name,
                'scope': 'Project-owned pipeline and archived experiment code; compact plans, results, audits, CSVs and research reports.',
                'transformation': 'Machine-specific filesystem roots replaced with placeholders; private IP addresses replaced with loopback. CRLF becomes LF; Markdown extra EOF blank lines removed. The progress report internal STATE link redirects to the public research README. Numerical result values unchanged. Original and public byte hashes are distinct.',
                'excluded': ['credentials', 'weights', 'latent and simulation arrays', 'per-action raw scores',
                             'logs', 'third-party simulator and encoder sources', 'slides', 'agent state'],
                'execution_limit': 'Historical drivers require external assets and original hash manifests; exported redacted manifests are evidence, not executable launch registrations.',
                'files': records, 'large_historical_json_not_exported': omitted}
    (REPO / 'RESEARCH_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'files': len(records), 'bytes': sum(r['bytes'] for r in records),
                      'large_historical_json_omitted': len(omitted)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    export(parser.parse_args().source)
