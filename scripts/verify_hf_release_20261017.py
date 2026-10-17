"""Verify public release receipts without reading any Hugging Face credential."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import requests


def safe_error(kind, value, trace):
    # Redirect failures may contain signed download URLs; do not echo them.
    frames = traceback.extract_tb(trace)
    print(json.dumps(dict(event='public_verification_failed', error_type=kind.__name__,
                          frames=[dict(file=Path(f.filename).name, line=f.lineno) for f in frames[-4:]])), flush=True)


sys.excepthook = safe_error

HERE = Path(os.environ.get('NEUROADAPTER_HF_RELEASE_DIR', str(Path(__file__).resolve().parent))).resolve()
manifest = json.loads((HERE / 'upload-manifest.json').read_text())
receipt = json.loads((HERE / 'upload-result.json').read_text())
assert receipt['completed'] and receipt['all_public_and_sha_verified']
class AnonymousSession(requests.Session):
    """Keep the configured proxy while suppressing netrc/service credentials."""

    def __init__(self):
        super().__init__()
        self.auth = lambda request: request

    def rebuild_auth(self, prepared_request, response):
        prepared_request.headers.pop('Authorization', None)


session = AnonymousSession()
results = []


def get_json(url):
    response = session.get(url, timeout=90)
    response.raise_for_status()
    return response.json()


for package in manifest['packages']:
    repo = package['repo']
    saved = next(r for r in receipt['repositories'] if r['repo'] == repo)
    info = get_json(f'https://huggingface.co/api/models/{repo}')
    assert info['private'] is False and info['sha'] == saved['revision']
    revision = info['sha']
    tree = get_json(f'https://huggingface.co/api/models/{repo}/tree/{revision}?recursive=true&expand=false')
    entries = {r['path']: r for r in tree if r['type'] == 'file'}
    public_manifest = get_json(f'https://huggingface.co/{repo}/resolve/{revision}/manifest.json')
    assert public_manifest['files'] == [{k: v for k, v in f.items() if k != 'source'} for f in package['files']]
    assert '/data1/' not in json.dumps(public_manifest)
    allowed = {f['path'] for f in package['files']} | {
        '.gitattributes', 'README.md', 'SHA256SUMS', 'manifest.json',
        'inference-config.json', 'evaluation-summary.json',
    }
    assert set(entries).issubset(allowed)
    checks = []
    for file in package['files']:
        row = entries[file['path']]
        assert row['size'] == file['size']
        url = f"https://huggingface.co/{repo}/resolve/{revision}/{file['path']}"
        if row.get('lfs'):
            assert row['lfs']['oid'] == file['sha256']
            # Also verify anonymous payload access, not just model metadata.
            with session.get(url, headers={'Range': 'bytes=0-0'}, stream=True, timeout=90) as response:
                response.raise_for_status()
                assert response.raw.read(1)
        else:
            assert file['size'] < 8 * 1024**2
            response = session.get(url, timeout=90)
            response.raise_for_status()
            assert hashlib.sha256(response.content).hexdigest() == file['sha256']
        checks.append(dict(path=file['path'], size=file['size'], sha256=file['sha256'], anonymous_payload_access=True))
    results.append(dict(repo=repo, revision=revision, public=True, files=checks))
    print(json.dumps(dict(event='independent_public_verification', repo=repo, files=len(checks))), flush=True)

result = dict(completed=True, unix_time=time.time(), credential_used=False,
              all_public_and_sha_verified=True, repositories=results,
              verification='Pinned public API SHA256/size plus anonymous payload access; small non-LFS files fully hashed.')
(HERE / 'public-verification.json').write_text(json.dumps(result, indent=2) + '\n')
