"""Read-only ImageNet access; all candidates, stable lexicographic filename IDs."""
import hashlib
import io
import json
import os
import sqlite3
from common import *


def prepare_manifest():
    ASSETS.mkdir(parents=True, exist_ok=True)
    db = ASSETS / 'candidates.sqlite'
    if (HERE / 'corpus.json').exists():
        d = json.loads((HERE / 'corpus.json').read_text())
        assert sha(db) == d['manifest_sha256']
        return
    assert not db.exists(), 'Preserve partial manifest; inspect explicitly'
    roots = [Path('/data1/matengyu/data/ImageNet/ImageNet/train'),
             Path('/data1/matengyu/data/ImageNet/ImageNet/train_gray/unpacked_gray')]
    records = []
    counts = []
    for root in roots:
        found = sorted(p for p in root.rglob('*') if p.is_file())
        counts.append(len(found))
        records.extend((p.name, str(p), p.stat().st_size) for p in found)
    assert counts == [1261125, 20042] and len(records) == 1281167
    records.sort()
    assert len(set(r[0] for r in records)) == len(records)
    con = sqlite3.connect(db)
    con.execute('CREATE TABLE candidates (idx INTEGER PRIMARY KEY, id TEXT UNIQUE NOT NULL, path TEXT NOT NULL, size INTEGER NOT NULL)')
    con.executemany('INSERT INTO candidates VALUES (?,?,?,?)', ((i, *r) for i, r in enumerate(records)))
    con.commit(); con.close()
    write_json(HERE / 'corpus.json', dict(n_candidates=len(records), root_counts=counts,
        roots=[str(p) for p in roots], manifest=str(db), manifest_sha256=sha(db),
        total_bytes=sum(r[2] for r in records), ordering='lexicographic unique original filename; idx zero-based',
        source_sha_and_decode_audit='recorded during scan for every candidate', **limitations()))


class Candidates:
    def __init__(self, indices):
        self.indices = indices
        self.con = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        import numpy as np
        import torch
        from PIL import Image
        from torchvision import transforms
        if self.con is None:
            self.con = sqlite3.connect(f'file:{ASSETS / "candidates.sqlite"}?mode=ro', uri=True)
        idx = int(self.indices[position])
        row = self.con.execute('SELECT id,path,size FROM candidates WHERE idx=?', (idx,)).fetchone()
        assert row is not None
        candidate_id, path, expected_size = row
        raw = Path(path).read_bytes()
        if len(raw) != expected_size:
            raise ValueError(f'Candidate size changed: {candidate_id}')
        digest = hashlib.sha256(raw).hexdigest()
        with Image.open(io.BytesIO(raw)) as im:
            mode, size = im.mode, im.size
            image = im.convert('RGB')
            tensor = transforms.Compose([transforms.ToTensor(), transforms.Resize(425), transforms.CenterCrop(425)])(image)
        assert tensor.shape == (3, 425, 425) and torch.isfinite(tensor).all()
        normalized = transforms.Normalize([.485, .456, .406], [.229, .224, .225])(tensor)
        padded = torch.zeros((3, 434, 434), dtype=torch.float32)
        padded[:, :425, :425] = normalized
        audit = json.dumps(dict(idx=idx, candidate_id=candidate_id, source_sha256=digest,
                                mode=mode, width=size[0], height=size[1], bytes=expected_size))
        return padded, idx, audit


if __name__ == '__main__':
    prepare_manifest()
