"""Drop hard-case records whose sentence MC Speech already has (the Common Voice question and
exclamation pool overlaps the MC Speech prompts). A real recording beats a clone, and a sentence
in two splits trips annotate's leakage check. Rewrites $HARD/manifest-*.jsonl in place, keeping
the first version as manifest-*.jsonl.orig; the dropped ids go to $HARD/dedupe-mc.json.

    python3 scripts/v2/dedupe_hard.py ~/sg-data/v2/mc ~/sg-data/v2/hard
"""

import json
import shutil
import sys
from pathlib import Path

mc_root, hard_root = map(Path, sys.argv[1:3])
mc = {json.loads(line)["text"].strip() for p in mc_root.glob("manifest-*.jsonl") for line in p.open(encoding="utf-8")}
dropped = {}
for path in sorted(hard_root.glob("manifest-*.jsonl")):
    original = path.with_suffix(".jsonl.orig")
    if not original.exists():
        shutil.copy(path, original)
    rows = [json.loads(line) for line in original.open(encoding="utf-8")]
    keep = [r for r in rows if r["text"].strip() not in mc]
    dropped[path.stem.removeprefix("manifest-")] = sorted(r["id"] for r in rows if r["text"].strip() in mc)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep), encoding="utf-8")
(hard_root / "dedupe-mc.json").write_text(json.dumps(dropped, indent=1))
print(json.dumps({"dropped_already_in_mc": {k: len(v) for k, v in dropped.items()}}))
