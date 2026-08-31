#!/usr/bin/env python3
"""
build_standalone.py — impacchetta l'atlante in un singolo file HTML.

In development `index.html` loads Three.js from ./vendor/ and the data from
./data_coh/ over fetch: right for a web server (GitHub Pages, `python -m
http.server`), useless when the file is opened by double click or pasted
somewhere else.

This script writes `dist/index.html` with everything inside, library and JSON, so
the file works anywhere including `file://`, without a single network request
apart from the fonts.

    python build_standalone.py

How the Three.js inlining works: the bundle is an ES module that ends with
`export{a as Scene, b as Vector3, ...}`. That export is rewritten into
`globalThis.THREE = {Scene:a, Vector3:b, ...}`, and in the app the line
`import * as THREE from 'three'` becomes `const THREE = globalThis.THREE`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "dist" / "index.html"

DATASETS = [("plates-data", HERE / "data_coh" / "atlas_plates.json")]


def three_as_global(src: str) -> str:
    """Trasforma il modulo ES di Three.js in un assegnamento a globalThis."""
    m = re.search(r"export\{(.+?)\};?\s*$", src, re.S)
    if not m:
        raise SystemExit("non trovo l'export finale in three.module.min.js")
    pairs = []
    for item in m.group(1).split(","):
        item = item.strip()
        if " as " in item:
            local, exported = (s.strip() for s in item.split(" as "))
        else:
            local = exported = item
        pairs.append(f"{exported}:{local}")
    return src[:m.start()] + ";globalThis.THREE={" + ",".join(pairs) + "};"


def main() -> None:
    html = (HERE / "index.html").read_text(encoding="utf-8")
    three = three_as_global((HERE / "vendor" / "three.module.min.js")
                            .read_text(encoding="utf-8"))

    # 1. importmap -> modulo Three.js inline
    #    (replacement passato come lambda: il bundle è pieno di backslash che
    #     re.sub interpreterebbe come escape del template)
    inlined = ('<script type="module">\n'
               '/* three.js r169 — https://threejs.org (MIT) */\n'
               + three.replace("</script>", "<\\/script>") + "\n</script>")
    html, n = re.subn(r'<script type="importmap">.*?</script>',
                      lambda _m: inlined, html, flags=re.S)
    if n != 1:
        raise SystemExit(f"importmap trovata {n} volte, attesa 1")

    # 2. import bare -> riferimento al globale
    html = html.replace("import * as THREE from 'three';",
                        "const THREE = globalThis.THREE;")

    # 3. dati incorporati, letti dal fallback di loadAtlas()
    blobs = []
    for el_id, path in DATASETS:
        if not path.exists():
            print(f"  [!] manca {path}, lo salto")
            continue
        payload = json.dumps(json.loads(path.read_text(encoding="utf-8")),
                             separators=(",", ":"))
        blobs.append(f'<script type="application/json" id="{el_id}">'
                     + payload.replace("</", "<\\/") + "</script>")
        print(f"  incorporato {path.relative_to(HERE)} "
              f"({len(payload)/1e3:.0f} KB) come #{el_id}")
    anchor = re.search(r"<canvas\b", html)
    if not anchor:
        raise SystemExit("non trovo il canvas su cui ancorare i dati")
    html = html[:anchor.start()] + "\n".join(blobs) + "\n\n" + html[anchor.start():]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print(f"  -> {OUT.relative_to(HERE)}  ({OUT.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
