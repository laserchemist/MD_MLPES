"""Convert the Slides deck (deck.json + slides/*.html) into src/slides.json for the React viewer.

Usage:
    python tools/sync_slides.py path/to/deck/project

Each slide keeps its inline-styled <section> markup; speaker notes (<aside>) are split out,
uploaded-image URLs (/_blob/<id>) are mapped to files in public/figures, and links open in a new tab.
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent

# asset id in the hosted deck -> local figure file in public/figures
BLOBS = {
    '62cf44353e5b0dc5314ae963ab746bda': 'zone.png',
    'cf94d4ee9f49586cbf21314cf0d729cb': 'diabatic_adiabatic.png',
    'c9ace7ab163f67f3f3f350fee600260e': 'topology.png',
    '893e035e29f70415ac37fb3254a18122': 'h3_learning.png',
    '64fc476838776c1f29a8bd40356cf126': 'h2o_dim.png',
    'c48c3e97de0a31c4449d48f3bacf7834': 'ch2oo_character.png',
    '981bba02ff56ab9fad941596fe31d26f': 'ch2oo_extrapolation.png',
    '6b3333f7645894e6fb5855d130936247': 'ch2oo_diabats.png',
    '836a78ba78ba5e516b2b248222786e81': 'frequency_validation.png',
    '2976bb5d970c830034c018f909dda549': 'expt_compare.png',
}


def strip_tags(s):
    return re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', '', s)).strip()


def main(project):
    project = Path(project)
    deck = json.loads((project / 'deck.json').read_text())
    starts = {v['start']: v['description'] for v in deck.get('sections', {}).values()}
    slides, section = [], ''
    for sid in deck['order']:
        html = (project / 'slides' / f'{sid}.html').read_text()
        notes = ''
        m = re.search(r'<aside>(.*?)</aside>', html, re.S)
        if m:
            notes = strip_tags(m.group(1))
            html = html[:m.start()] + html[m.end():]
        missing = [b for b in re.findall(r'/_blob/([0-9a-f]{32})', html) if b not in BLOBS]
        if missing:
            sys.exit(f'{sid}: unknown image ids {missing}; add them to BLOBS')
        html = re.sub(r'/_blob/([0-9a-f]{32})', lambda mm: 'figures/' + BLOBS[mm.group(1)], html)
        html = html.replace('<a href=', '<a target="_blank" rel="noopener" href=')
        t = re.search(r'<h[12][^>]*>(.*?)</h[12]>', html, re.S)
        if sid in starts:
            section = starts[sid]
        slides.append({'id': sid, 'title': strip_tags(t.group(1)) if t else sid,
                       'section': section, 'html': html.strip(), 'notes': notes})
    out = HERE / 'src' / 'slides.json'
    out.write_text(json.dumps({'title': deck.get('title', ''), 'slides': slides}, ensure_ascii=False, indent=1))
    print(f'wrote {len(slides)} slides to {out}')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '.')
