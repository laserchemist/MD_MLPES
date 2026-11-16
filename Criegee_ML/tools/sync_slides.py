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
    '5843035d4607fe876647e61decb8cf6e': 'ozonide.png',
    '59d9bbae42b366b7baba34043d3a73ad': 'ch2oo.png',
    'e11deb102e316f94fed8fb4cb2ef7c29': 'ozonolysis_energetics.png',
    '01ab75c67f4f3d7e3ddee20f3ffb68fe': 'acs2025_protocol.png',
    '5bdebfc7ac4d0321161cc75ec7ce30e2': 'levels3d.png',
    '3eb4f35f4388a8ec8c26267aeae5b95a': 'rot_doublets.png',
    '07c5724d6c5c1355b2362c6e123257f8': 'contour3d.png',
    '92ca06a62db9a2a0989728f9b4032c5c': 'nevpt2_curvature.png',
    'cd5b162e39a61798945428fe0d8f5b84': 'method_cut.png',
    'd3c2040a37daebfcd9a040145a3ab150': 'ir4d_zoom.png',
    '13b2e0da6fde37f0e7d9e7c5d11ec59b': 'ir4d_contour.png',
    '0c2a07063c18bec1a5183325c3e73625': 'delta_idea.png',
    'b046159feb6d44d74d9d05d03fa4f09c': 'switch_penalty.png',
    '38e497b488fb134b6622afa87e910e1e': 'gp_vs_poly.png',
    'b45af018b63775ae62965153a399aac7': 'vci_spectra.png',
    '87c7b3b8cd469e8149feb13b0d11f6a2': 'vci_bands.png',
    '07709e3211524d0d7bf8196504be5143': 'vci_isotopes.png',
    '25d3b46db80362cce2a36134ced55b53': 'vci_dynamics.png',
    'eb0b9854633ba09d5a8ef48b6a46f1dd': 'trpmd_spectra.png',
    'ad4e3e21e34b5c2adced25636392cec2': 'well9d_extend.png',
    '82382c426a9b3727a44217abbc1d32d9': 'well9d_fit.png',
    '34dec07f17807b6304c154343d33fff8': 'ch2oo_diabats.png',
    '044f7974867e56589dbd7d4f08aeed9a': 'ch2oo_trajectory.webp',
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
        html = re.sub(r'<a href="#slide-(\d+)"', r'<a data-internal href="#/\1"', html)   # links to other slides
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
