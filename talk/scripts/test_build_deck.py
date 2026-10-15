"""Unit tests for build_deck.py (run: python3 -m pytest talk/scripts -v)."""

import re

import pytest

import build_deck as bd

SVG = """<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "x.dtd">
<!-- Created with matplotlib -->
<svg xmlns="http://www.w3.org/2000/svg" width="720pt" height="300pt" viewBox="0 0 720 300">
 <defs><clipPath id="p1"><rect width="10" height="10"/></clipPath></defs>
 <rect width="5" height="5" clip-path="url(#p1)"/>
 <use href="#p1"/>
</svg>"""


def test_clean_svg_strips_prolog_and_root_size():
    out = bd.clean_svg(SVG, "f9")
    assert out.startswith("<svg")
    assert "<?xml" not in out and "DOCTYPE" not in out and "<!--" not in out
    root = out[: out.find(">")]
    assert "width=" not in root and "height=" not in root
    assert 'viewBox="0 0 720 300"' in root
    # child elements keep their own sizes
    assert '<rect width="5" height="5"' in out


def test_clean_svg_prefixes_ids_and_references():
    out = bd.clean_svg(SVG, "f9")
    assert 'id="f9-p1"' in out
    assert "url(#f9-p1)" in out and 'href="#f9-p1"' in out
    assert 'id="p1"' not in out


def test_clean_svg_rejects_non_svg():
    with pytest.raises(ValueError):
        bd.clean_svg("<html></html>", "x")


def test_build_inlines_markers(tmp_path):
    figs = tmp_path / "figs"
    figs.mkdir()
    (figs / "a.svg").write_text(SVG)
    src = tmp_path / "src.html"
    src.write_text("<!-- SOURCE FILE. x -->\n<section><!--FIG:a--></section>"
                   "<section><!--FIG:a--></section>")
    out = tmp_path / "index.html"
    used = bd.build(src, figs, out)
    html = out.read_text()
    assert used == ["a", "a"]
    assert "<!--FIG:" not in html
    assert html.count("<svg") == 2
    assert "GENERATED" in html
    # the two copies get distinct id prefixes
    assert len(set(re.findall(r'id="(f\d+)-p1"', html))) == 2


def test_build_missing_figure_names_it(tmp_path):
    src = tmp_path / "src.html"
    src.write_text("<!--FIG:nothere-->")
    with pytest.raises(FileNotFoundError, match="nothere"):
        bd.build(src, tmp_path, tmp_path / "o.html")


DECK = bd.TALK_DIR / "index.html"


@pytest.mark.skipif(not DECK.exists(), reason="deck not built")
def test_deck_local_links_resolve():
    """Every relative href/src in the built deck points to an existing file."""
    html = DECK.read_text(encoding="utf-8")
    refs = re.findall(r'(?:href|src)="([^"#:]+)"', html)
    local = [r for r in refs if not r.startswith(("http", "//", "data:"))]
    assert local, "expected some local links (poster, research log)"
    missing = [r for r in local if not (bd.TALK_DIR / r).exists()]
    assert missing == []
