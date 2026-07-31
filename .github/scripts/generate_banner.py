#!/usr/bin/env python3
"""Regenerate dark.svg and light.svg.

Install: pip install pillow numpy scipy
Run placeholder: python .github/scripts/generate_banner.py
Run with a real portrait: python .github/scripts/generate_banner.py --photo portrait.jpg

This file is generated from the project package's build logic. For a photo-specific
version, provide a sharp head-and-shoulders image with a flat background.
"""
from pathlib import Path
import argparse, runpy, shutil, sys

# The complete source is kept at the repository root for transparent regeneration.
root = Path(__file__).resolve().parents[2]
script = Path(__file__).resolve().parent / 'banner_generator_full.py'
if not script.exists():
    raise SystemExit('banner_generator_full.py is missing')
ns = runpy.run_path(str(script))
parser = argparse.ArgumentParser()
parser.add_argument('--photo', type=Path)
args = parser.parse_args()
for theme in ('dark','light'):
    (root/f'{theme}.svg').write_text(ns['build_banner'](theme, args.photo), encoding='utf-8')
print('wrote dark.svg and light.svg')
