from __future__ import annotations

import base64
import json
import math
import os
import random
import shutil
import textwrap
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import binary_closing, binary_fill_holes, label

OUT_ROOT = Path('/mnt/data/PDOX103-reference-style')
ZIP_PATH = Path('/mnt/data/PDOX103-reference-style-profile.zip')
REF_ROOT = Path('/mnt/data/arif_ref/arifhaxn-main')


# -------------------------- profile config --------------------------
PROFILE = {
    'name': 'Md. Fahmidul Karim Rafi',
    'username': 'PDOX103',
    'role': 'CSE Student + Software Developer',
    'origin': 'Bangladesh',
    'education': 'BSc in CSE · AUST',
    'status': 'Exploring + Building + Learning',
    'toolchain': 'VS Code, Git, Docker, Figma',
    'languages': 'C, C++, Java, JavaScript, PHP, Dart',
    'frontend': 'React, Tailwind, Bootstrap, Flutter',
    'backend': 'Node.js, Express, Laravel',
    'database': 'MongoDB, MySQL, Firebase',
    'infra': 'AWS, Docker, Git',
    'email': 'fahmidulkarimrafi2.0@gmail.com',
    'portfolio': 'coming soon',
    'linkedin': 'md-fahmidul-karim-rafi',
    'facebook': 'fahmidulkarim.paradox.103',
    'instagram': 'fahmidulkarim',
    'youtube': 'RaFi-cf8cn',
}

THEMES = {
    'dark': {
        'OUTER': '#070B16', 'BG': '#0A101F', 'PANEL': '#0C1426', 'BAR': '#0B1222',
        'CYAN': '#22D3EE', 'VIOLET': '#A78BFA', 'VIOLET2': '#7C3AED',
        'EMERALD': '#10B981', 'TEXT': '#F8FAFC', 'MUTED': '#94A3B8', 'DIM': '#475569',
        'PORTRAIT': '#A78BFA', 'PANEL_LINE': 'rgba(34,211,238,0.35)',
    },
    'light': {
        'OUTER': '#E2E8F0', 'BG': '#F8FAFC', 'PANEL': '#FFFFFF', 'BAR': '#F1F5F9',
        'CYAN': '#0891B2', 'VIOLET': '#7C3AED', 'VIOLET2': '#7C3AED',
        'EMERALD': '#059669', 'TEXT': '#0F172A', 'MUTED': '#475569', 'DIM': '#94A3B8',
        'PORTRAIT': '#7C3AED', 'PANEL_LINE': 'rgba(8,145,178,0.35)',
    }
}

# -------------------------- image helpers --------------------------

def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf' if bold else '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf' if bold else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    ]
    for p in candidates:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def synthetic_portrait(size=(300, 340)) -> Image.Image:
    """Create an abstract, non-identifying head-and-shoulders placeholder."""
    w, h = size
    im = Image.new('L', size, 0)
    d = ImageDraw.Draw(im)

    # shoulders / blazer with soft tonal structure
    d.ellipse((25, 225, 275, 390), fill=150)
    d.polygon([(42, 340), (65, 245), (119, 218), (181, 218), (237, 245), (260, 340)], fill=175)
    d.polygon([(95, 225), (150, 285), (205, 225), (177, 340), (123, 340)], fill=75)
    d.rectangle((130, 195, 170, 245), fill=145)

    # head and hair
    d.ellipse((86, 42, 214, 216), fill=178)
    d.ellipse((80, 26, 220, 128), fill=62)
    d.polygon([(83, 98), (88, 55), (110, 28), (150, 18), (195, 35), (218, 74), (214, 110),
               (198, 82), (177, 62), (148, 58), (122, 72), (105, 96)], fill=70)

    # face modelling
    d.ellipse((100, 88, 200, 205), fill=188)
    d.ellipse((95, 105, 112, 145), fill=118)
    d.ellipse((188, 105, 205, 145), fill=118)
    d.polygon([(150, 112), (143, 155), (154, 158)], fill=115)
    d.arc((116, 143, 184, 190), 20, 160, fill=85, width=4)
    d.line((119, 125, 141, 122), fill=65, width=4)
    d.line((159, 122, 181, 125), fill=65, width=4)
    d.ellipse((126, 125, 134, 132), fill=35)
    d.ellipse((166, 125, 174, 132), fill=35)
    # glasses-like accent, generic not identity specific
    d.rounded_rectangle((111, 115, 143, 139), radius=5, outline=80, width=3)
    d.rounded_rectangle((157, 115, 189, 139), radius=5, outline=80, width=3)
    d.line((143, 124, 157, 124), fill=80, width=3)

    # soft light from top-left and texture
    yy, xx = np.mgrid[0:h, 0:w]
    light = 30 * (1 - xx / w) + 20 * (1 - yy / h)
    arr = np.array(im, dtype=np.float32)
    arr = np.clip(arr + light * (arr > 0), 0, 255)
    rng = np.random.default_rng(103)
    arr += rng.normal(0, 4.0, arr.shape) * (arr > 0)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    im = Image.fromarray(arr, 'L').filter(ImageFilter.GaussianBlur(0.6))
    return im


def prepare_photo(path: Path | None, size=(300, 340), dark=False) -> tuple[Image.Image, np.ndarray]:
    if path and path.exists():
        src = Image.open(path).convert('RGB')
        src = ImageOps.fit(src, size, method=Image.Resampling.LANCZOS, centering=(0.5, 0.42))
        # Approximate background segmentation using border median colour.
        a = np.array(src).astype(np.float32)
        border = np.concatenate([a[:12].reshape(-1,3), a[-12:].reshape(-1,3), a[:, :12].reshape(-1,3), a[:, -12:].reshape(-1,3)])
        bg = np.median(border, axis=0)
        dist = np.linalg.norm(a - bg[None,None,:], axis=2)
        mask = dist > max(20.0, np.percentile(dist, 55))
        mask = binary_closing(mask, iterations=2)
        mask = binary_fill_holes(mask)
        lab, n = label(mask)
        if n:
            counts = np.bincount(lab.ravel()); counts[0] = 0
            mask = lab == counts.argmax()
        gray = ImageOps.grayscale(src)
    else:
        gray = synthetic_portrait(size)
        mask = np.array(gray) > 8

    gray = ImageOps.autocontrast(gray, cutoff=1)
    gray = ImageEnhance.Contrast(gray).enhance(1.3)
    gray = gray.filter(ImageFilter.UnsharpMask(radius=3, percent=140, threshold=2))
    return gray, mask


def floyd_steinberg_serpentine(gray: Image.Image, mask: np.ndarray, dark: bool) -> np.ndarray:
    arr = np.asarray(gray, dtype=np.float64) / 255.0
    # Dark mode draws illuminated subject; light mode draws darker tones and retains backdrop impression.
    signal = arr if dark else (1.0 - arr)
    out = np.zeros_like(signal, dtype=np.uint8)
    h, w = signal.shape
    work = signal.copy()
    for y in range(h):
        if y % 2 == 0:
            xs = range(w); direction = 1
        else:
            xs = range(w - 1, -1, -1); direction = -1
        for x in xs:
            if dark and not mask[y, x]:
                work[y, x] = 0
                continue
            old = work[y, x]
            new = 1.0 if old >= 0.5 else 0.0
            out[y, x] = int(new)
            err = old - new
            nx = x + direction
            if 0 <= nx < w:
                work[y, nx] += err * 7/16
            if y + 1 < h:
                if 0 <= x - direction < w:
                    work[y+1, x-direction] += err * 3/16
                work[y+1, x] += err * 5/16
                if 0 <= x + direction < w:
                    work[y+1, x+direction] += err * 1/16
    if dark:
        out[~mask] = 0  # hard-clear error diffusion bleed at mask edge
    return out.astype(bool)


def points_to_runs(points: np.ndarray) -> str:
    """Compress integer dot coordinates into SVG horizontal 1px path runs."""
    if len(points) == 0:
        return ''
    # points are x,y
    order = np.lexsort((points[:,0], points[:,1]))
    p = points[order]
    chunks = []
    i = 0
    while i < len(p):
        y = int(p[i,1]); x0 = int(p[i,0]); x1 = x0
        i += 1
        while i < len(p) and int(p[i,1]) == y and int(p[i,0]) == x1 + 1:
            x1 = int(p[i,0]); i += 1
        chunks.append(f'M{x0} {y}h{x1-x0+1}v1h-{x1-x0+1}z')
    return ''.join(chunks)


def sample_mask(mask: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    pts = np.column_stack([xs, ys])
    if len(pts) == 0:
        return np.zeros((n,2), dtype=float)
    idx = rng.choice(len(pts), size=n, replace=len(pts) < n)
    return pts[idx].astype(float)


def shape_masks(size=(300,340)) -> list[np.ndarray]:
    w,h = size
    # Code glyph
    im1 = Image.new('L', size, 0); d1 = ImageDraw.Draw(im1)
    f = font(105, bold=True)
    txt = '</>'
    bb = d1.textbbox((0,0), txt, font=f)
    d1.text(((w-(bb[2]-bb[0]))/2, 115), txt, font=f, fill=255)
    im1 = im1.filter(ImageFilter.GaussianBlur(0.4))

    # Brain symbol: two lobes and internal folds, drawn as strokes
    im2 = Image.new('L', size, 0); d2 = ImageDraw.Draw(im2)
    c1=(128,175); c2=(172,175)
    for cx in (128,172):
        d2.ellipse((cx-45, 105, cx+25, 245), outline=255, width=8)
    d2.line((150,110,150,240), fill=255, width=7)
    folds=[[(103,140),(125,128),(139,145),(121,164),(101,157)],
           [(101,185),(124,173),(140,194),(120,211),(103,205)],
           [(197,140),(175,128),(161,145),(179,164),(199,157)],
           [(199,185),(176,173),(160,194),(180,211),(197,205)]]
    for pts in folds:
        d2.line(pts, fill=255, width=6, joint='curve')
    im2 = im2.filter(ImageFilter.GaussianBlur(0.5))

    # Triangle / cloud-engineering mark
    im3 = Image.new('L', size, 0); d3 = ImageDraw.Draw(im3)
    tri=[(150,92),(72,244),(228,244)]
    d3.polygon(tri, fill=255)
    d3.polygon([(150,125),(102,225),(198,225)], fill=0)
    d3.line((150,92,72,244,228,244,150,92), fill=255, width=8, joint='curve')

    return [np.asarray(x) > 80 for x in (im1,im2,im3)]


def leader_geometry(label: str, value: str, x_label=482, x_right=1140, fs=14):
    label_w = min(160, max(35, len(label) * fs * 0.58))
    value_w = min(380, max(45, len(value) * fs * 0.57))
    x1 = x_label + label_w + 12
    x2 = x_right - value_w - 12
    if x2 < x1 + 20:
        x2 = x1 + 20
    return x1, x2, value_w


def esc(s: str) -> str:
    return (s.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;').replace('"','&quot;'))


def build_banner(theme_name: str, photo: Path | None = None) -> str:
    t = THEMES[theme_name]
    dark = theme_name == 'dark'
    rng = np.random.default_rng(103)
    gray, mask = prepare_photo(photo, dark=dark)
    dots = floyd_steinberg_serpentine(gray, mask, dark=dark)

    # Keep density close to the reference target (~17k). If too dense/sparse, deterministic resample.
    ys, xs = np.nonzero(dots)
    pts = np.column_stack([xs, ys])
    target = 17000
    if len(pts) > target:
        idx = rng.choice(len(pts), target, replace=False); pts = pts[idx]
    elif len(pts) < target:
        extra = sample_mask(mask if dark else np.ones_like(mask, bool), target-len(pts), rng)
        pts = np.vstack([pts, extra.astype(int)])
    pts = pts.astype(int)

    # 60 interleaved random intro groups, scattered across the portrait.
    intro_assign = rng.integers(0, 60, len(pts))
    intro_groups = [pts[intro_assign == i] for i in range(60)]

    # 94 organic drift bands: linear projection + per-dot noise sigma 4 before grouping.
    proj = pts[:,0] * 0.71 + pts[:,1] * 0.37 + rng.normal(0,4.0,len(pts))
    order = np.argsort(proj)
    drift_groups = np.array_split(pts[order], 94)

    masks = shape_masks()
    ntrav = 900
    shape_pts = [sample_mask(m, ntrav, rng) for m in masks]
    # Optimal transport matching between consecutive shapes.
    p1, p2, p3 = shape_pts
    c12 = ((p1[:,None,:]-p2[None,:,:])**2).sum(axis=2)
    r,c = linear_sum_assignment(c12); p1=p1[r]; p2=p2[c]
    c23 = ((p2[:,None,:]-p3[None,:,:])**2).sum(axis=2)
    r2,c2 = linear_sum_assignment(c23); p2=p2[r2]; p1=p1[r2]; p3=p3[c2]

    cx1,cy1 = p1.mean(axis=0)
    sx, sy = 1.24, 1.4471
    tx0, ty0 = 50, 86

    # Timing: 3.0 portrait; 1.3 transition; each logo 2.0; transitions 1.3; total 14.2.
    kt = [0, 3/14.2, 4.3/14.2, 6.3/14.2, 7.6/14.2, 9.6/14.2, 10.9/14.2, 12.9/14.2, 1]
    keytimes = ';'.join(f'{x:.5f}' for x in kt)

    rows = [
        ('Subject', PROFILE['name']), ('Role', PROFILE['role']), ('Origin', PROFILE['origin']),
        ('Education', PROFILE['education']), ('Status', PROFILE['status']), ('ToolChain', PROFILE['toolchain']),
        ('Core.Lang', PROFILE['languages']), ('Core.Frontend', PROFILE['frontend']), ('Core.Backend', PROFILE['backend']),
        ('Core.Database', PROFILE['database']), ('Core.Infra', PROFILE['infra']),
        ('Grid.Mail', PROFILE['email']), ('Grid.Portfolio', PROFILE['portfolio']), ('Grid.LinkedIn', PROFILE['linkedin']),
        ('Grid.GitHub', '@'+PROFILE['username']), ('Grid.Facebook', '@'+PROFILE['facebook']),
    ]

    s=[]; a=s.append
    a(f'''<svg xmlns="http://www.w3.org/2000/svg" width="1180" height="610" viewBox="0 0 1180 610" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,'Liberation Mono',monospace" role="img" aria-label="{esc(PROFILE['name'])} — profile.sh --live">
<defs>
<linearGradient id="accent" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="{t['VIOLET2']}"><animate attributeName="stop-color" values="{t['VIOLET2']};{t['CYAN']};{t['EMERALD']};{t['VIOLET2']}" dur="10s" repeatCount="indefinite"/></stop><stop offset="0.5" stop-color="{t['CYAN']}"><animate attributeName="stop-color" values="{t['CYAN']};{t['EMERALD']};{t['VIOLET2']};{t['CYAN']}" dur="10s" repeatCount="indefinite"/></stop><stop offset="1" stop-color="{t['EMERALD']}"><animate attributeName="stop-color" values="{t['EMERALD']};{t['VIOLET2']};{t['CYAN']};{t['EMERALD']}" dur="10s" repeatCount="indefinite"/></stop></linearGradient>
<linearGradient id="panelGrad" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{t['BG']}"/><stop offset="1" stop-color="{t['PANEL']}"/></linearGradient>
<filter id="glow3" x="-60%" y="-60%" width="220%" height="220%"><feGaussianBlur stdDeviation="3"/></filter>
<clipPath id="winClip"><rect x="2" y="2" width="1176" height="606" rx="18"/></clipPath>
</defs>
<rect x="2" y="2" width="1176" height="606" rx="18" fill="{t['OUTER']}"/>
<g clip-path="url(#winClip)"><rect x="2" y="2" width="1176" height="606" fill="url(#panelGrad)"/><rect x="2" y="2" width="1176" height="46" fill="{t['BAR']}"/><line x1="2" y1="48" x2="1178" y2="48" stroke="{t['PANEL_LINE']}"/><circle cx="30" cy="25" r="5.5" fill="#ff5f56"/><circle cx="50" cy="25" r="5.5" fill="#ffbd2e"/><circle cx="70" cy="25" r="5.5" fill="#27c93f"/><text x="590" y="29" text-anchor="middle" font-size="12" fill="{t['MUTED']}">{esc(PROFILE['email'])} - % ./profile.sh --live</text>
<text x="38" y="74" font-size="10" letter-spacing="3" fill="{t['DIM']}">VISUAL.MAP</text><rect x="36" y="84" width="400" height="492" rx="10" fill="none" stroke="{t['CYAN']}" stroke-width="2" opacity="0.45" filter="url(#glow3)"/><rect x="36" y="84" width="400" height="492" rx="10" fill="{t['BG']}" stroke="{t['PANEL_LINE']}"/>
''')

    # Intro portrait (duplicate layer) — random interleaved groups.
    a(f'<g transform="translate({tx0},{ty0}) scale({sx:.4f},{sy:.4f})" fill="{t["PORTRAIT"]}" shape-rendering="crispEdges"><set attributeName="opacity" to="0" begin="3.2s"/>')
    for i,g in enumerate(intro_groups):
        if len(g)==0: continue
        begin=0.20 + i*0.022
        a(f'<g opacity="0"><animate attributeName="opacity" values="0;1" dur="0.9s" begin="{begin:.2f}s" fill="freeze" calcMode="spline" keyTimes="0;1" keySplines=".4 0 .2 1"/><path d="{points_to_runs(g)}"/></g>')
    a('</g>')

    # Loop portrait dense layer.
    a(f'<g transform="translate({tx0},{ty0}) scale({sx:.4f},{sy:.4f})" fill="{t["PORTRAIT"]}" shape-rendering="crispEdges" opacity="0"><set attributeName="opacity" to="1" begin="3.2s"/>')
    for g in drift_groups:
        if len(g)==0: continue
        gx,gy=g.mean(axis=0)
        dx=(cx1-gx)*0.42; dy=(cy1-gy)*0.42
        a(f'<g><animateTransform attributeName="transform" type="translate" values="0 0;0 0;{dx:.2f} {dy:.2f};{dx:.2f} {dy:.2f};{dx:.2f} {dy:.2f};{dx:.2f} {dy:.2f};{dx:.2f} {dy:.2f};{dx:.2f} {dy:.2f};0 0" keyTimes="{keytimes}" dur="14.2s" begin="3.2s" repeatCount="indefinite"/><animate attributeName="opacity" values="1;1;0;0;0;0;0;0;1" keyTimes="{keytimes}" dur="14.2s" begin="3.2s" repeatCount="indefinite"/><path d="{points_to_runs(g)}"/></g>')
    a('</g>')

    # Travellers morph through code, brain, triangle.
    a(f'<g transform="translate({tx0},{ty0}) scale({sx:.4f},{sy:.4f})" fill="{t["PORTRAIT"]}" shape-rendering="crispEdges">')
    for i in range(ntrav):
        x1,y1=p1[i]; x2,y2=p2[i]; x3,y3=p3[i]
        xv=f'{x1:.1f};{x1:.1f};{x1:.1f};{x1:.1f};{x2:.1f};{x2:.1f};{x3:.1f};{x3:.1f};{x1:.1f}'
        yv=f'{y1:.1f};{y1:.1f};{y1:.1f};{y1:.1f};{y2:.1f};{y2:.1f};{y3:.1f};{y3:.1f};{y1:.1f}'
        a(f'<rect x="{x1:.1f}" y="{y1:.1f}" width="1.8" height="1.8" opacity="0"><animate attributeName="x" values="{xv}" keyTimes="{keytimes}" dur="14.2s" begin="3.2s" repeatCount="indefinite"/><animate attributeName="y" values="{yv}" keyTimes="{keytimes}" dur="14.2s" begin="3.2s" repeatCount="indefinite"/><animate attributeName="opacity" values="0;0;1;1;1;1;1;1;0" keyTimes="{keytimes}" dur="14.2s" begin="3.2s" repeatCount="indefinite"/></rect>')
    a('</g>')

    # Honest placeholder label: disappears once user regenerates with --photo.
    if not photo:
        a(f'<text x="236" y="559" text-anchor="middle" font-size="9" letter-spacing="1.5" fill="{t["DIM"]}">ABSTRACT PORTRAIT · ADD PHOTO TO PERSONALIZE</text>')

    # Right info panel.
    a(f'<text x="474" y="74" font-size="13" letter-spacing="2" fill="{t["CYAN"]}">SYSTEM.INFO</text><circle cx="1012" cy="70" r="4" fill="#EF4444"><animate attributeName="opacity" values="1;.25;1" dur="1.4s" repeatCount="indefinite"/></circle><text x="1024" y="74" font-size="12" fill="#EF4444">LIVE</text><rect x="1064" y="55" width="88" height="25" rx="12.5" fill="{t["VIOLET2"]}" opacity=".25" stroke="{t["VIOLET"]}"/><text x="1108" y="72" text-anchor="middle" font-size="14" fill="{t["VIOLET"]}">@{PROFILE['username']}</text><line x1="474" y1="84" x2="1144" y2="84" stroke="url(#accent)" stroke-width="1.4" opacity=".75"/>')

    y=111
    for idx,(lab,val) in enumerate(rows):
        if idx==11:
            a(f'<text x="474" y="{y}" font-size="11" fill="{t["DIM"]}">- Contact</text><line x1="552" y1="{y-4}" x2="1140" y2="{y-4}" stroke="{t["DIM"]}" opacity=".28"/>')
            y += 23
        x1,x2,vw=leader_geometry(lab,val)
        a(f'<text x="482" y="{y}" font-size="14" fill="{t["CYAN"]}">{esc(lab)}</text><line x1="{x1:.1f}" y1="{y-4}" x2="{x2:.1f}" y2="{y-4}" stroke="{t["DIM"]}" stroke-dasharray="2 4" opacity=".55"/><text x="1140" y="{y}" text-anchor="end" font-size="14" fill="{t["TEXT"]}" textLength="{vw:.1f}" lengthAdjust="spacingAndGlyphs">{esc(val)}</text>')
        y += 23

    a(f'<text x="474" y="585" font-size="11" fill="{t["EMERALD"]}">&#9656; More about me &amp; projects below in README &#8595;</text><rect x="1137" y="573" width="7" height="13" fill="{t["CYAN"]}"><animate attributeName="opacity" values="1;0;1" dur="1.1s" repeatCount="indefinite"/></rect></g></svg>')
    return ''.join(s)


