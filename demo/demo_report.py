"""Demo: CASPULE multi-configuration bond-network report.

Runs three distinct CASPULE-style simulations:

1. **Polymer chain dynamics** — pre-bonded polymer in solvent. Static
   bond list; the demo shows the integrator working through the
   wrapper.
2. **Dynamic crosslinking** — sticker (type 1) + spacer (type 2) atoms
   with `fix bond/create` forming bonds as the system diffuses.
3. **Bond formation/breaking equilibrium** — `fix bond/create` plus
   `fix bond/break` driving a steady-state bond network.

Each section gets a 3D viewer (atoms as spheres, bonds as sticks),
Plotly time-series for bonds / energy / clusters, a colored
bigraph-viz architecture diagram, and a collapsible JSON tree of the
composite document.
"""

import base64
import json
import os
import tempfile
import time

import numpy as np
from process_bigraph import allocate_core

from pbg_caspule.composites import make_caspule_document
from pbg_caspule.processes import CASPULEProcess


# ────────────────────────────────────────────────────────────────────
# Configs
# ────────────────────────────────────────────────────────────────────


def _build_chain_script(n_beads=50, x_off=1.0, dx=1.2, y=10.0, z=10.0):
    """Generate a LAMMPS script that creates a single linear chain of length n.

    LAMMPS' `variable loop` + `jump SELF` macros can interact awkwardly
    with `$(...)` immediate evaluation, so we just unroll the atom and
    bond construction as plain Python-generated commands. Beads are
    spaced at `dx > 1.0` so `create_atoms single` doesn't reject them
    (the default proximity threshold is 1.0).
    """
    head = """
units lj
atom_style bond
boundary p p p
neighbor 0.4 bin
neigh_modify every 1 delay 0
region box block 0 80 0 20 0 20
create_box 1 box bond/types 1 extra/bond/per/atom 4
mass 1 1.0
"""
    atoms = '\n'.join(
        f'create_atoms 1 single {x_off + i * dx:.3f} {y} {z}'
        for i in range(n_beads))
    bonds = '\n'.join(
        f'create_bonds single/bond 1 {i+1} {i+2}'
        for i in range(n_beads - 1))
    tail = """
pair_style lj/cut 1.5
pair_coeff * * 1.0 1.0 1.5
bond_style harmonic
bond_coeff 1 30.0 1.2
velocity all create 1.0 87287 dist gaussian
fix nve all nve
fix lan all langevin 1.0 1.0 0.5 12345
timestep 0.005
thermo 5
"""
    return head + atoms + '\n' + bonds + tail


CHAIN_SCRIPT = _build_chain_script(50)


CROSSLINK_SCRIPT = """
units lj
atom_style bond
boundary p p p
neighbor 0.5 bin
neigh_modify every 1 delay 0 check yes
region box block 0 8 0 8 0 8
create_box 2 box bond/types 1 extra/bond/per/atom 4 extra/special/per/atom 50
mass * 1.0
lattice sc 1.6
create_atoms 1 box
set group all type/fraction 2 0.5 12345
pair_style soft 2.0
pair_coeff * * 30.0
bond_style harmonic
bond_coeff 1 5.0 1.5
minimize 1e-4 1e-4 100 100
fix nve all nve/limit 0.05
fix lan all langevin 0.4 0.4 1.0 12345
timestep 0.005
fix mkbnd all bond/create 1 1 2 1.8 1 iparam 1 1 jparam 1 2
thermo 5
"""


EQUILIBRIUM_SCRIPT = """
units lj
atom_style bond
boundary p p p
neighbor 0.5 bin
neigh_modify every 1 delay 0 check yes
region box block 0 8 0 8 0 8
create_box 2 box bond/types 1 extra/bond/per/atom 4 extra/special/per/atom 50
mass * 1.0
lattice sc 1.6
create_atoms 1 box
set group all type/fraction 2 0.5 54321
pair_style soft 2.0
pair_coeff * * 30.0
bond_style harmonic
bond_coeff 1 3.0 1.5
minimize 1e-4 1e-4 100 100
fix nve all nve/limit 0.05
fix lan all langevin 0.6 0.6 1.0 54321
timestep 0.005
# Form bonds at short range, break them at long range — equilibrium
fix mkbnd all bond/create 2 1 2 1.7 1 iparam 1 1 jparam 1 2
fix brkbnd all bond/break 4 1 2.4
thermo 5
"""


CONFIGS = [
    {
        'id': 'chain',
        'title': 'Polymer Chain Dynamics',
        'subtitle': '50-bead linear polymer in implicit solvent',
        'description': (
            'A pre-bonded 50-bead linear polymer is thermalised in implicit '
            'solvent (Langevin thermostat) at T=1.0. The bond topology is '
            'fixed: the wrapper exposes the static bond list, the connected '
            'component analysis (one cluster of 50), and the bond energy '
            'oscillations driven by thermal motion.'
        ),
        'script': CHAIN_SCRIPT,
        'n_snapshots': 30,
        'interval': 0.5,
        'color_scheme': 'indigo',
        'camera': [80.0, 30.0, 60.0],
    },
    {
        'id': 'crosslink',
        'title': 'Dynamic Crosslinking',
        'subtitle': 'Sticker + spacer atoms forming bonds via fix bond/create',
        'description': (
            'A 50/50 mixture of sticker (type 1) and spacer (type 2) atoms '
            'on a soft pair potential. `fix bond/create` adds bonds whenever '
            'a sticker comes within range of a free spacer (max 1 bond per '
            'atom). The wrapper reports formed-bond counts each step plus '
            'the live cluster-size distribution as the network grows.'
        ),
        'script': CROSSLINK_SCRIPT,
        'n_snapshots': 30,
        'interval': 0.05,
        'color_scheme': 'emerald',
        'camera': [14.0, 6.0, 14.0],
    },
    {
        'id': 'equilibrium',
        'title': 'Bond Equilibrium',
        'subtitle': 'fix bond/create + fix bond/break — formation vs. dissociation',
        'description': (
            'Same sticker/spacer mixture, but now `fix bond/break` removes '
            'bonds when partners stretch beyond 2.4σ while `fix bond/create` '
            'continues to form new bonds at <1.7σ. The system relaxes to a '
            'dynamic equilibrium between the two rates. The wrapper tracks '
            'the formation and break events as deltas per update.'
        ),
        'script': EQUILIBRIUM_SCRIPT,
        'n_snapshots': 30,
        'interval': 0.05,
        'color_scheme': 'rose',
        'camera': [14.0, 6.0, 14.0],
    },
]


COLOR_SCHEMES = {
    'indigo': {'primary': '#6366f1', 'light': '#e0e7ff', 'dark': '#4338ca'},
    'emerald': {'primary': '#10b981', 'light': '#d1fae5', 'dark': '#059669'},
    'rose': {'primary': '#f43f5e', 'light': '#ffe4e6', 'dark': '#e11d48'},
}


# ────────────────────────────────────────────────────────────────────
# Simulation runner
# ────────────────────────────────────────────────────────────────────


def run_simulation(cfg_entry):
    """Run a single simulation; return snapshots and wall-clock runtime."""
    core = allocate_core()
    core.register_link('CASPULEProcess', CASPULEProcess)

    t0 = time.perf_counter()
    proc = CASPULEProcess(config={'input_script': cfg_entry['script']}, core=core)
    state0 = proc.initial_state()

    snapshots = [_snap(0.0, state0)]
    t = 0.0
    for _ in range(cfg_entry['n_snapshots']):
        result = proc.update({}, interval=cfg_entry['interval'])
        t += cfg_entry['interval']
        snapshots.append(_snap(round(t, 4), result))

    runtime = time.perf_counter() - t0
    proc.close()
    return snapshots, runtime


def _snap(t, s):
    return {
        'time': t,
        'positions': s['positions'],
        'atom_types': s['atom_types'],
        'bonds': s['bonds'],
        'box_dimensions': s['box_dimensions'],
        'num_atoms': s['num_atoms'],
        'num_bonds': s['num_bonds'],
        'formed_bonds': s['formed_bonds'],
        'broken_bonds': s['broken_bonds'],
        'bond_energy': s['bond_energy'],
        'potential_energy': s['potential_energy'],
        'kinetic_energy': s['kinetic_energy'],
        'temperature': s['temperature'],
        'num_clusters': s['num_clusters'],
        'largest_cluster': s['largest_cluster'],
        'cluster_sizes': s['cluster_sizes'],
    }


# ────────────────────────────────────────────────────────────────────
# Bigraph diagram
# ────────────────────────────────────────────────────────────────────


def generate_bigraph_image(cfg_entry):
    """Generate a colored bigraph-viz PNG of the (simplified) document."""
    from bigraph_viz import plot_bigraph

    doc = {
        'caspule': {
            '_type': 'process',
            'address': 'local:CASPULEProcess',
            'config': {'input_script': '<script>'},
            'interval': cfg_entry['interval'],
            'inputs': {},
            'outputs': {
                'num_bonds': ['stores', 'num_bonds'],
                'formed_bonds': ['stores', 'formed_bonds'],
                'broken_bonds': ['stores', 'broken_bonds'],
                'num_clusters': ['stores', 'num_clusters'],
                'temperature': ['stores', 'temperature'],
            },
        },
        'stores': {},
        'emitter': {
            '_type': 'step',
            'address': 'local:ram-emitter',
            'config': {'emit': {
                'num_bonds': 'integer',
                'temperature': 'float',
                'time': 'float',
            }},
            'inputs': {
                'num_bonds': ['stores', 'num_bonds'],
                'temperature': ['stores', 'temperature'],
                'time': ['global_time'],
            },
        },
    }

    node_colors = {
        ('caspule',): '#6366f1',
        ('emitter',): '#8b5cf6',
        ('stores',): '#e0e7ff',
    }

    outdir = tempfile.mkdtemp()
    plot_bigraph(
        state=doc, out_dir=outdir, filename='bigraph',
        file_format='png',
        remove_process_place_edges=True,
        rankdir='LR',
        node_fill_colors=node_colors,
        node_label_size='16pt',
        port_labels=False,
        dpi='150',
    )
    with open(os.path.join(outdir, 'bigraph.png'), 'rb') as f:
        return 'data:image/png;base64,' + base64.b64encode(f.read()).decode()


def build_pbg_document(cfg_entry):
    """Trimmed PBG composite document for the JSON tree viewer."""
    return make_caspule_document(
        input_script='<...LAMMPS/CASPULE script for ' + cfg_entry['id'] + '...>',
        interval=cfg_entry['interval'],
    )


# ────────────────────────────────────────────────────────────────────
# HTML
# ────────────────────────────────────────────────────────────────────


def generate_html(sim_results, output_path):
    sections_html = []
    js_data = {}

    for idx, (cfg, (snapshots, runtime)) in enumerate(sim_results):
        sid = cfg['id']
        cs = COLOR_SCHEMES[cfg['color_scheme']]
        first = snapshots[0]
        last = snapshots[-1]
        n_atoms = first['num_atoms']

        # Time-series
        times = [s['time'] for s in snapshots]
        n_bonds = [s['num_bonds'] for s in snapshots]
        formed = [s['formed_bonds'] for s in snapshots]
        broken = [s['broken_bonds'] for s in snapshots]
        bond_e = [s['bond_energy'] for s in snapshots]
        pot_e = [s['potential_energy'] for s in snapshots]
        kin_e = [s['kinetic_energy'] for s in snapshots]
        temp = [s['temperature'] for s in snapshots]
        n_clusters = [s['num_clusters'] for s in snapshots]
        largest = [s['largest_cluster'] for s in snapshots]

        # Final cluster-size histogram (sizes >=2 only — singletons dominate
        # otherwise and obscure the network structure)
        final_sizes = [c for c in last['cluster_sizes'] if c >= 2]
        if final_sizes:
            max_size = max(final_sizes)
            bins = list(range(2, max_size + 2))
            counts = [final_sizes.count(b) for b in bins[:-1]]
        else:
            bins = [2]
            counts = [0]

        # JS data (positions + bonds for the 3D viewer)
        js_data[sid] = {
            'snapshots': [{
                'time': s['time'],
                'positions': s['positions'],
                'atom_types': s['atom_types'],
                'bonds': s['bonds'],
            } for s in snapshots],
            'box': first['box_dimensions'],
            'camera': cfg['camera'],
            'charts': {
                'times': times,
                'num_bonds': n_bonds,
                'formed': formed,
                'broken': broken,
                'bond_energy': bond_e,
                'potential_energy': pot_e,
                'kinetic_energy': kin_e,
                'temperature': temp,
                'num_clusters': n_clusters,
                'largest': largest,
                'hist_bins': bins[:-1],
                'hist_counts': counts,
            },
        }

        print(f'  Generating bigraph diagram for {sid}...')
        bigraph_img = generate_bigraph_image(cfg)
        pbg_docs[sid] = build_pbg_document(cfg)

        bonds0 = first['num_bonds']
        bonds1 = last['num_bonds']
        formed_total = sum(formed)
        broken_total = sum(broken)

        section = f'''
    <div class="sim-section" id="sim-{sid}">
      <div class="sim-header" style="border-left: 4px solid {cs['primary']};">
        <div class="sim-number" style="background:{cs['light']}; color:{cs['dark']};">{idx+1}</div>
        <div>
          <h2 class="sim-title">{cfg['title']}</h2>
          <p class="sim-subtitle">{cfg['subtitle']}</p>
        </div>
      </div>
      <p class="sim-description">{cfg['description']}</p>

      <div class="metrics-row">
        <div class="metric"><span class="metric-label">Atoms</span><span class="metric-value">{n_atoms:,}</span></div>
        <div class="metric"><span class="metric-label">Bonds (start &rarr; end)</span><span class="metric-value">{bonds0} &rarr; {bonds1}</span></div>
        <div class="metric"><span class="metric-label">Formed (total)</span><span class="metric-value">{formed_total}</span></div>
        <div class="metric"><span class="metric-label">Broken (total)</span><span class="metric-value">{broken_total}</span></div>
        <div class="metric"><span class="metric-label">Final clusters</span><span class="metric-value">{last['num_clusters']}</span><span class="metric-sub">largest = {last['largest_cluster']}</span></div>
        <div class="metric"><span class="metric-label">Snapshots</span><span class="metric-value">{len(snapshots)}</span></div>
        <div class="metric"><span class="metric-label">Runtime</span><span class="metric-value">{runtime:.2f}s</span></div>
      </div>

      <h3 class="subsection-title">3D Bond Network Viewer</h3>
      <div class="viewer-wrap">
        <canvas id="canvas-{sid}" class="mesh-canvas"></canvas>
        <div class="viewer-info">
          Type 1 = sticker (orange) &middot; Type 2 = spacer (blue)<br>
          Drag to rotate &middot; Scroll to zoom
        </div>
        <div class="slider-controls">
          <button class="play-btn" style="border-color:{cs['primary']}; color:{cs['primary']};" onclick="togglePlay('{sid}')">Play</button>
          <label>Time</label>
          <input type="range" class="time-slider" id="slider-{sid}" min="0" max="{len(snapshots)-1}" value="0" step="1"
                 style="accent-color:{cs['primary']};">
          <span class="time-val" id="tval-{sid}">t = 0</span>
        </div>
      </div>

      <h3 class="subsection-title">Bond Network &amp; Energy</h3>
      <div class="charts-row">
        <div class="chart-box"><div id="chart-bonds-{sid}" class="chart"></div></div>
        <div class="chart-box"><div id="chart-events-{sid}" class="chart"></div></div>
        <div class="chart-box"><div id="chart-energy-{sid}" class="chart"></div></div>
        <div class="chart-box"><div id="chart-clusters-{sid}" class="chart"></div></div>
      </div>

      <div class="pbg-row">
        <div class="pbg-col">
          <h3 class="subsection-title">Bigraph Architecture</h3>
          <div class="bigraph-img-wrap">
            <img src="{bigraph_img}" alt="Bigraph architecture diagram">
          </div>
        </div>
        <div class="pbg-col">
          <h3 class="subsection-title">Composite Document</h3>
          <div class="json-tree" id="json-{sid}"></div>
        </div>
      </div>
    </div>
'''
        sections_html.append(section)

    nav_items = ''.join(
        f'<a href="#sim-{c["id"]}" class="nav-link" '
        f'style="border-color:{COLOR_SCHEMES[c["color_scheme"]]["primary"]};">'
        f'{c["title"]}</a>'
        for c in [r[0] for r in sim_results])

    # The template uses many literal { } pairs (CSS, JS), so we
    # substitute via plain string replace rather than .format().
    html = (
        HTML_TEMPLATE
        .replace('__NAV_ITEMS__', nav_items)
        .replace('__SECTIONS__', ''.join(sections_html))
        .replace('__DATA__', json.dumps(js_data))
        .replace('__DOCS__', json.dumps(pbg_docs, indent=2))
    )

    with open(output_path, 'w') as f:
        f.write(html)
    print(f'Report saved to {output_path}')


pbg_docs = {}


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CASPULE Bond Network Report</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
       background:#fff; color:#1e293b; line-height:1.6; }
.page-header {
  background:linear-gradient(135deg,#f8fafc 0%,#eef2ff 50%,#fdf2f8 100%);
  border-bottom:1px solid #e2e8f0; padding:3rem;
}
.page-header h1 { font-size:2.2rem; font-weight:800; color:#0f172a; margin-bottom:.3rem; }
.page-header p { color:#64748b; font-size:.95rem; max-width:780px; }
.page-header code { background:#e2e8f0; padding:0 .25em; border-radius:4px;
                    font-family:'SF Mono',Menlo,monospace; font-size:.85em; }
.nav { display:flex; gap:.8rem; padding:1rem 3rem; background:#f8fafc;
        border-bottom:1px solid #e2e8f0; position:sticky; top:0; z-index:100; }
.nav-link { padding:.4rem 1rem; border-radius:8px; border:1.5px solid;
             text-decoration:none; font-size:.85rem; font-weight:600;
             color:#334155; transition:all .15s; }
.nav-link:hover { transform:translateY(-1px); box-shadow:0 2px 8px rgba(0,0,0,.08); }
.sim-section { padding:2.5rem 3rem; border-bottom:1px solid #e2e8f0; }
.sim-header { display:flex; align-items:center; gap:1rem; margin-bottom:.8rem;
               padding-left:1rem; }
.sim-number { width:36px; height:36px; border-radius:10px; display:flex;
               align-items:center; justify-content:center; font-weight:800; font-size:1.1rem; }
.sim-title { font-size:1.5rem; font-weight:700; color:#0f172a; }
.sim-subtitle { font-size:.9rem; color:#64748b; }
.sim-description { color:#475569; font-size:.9rem; margin-bottom:1.5rem; max-width:820px; }
.subsection-title { font-size:1.05rem; font-weight:600; color:#334155;
                     margin:1.5rem 0 .8rem; }
.metrics-row { display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr));
                gap:.8rem; margin-bottom:1.5rem; }
.metric { background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px;
           padding:.8rem; text-align:center; }
.metric-label { display:block; font-size:.7rem; text-transform:uppercase;
                 letter-spacing:.06em; color:#94a3b8; margin-bottom:.2rem; }
.metric-value { display:block; font-size:1.3rem; font-weight:700; color:#1e293b; }
.metric-sub { display:block; font-size:.7rem; color:#94a3b8; }
.viewer-wrap { position:relative; background:#f1f5f9; border:1px solid #e2e8f0;
                border-radius:14px; overflow:hidden; margin-bottom:1rem; }
.mesh-canvas { width:100%; height:500px; display:block; cursor:grab; }
.mesh-canvas:active { cursor:grabbing; }
.viewer-info { position:absolute; top:.8rem; left:.8rem; background:rgba(255,255,255,.92);
                border:1px solid #e2e8f0; border-radius:8px; padding:.5rem .8rem;
                font-size:.75rem; color:#64748b; backdrop-filter:blur(4px); }
.viewer-info strong { color:#1e293b; }
.slider-controls { position:absolute; bottom:0; left:0; right:0;
                    background:linear-gradient(transparent,rgba(241,245,249,.97));
                    padding:1.5rem 1.5rem 1rem; display:flex; align-items:center; gap:.8rem; }
.slider-controls label { font-size:.8rem; color:#64748b; }
.time-slider { flex:1; height:5px; }
.time-val { font-size:.95rem; font-weight:600; color:#334155; min-width:120px; text-align:right; }
.play-btn { background:#fff; border:1.5px solid; padding:.3rem .8rem; border-radius:7px;
             cursor:pointer; font-size:.8rem; font-weight:600; transition:all .15s; }
.play-btn:hover { transform:scale(1.05); }
.charts-row { display:grid; grid-template-columns:1fr 1fr; gap:1rem; margin-bottom:1rem; }
.chart-box { background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; overflow:hidden; }
.chart { height:280px; }
.pbg-row { display:grid; grid-template-columns:1fr 1fr; gap:1.5rem; margin-top:1rem; }
.pbg-col { min-width:0; }
.bigraph-img-wrap { background:#fafafa; border:1px solid #e2e8f0; border-radius:10px;
                     padding:1.5rem; text-align:center; }
.bigraph-img-wrap img { max-width:100%; height:auto; }
.json-tree { background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px;
              padding:1rem; max-height:500px; overflow-y:auto; font-family:'SF Mono',
              Menlo,Monaco,'Courier New',monospace; font-size:.78rem; line-height:1.5; }
.jt-key { color:#7c3aed; font-weight:600; }
.jt-str { color:#059669; }
.jt-num { color:#2563eb; }
.jt-bool { color:#d97706; }
.jt-null { color:#94a3b8; }
.jt-toggle { cursor:pointer; user-select:none; color:#94a3b8; margin-right:.3rem; }
.jt-toggle:hover { color:#1e293b; }
.jt-collapsed { display:none; }
.jt-bracket { color:#64748b; }
.footer { text-align:center; padding:2rem; color:#94a3b8; font-size:.8rem;
           border-top:1px solid #e2e8f0; }
@media(max-width:900px) {
  .charts-row,.pbg-row { grid-template-columns:1fr; }
  .sim-section,.page-header { padding:1.5rem; }
}
</style>
</head>
<body>

<div class="page-header">
  <h1>CASPULE Bond Network Report</h1>
  <p><strong>pbg-caspule</strong> wraps CASPULE — a modified LAMMPS that
  adds <code>fix bond/create/random</code> on top of the built-in
  <code>fix bond/create</code> and <code>fix bond/break</code> — as a
  process-bigraph Process. The wrapper pulls the live bond list from
  LAMMPS at every update step and reports it through dedicated PBG
  ports: bond list, formed/broken event counts, per-type bond counts,
  bond energy, and connected-component cluster sizes.</p>
</div>

<div class="nav">__NAV_ITEMS__</div>

__SECTIONS__

<div class="footer">
  Generated by <strong>pbg-caspule</strong> &mdash;
  CASPULE / LAMMPS + process-bigraph &mdash;
  Sticker–spacer polymer dynamics with first-class bond ports
</div>

<script>
const DATA = __DATA__;
const DOCS = __DOCS__;

// ─── JSON Tree ───
function renderJson(obj, depth) {
  if (depth === undefined) depth = 0;
  if (obj === null) return '<span class="jt-null">null</span>';
  if (typeof obj === 'boolean') return '<span class="jt-bool">' + obj + '</span>';
  if (typeof obj === 'number') return '<span class="jt-num">' + obj + '</span>';
  if (typeof obj === 'string') return '<span class="jt-str">"' + obj.replace(/</g,'&lt;') + '"</span>';
  if (Array.isArray(obj)) {
    if (obj.length === 0) return '<span class="jt-bracket">[]</span>';
    if (obj.length <= 5 && obj.every(x => typeof x !== 'object' || x === null)) {
      const items = obj.map(x => renderJson(x, depth+1)).join(', ');
      return '<span class="jt-bracket">[</span>' + items + '<span class="jt-bracket">]</span>';
    }
    const id = 'jt' + Math.random().toString(36).slice(2,9);
    let html = '<span class="jt-toggle" onclick="toggleJt(\'' + id + '\')">&blacktriangledown;</span>';
    html += '<span class="jt-bracket">[</span> <span style="color:#94a3b8;font-size:.7rem;">' + obj.length + ' items</span>';
    html += '<div id="' + id + '" style="margin-left:1.2rem;">';
    obj.forEach((v, i) => { html += '<div>' + renderJson(v, depth+1) + (i < obj.length-1 ? ',' : '') + '</div>'; });
    html += '</div><span class="jt-bracket">]</span>';
    return html;
  }
  if (typeof obj === 'object') {
    const keys = Object.keys(obj);
    if (keys.length === 0) return '<span class="jt-bracket">{}</span>';
    const id = 'jt' + Math.random().toString(36).slice(2,9);
    const collapsed = depth >= 2;
    let html = '<span class="jt-toggle" onclick="toggleJt(\'' + id + '\')">' +
               (collapsed ? '&blacktriangleright;' : '&blacktriangledown;') + '</span>';
    html += '<span class="jt-bracket">{</span>';
    html += '<div id="' + id + '"' + (collapsed ? ' class="jt-collapsed"' : '') + ' style="margin-left:1.2rem;">';
    keys.forEach((k, i) => {
      html += '<div><span class="jt-key">' + k + '</span>: ' +
              renderJson(obj[k], depth+1) + (i < keys.length-1 ? ',' : '') + '</div>';
    });
    html += '</div><span class="jt-bracket">}</span>';
    return html;
  }
  return String(obj);
}
function toggleJt(id) {
  const el = document.getElementById(id);
  if (el.classList.contains('jt-collapsed')) {
    el.classList.remove('jt-collapsed');
    const prev = el.previousElementSibling;
    if (prev && prev.previousElementSibling && prev.previousElementSibling.classList.contains('jt-toggle'))
      prev.previousElementSibling.innerHTML = '&blacktriangledown;';
  } else {
    el.classList.add('jt-collapsed');
    const prev = el.previousElementSibling;
    if (prev && prev.previousElementSibling && prev.previousElementSibling.classList.contains('jt-toggle'))
      prev.previousElementSibling.innerHTML = '&blacktriangleright;';
  }
}
Object.keys(DOCS).forEach(sid => {
  const el = document.getElementById('json-' + sid);
  if (el) el.innerHTML = renderJson(DOCS[sid], 0);
});

// ─── Three.js viewers ───
const viewers = {};
const playStates = {};

const ATOM_COLORS = { 1: 0xf97316, 2: 0x3b82f6, default: 0x64748b };
const BOND_COLOR = 0x1e293b;

function initViewer(sid) {
  const d = DATA[sid];
  const canvas = document.getElementById('canvas-' + sid);
  const W = canvas.parentElement.clientWidth;
  const H = 500;

  const renderer = new THREE.WebGLRenderer({canvas, antialias:true});
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(W, H);
  renderer.setClearColor(0xf1f5f9);

  const scene = new THREE.Scene();
  const cam = new THREE.PerspectiveCamera(45, W/H, 0.01, 500);
  cam.position.set(d.camera[0], d.camera[1], d.camera[2]);

  const controls = new THREE.OrbitControls(cam, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.autoRotate = true;
  controls.autoRotateSpeed = 0.6;

  scene.add(new THREE.AmbientLight(0xffffff, 0.55));
  const dl1 = new THREE.DirectionalLight(0xffffff, 0.7);
  dl1.position.set(10, 15, 10); scene.add(dl1);
  const dl2 = new THREE.DirectionalLight(0xcbd5e1, 0.4);
  dl2.position.set(-8, -5, -8); scene.add(dl2);

  // Center camera on box
  const box = d.box;
  controls.target.set(box[0]/2, box[1]/2, box[2]/2);

  // One mesh per atom, colored by type. We use InstancedMesh for performance.
  const snap0 = d.snapshots[0];
  const N = snap0.positions.length;
  const sphereGeom = new THREE.SphereGeometry(0.35, 12, 12);
  const sphereMat = new THREE.MeshPhongMaterial({ shininess: 80, vertexColors: true });
  const inst = new THREE.InstancedMesh(sphereGeom, sphereMat, N);
  const colorAttr = new Float32Array(N * 3);
  inst.instanceColor = new THREE.InstancedBufferAttribute(colorAttr, 3);
  scene.add(inst);

  // Bond line segments
  const bondGeom = new THREE.BufferGeometry();
  const bondMat = new THREE.LineBasicMaterial({ color: BOND_COLOR, linewidth: 2 });
  const bondLines = new THREE.LineSegments(bondGeom, bondMat);
  scene.add(bondLines);

  const dummy = new THREE.Object3D();
  const tmpColor = new THREE.Color();

  function updateSnapshot(idx) {
    const snap = d.snapshots[idx];
    for (let i = 0; i < N; i++) {
      const p = snap.positions[i];
      dummy.position.set(p[0], p[1], p[2]);
      dummy.updateMatrix();
      inst.setMatrixAt(i, dummy.matrix);
      const t = snap.atom_types[i] || 1;
      const c = ATOM_COLORS[t] !== undefined ? ATOM_COLORS[t] : ATOM_COLORS.default;
      tmpColor.setHex(c);
      colorAttr[i*3]   = tmpColor.r;
      colorAttr[i*3+1] = tmpColor.g;
      colorAttr[i*3+2] = tmpColor.b;
    }
    inst.instanceMatrix.needsUpdate = true;
    inst.instanceColor.needsUpdate = true;

    // Build bond segments
    const bonds = snap.bonds;
    const seg = new Float32Array(bonds.length * 6);
    for (let k = 0; k < bonds.length; k++) {
      const a = bonds[k][1] - 1;
      const b = bonds[k][2] - 1;
      if (a < 0 || b < 0 || a >= N || b >= N) continue;
      const pa = snap.positions[a];
      const pb = snap.positions[b];
      seg[k*6]   = pa[0]; seg[k*6+1] = pa[1]; seg[k*6+2] = pa[2];
      seg[k*6+3] = pb[0]; seg[k*6+4] = pb[1]; seg[k*6+5] = pb[2];
    }
    bondGeom.setAttribute('position', new THREE.BufferAttribute(seg, 3));
    bondGeom.computeBoundingSphere();
  }

  updateSnapshot(0);

  const slider = document.getElementById('slider-' + sid);
  const tval = document.getElementById('tval-' + sid);
  slider.addEventListener('input', () => {
    const idx = parseInt(slider.value);
    updateSnapshot(idx);
    tval.textContent = 't = ' + d.snapshots[idx].time +
                       '  | bonds = ' + d.snapshots[idx].bonds.length;
  });
  tval.textContent = 't = ' + d.snapshots[0].time +
                     '  | bonds = ' + d.snapshots[0].bonds.length;

  viewers[sid] = { renderer, scene, cam, controls, updateSnapshot, slider, tval };
  playStates[sid] = { playing: false, interval: null };

  function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, cam);
  }
  animate();
}

function togglePlay(sid) {
  const ps = playStates[sid];
  const v = viewers[sid];
  const d = DATA[sid];
  const btn = event.target;
  ps.playing = !ps.playing;
  if (ps.playing) {
    btn.textContent = 'Pause';
    v.controls.autoRotate = false;
    ps.interval = setInterval(() => {
      let idx = parseInt(v.slider.value) + 1;
      if (idx >= d.snapshots.length) idx = 0;
      v.slider.value = idx;
      v.updateSnapshot(idx);
      v.tval.textContent = 't = ' + d.snapshots[idx].time +
                           '  | bonds = ' + d.snapshots[idx].bonds.length;
    }, 250);
  } else {
    btn.textContent = 'Play';
    v.controls.autoRotate = true;
    clearInterval(ps.interval);
  }
}

Object.keys(DATA).forEach(sid => initViewer(sid));

// ─── Plotly charts ───
const pLayout = {
  paper_bgcolor:'#f8fafc', plot_bgcolor:'#f8fafc',
  font:{ color:'#64748b', family:'-apple-system,sans-serif', size:11 },
  margin:{ l:55, r:15, t:35, b:40 },
  xaxis:{ gridcolor:'#e2e8f0', zerolinecolor:'#e2e8f0' },
  yaxis:{ gridcolor:'#e2e8f0', zerolinecolor:'#e2e8f0' },
};
const pCfg = { responsive:true, displayModeBar:false };

Object.keys(DATA).forEach(sid => {
  const c = DATA[sid].charts;

  Plotly.newPlot('chart-bonds-'+sid, [{
    x:c.times, y:c.num_bonds, type:'scatter', mode:'lines+markers',
    line:{ color:'#6366f1', width:2 }, marker:{ size:4 },
    fill:'tozeroy', fillcolor:'rgba(99,102,241,0.07)',
  }], Object.assign({}, pLayout, {
    title:{ text:'Total bonds vs time', font:{ size:12, color:'#334155' } },
    xaxis: Object.assign({}, pLayout.xaxis, { title:{ text:'Time', font:{ size:10 } } }),
    yaxis: Object.assign({}, pLayout.yaxis, { title:{ text:'Bonds', font:{ size:10 } } }),
    showlegend:false,
  }), pCfg);

  Plotly.newPlot('chart-events-'+sid, [
    { x:c.times, y:c.formed, type:'bar', name:'Formed',
      marker:{ color:'#10b981' } },
    { x:c.times, y:c.broken.map(v => -v), type:'bar', name:'Broken',
      marker:{ color:'#f43f5e' } },
  ], Object.assign({}, pLayout, {
    title:{ text:'Bond formation / breaking events', font:{ size:12, color:'#334155' } },
    xaxis: Object.assign({}, pLayout.xaxis, { title:{ text:'Time', font:{ size:10 } } }),
    yaxis: Object.assign({}, pLayout.yaxis, { title:{ text:'Δ bonds per step', font:{ size:10 } } }),
    barmode:'relative',
    legend:{ font:{ size:9 }, bgcolor:'rgba(0,0,0,0)' }, showlegend:true,
  }), pCfg);

  Plotly.newPlot('chart-energy-'+sid, [
    { x:c.times, y:c.bond_energy, type:'scatter', mode:'lines',
      line:{ color:'#6366f1', width:1.6 }, name:'Bond' },
    { x:c.times, y:c.potential_energy, type:'scatter', mode:'lines',
      line:{ color:'#10b981', width:1.6 }, name:'Potential' },
    { x:c.times, y:c.kinetic_energy, type:'scatter', mode:'lines',
      line:{ color:'#f43f5e', width:1.6 }, name:'Kinetic' },
  ], Object.assign({}, pLayout, {
    title:{ text:'Energy components', font:{ size:12, color:'#334155' } },
    xaxis: Object.assign({}, pLayout.xaxis, { title:{ text:'Time', font:{ size:10 } } }),
    yaxis: Object.assign({}, pLayout.yaxis, { title:{ text:'Energy', font:{ size:10 } } }),
    legend:{ font:{ size:9 }, bgcolor:'rgba(0,0,0,0)' }, showlegend:true,
  }), pCfg);

  Plotly.newPlot('chart-clusters-'+sid, [
    { x:c.times, y:c.num_clusters, type:'scatter', mode:'lines',
      line:{ color:'#8b5cf6', width:2 }, name:'# clusters', yaxis:'y' },
    { x:c.times, y:c.largest, type:'scatter', mode:'lines',
      line:{ color:'#f59e0b', width:2 }, name:'Largest cluster', yaxis:'y2' },
  ], Object.assign({}, pLayout, {
    title:{ text:'Cluster connectivity', font:{ size:12, color:'#334155' } },
    xaxis: Object.assign({}, pLayout.xaxis, { title:{ text:'Time', font:{ size:10 } } }),
    yaxis: Object.assign({}, pLayout.yaxis, {
      title:{ text:'# clusters', font:{ size:10, color:'#8b5cf6' } } }),
    yaxis2:{ title:{ text:'Largest size', font:{ size:10, color:'#f59e0b' } },
             gridcolor:'#e2e8f0', overlaying:'y', side:'right' },
    legend:{ font:{ size:9 }, bgcolor:'rgba(0,0,0,0)' }, showlegend:true,
  }), pCfg);
});
</script>
</body>
</html>"""


def run_demo():
    demo_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(demo_dir, 'report.html')

    sim_results = []
    for cfg in CONFIGS:
        print(f'Running: {cfg["title"]}...')
        snapshots, runtime = run_simulation(cfg)
        sim_results.append((cfg, (snapshots, runtime)))
        last = snapshots[-1]
        print(f'  Runtime: {runtime:.2f}s | snapshots: {len(snapshots)} | '
              f'final bonds: {last["num_bonds"]} | clusters: {last["num_clusters"]}')

    print('Generating HTML report...')
    generate_html(sim_results, output_path)

    # Open in Safari
    import subprocess
    subprocess.run(['open', '-a', 'Safari', output_path])


if __name__ == '__main__':
    run_demo()
