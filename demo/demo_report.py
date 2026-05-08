"""Demo: CASPULE multi-configuration polymer-network report.

Three bond-network experiments showcasing the wrapper's first-class
bond ports:

1. **Self-assembly polymerization** — 500 free monomers polymerise
   into linear chains and rings under `fix bond/create` with `iparam
   2 1 jparam 2 1` (max two bonds per atom). The wrapper reports the
   chain-length distribution as the system gels.
2. **Sticker–spacer condensation** — 12 pre-bonded heteropolymer
   chains with sticker (type 1) and spacer (type 2) beads. Stickers
   crosslink across chains via a second bond type, driving a
   percolation transition: 12 chains → 1 condensate within a few
   steps.
3. **Associative network turnover** — 48 short bifunctional chains
   in a dense pack with `fix bond/create` + `fix bond/break` running
   simultaneously. Crosslinks reach a steady-state count with
   non-zero formation **and** breaking rates per step — the
   signature of an associative polymer.
4. **Programmed atom removal** — same polymer self-assembly as #1,
   but at a scheduled mid-run step we feed the new
   `atoms_to_remove` input port a list of atom IDs taken from the
   largest cluster. The wrapper issues `delete_atoms group ...` to
   LAMMPS before the next integration window, and the bond-network
   plot shows a sharp pruning event with no need for an external
   composite.

Each section gets a 3D viewer (atoms colored by type, bonds colored
by type), Plotly time-series for bonds / events / energy / clusters,
a final-step cluster-size histogram, a colored bigraph-viz
architecture diagram, and a collapsible JSON tree of the composite
document.
"""

import base64
import json
import os
import tempfile
import time

from process_bigraph import allocate_core

from pbg_caspule.composites import make_caspule_document
from pbg_caspule.processes import CASPULEProcess


# ────────────────────────────────────────────────────────────────────
# LAMMPS script builders
# ────────────────────────────────────────────────────────────────────


def build_polymerization_script(box=9.0, n_atoms=500, seed=12345):
    """Free-monomer polymerization: random monomers + bond/create with iparam 2.

    `iparam/jparam 2 1` caps each atom at two bonds, so the network
    can only grow into linear chains and rings — no branching. The
    wrapper's `cluster_sizes` port directly reports the chain-length
    distribution.
    """
    return f"""
units lj
atom_style bond
boundary p p p
neighbor 0.5 bin
neigh_modify every 1 delay 0 check yes
region box block 0 {box} 0 {box} 0 {box}
create_box 1 box bond/types 1 extra/bond/per/atom 2 extra/special/per/atom 30
mass 1 1.0
create_atoms 1 random {n_atoms} {seed} box overlap 0.85 maxtry 100
pair_style soft 1.6
pair_coeff * * 25.0
bond_style harmonic
bond_coeff 1 5.0 1.2
minimize 1e-4 1e-4 300 300
fix nve all nve/limit 0.05
fix lan all langevin 0.6 0.6 1.0 {seed}
timestep 0.005
# Form bonds at <1.4σ; max 2 bonds per atom -> linear chains/rings only
fix mkbnd all bond/create 4 1 1 1.4 1 iparam 2 1 jparam 2 1
thermo 5
"""


def build_condensate_script(grid_x=4, grid_y=3, chain_len=12,
                            spacing=1.8, dx=1.2, seed=54321):
    """Sticker–spacer condensate: pre-built chains + dynamic crosslinks.

    Each chain has the repeating bead pattern [sticker, spacer,
    spacer]. Type-1 stickers crosslink across chains via bond type 2,
    while the type-1 backbone bonds remain fixed. The wrapper reports
    the two bond populations independently through `bonds_by_type`.

    Chains are arranged on a grid so they don't overlap at startup
    (`create_atoms single` rejects atoms within 1.0σ of an existing
    one). Box dimensions are sized to fit the grid tightly.
    """
    pattern = [1, 2, 2]
    box_x = chain_len * dx + 2.0
    box_y = grid_x * spacing + 2.0
    box_z = grid_y * spacing + 2.0

    atoms, bonds = [], []
    aid = 0
    for ix in range(grid_x):
        for iy in range(grid_y):
            y0 = (ix + 0.5) * spacing
            z0 = (iy + 0.5) * spacing
            for k in range(chain_len):
                aid += 1
                t = pattern[k % len(pattern)]
                atoms.append(
                    f'create_atoms {t} single {1.0 + k*dx:.3f} {y0:.3f} {z0:.3f}')
                if k > 0:
                    bonds.append(f'create_bonds single/bond 1 {aid-1} {aid}')

    head = (
        f'units lj\natom_style bond\nboundary p p p\n'
        f'neighbor 0.5 bin\nneigh_modify every 1 delay 0 check yes\n'
        f'region box block 0 {box_x:.2f} 0 {box_y:.2f} 0 {box_z:.2f}\n'
        f'create_box 2 box bond/types 2 extra/bond/per/atom 8 '
        f'extra/special/per/atom 100\nmass * 1.0\n'
    )
    tail = f"""
group stickers type 1
pair_style soft 1.5
pair_coeff * * 15.0
bond_style harmonic
bond_coeff 1 30.0 1.2
bond_coeff 2 8.0 1.5
minimize 1e-4 1e-4 200 200
fix nve all nve/limit 0.05
fix lan all langevin 0.8 0.8 1.0 {seed}
timestep 0.005
# Stickers crosslink (bond type 2) at <1.5σ; up to 3 bonds per sticker
fix mkbnd stickers bond/create 4 1 1 1.5 2 iparam 3 1 jparam 3 1
thermo 5
"""
    return head + '\n'.join(atoms) + '\n' + '\n'.join(bonds) + tail


def build_associative_script(grid_x=4, grid_y=4, grid_z=3, chain_len=5,
                             spacing=1.6, dx=1.2, seed=67890):
    """Associative network: dense bifunctional chains, formation + breaking.

    Each chain is [sticker, spacer, spacer, spacer, sticker]. Both
    `fix bond/create` and `fix bond/break` are active with stochastic
    `prob` parameters, driving the system to a steady state where
    formed_bonds and broken_bonds are both non-zero per step. The
    crosslink count `bonds_by_type['2']` plateaus while individual
    bonds turn over.
    """
    pattern = [1, 2, 2, 2, 1]
    chain_x_extent = chain_len * dx + 0.4
    box_x = grid_z * chain_x_extent + 2.0
    box_y = grid_x * spacing + 2.0
    box_z = grid_y * spacing + 2.0

    atoms, bonds = [], []
    aid = 0
    for ix in range(grid_x):
        for iy in range(grid_y):
            for iz in range(grid_z):
                y0 = (ix + 0.5) * spacing
                z0 = (iy + 0.5) * spacing
                x0 = 1.0 + iz * chain_x_extent
                for k in range(chain_len):
                    aid += 1
                    t = pattern[k]
                    atoms.append(
                        f'create_atoms {t} single {x0 + k*dx:.3f} {y0:.3f} {z0:.3f}')
                    if k > 0:
                        bonds.append(
                            f'create_bonds single/bond 1 {aid-1} {aid}')

    head = (
        f'units lj\natom_style bond\nboundary p p p\n'
        f'neighbor 0.5 bin\nneigh_modify every 1 delay 0 check yes\n'
        f'region box block 0 {box_x:.2f} 0 {box_y:.2f} 0 {box_z:.2f}\n'
        f'create_box 2 box bond/types 2 extra/bond/per/atom 8 '
        f'extra/special/per/atom 200\nmass * 1.0\n'
    )
    tail = f"""
group stickers type 1
pair_style soft 1.5
pair_coeff * * 12.0
bond_style harmonic
bond_coeff 1 30.0 1.2
bond_coeff 2 8.0 1.4
minimize 1e-4 1e-4 200 200
fix nve all nve/limit 0.05
fix lan all langevin 0.7 0.7 1.0 {seed}
timestep 0.005
# Crosslinks form at <1.5σ (prob 0.6) and break beyond 2.1σ (prob 0.15)
fix mkbnd stickers bond/create 4 1 1 1.5 2 iparam 4 1 jparam 4 1 prob 0.6 {seed * 7 + 1}
fix brkbnd all bond/break 8 2 2.1 prob 0.15 {seed * 11 + 3}
thermo 10
"""
    return head + '\n'.join(atoms) + '\n' + '\n'.join(bonds) + tail


# ────────────────────────────────────────────────────────────────────
# Configs
# ────────────────────────────────────────────────────────────────────


CONFIGS = [
    {
        'id': 'polymerize',
        'title': 'Self-Assembly Polymerization',
        'subtitle': 'Free monomers → linear chains via fix bond/create',
        'description': (
            'A box of 500 randomly-placed monomers polymerises into linear '
            'chains and rings. Each atom is capped at two bonds (`iparam 2 1 '
            'jparam 2 1`), so branching is forbidden — the network grows '
            'strictly as 1D filaments. The `cluster_sizes` port directly '
            'reports the chain-length distribution as it evolves from a '
            'monomer soup to a fibrous gel.'
        ),
        'script': build_polymerization_script(),
        'n_snapshots': 30,
        'interval': 0.2,
        'color_scheme': 'indigo',
        'camera': [12.0, 4.5, 12.0],
        'box_center_offset': True,
    },
    {
        'id': 'condensate',
        'title': 'Sticker–Spacer Condensate',
        'subtitle': 'Pre-built chains crosslink into a percolating gel',
        'description': (
            '12 heteropolymer chains, each 12 beads long with the repeating '
            'pattern <strong>sticker (type 1) – spacer – spacer</strong>, are '
            'arranged on a grid. Backbone bonds (type 1) hold each chain '
            'together; sticker–sticker crosslinks (type 2) form dynamically '
            'across chains. Within ~2 simulation-time units the 12 separate '
            'clusters merge into a single percolating condensate — exactly '
            'the use case CASPULE’s `bond/create/random` was designed for.'
        ),
        'script': build_condensate_script(),
        'n_snapshots': 30,
        'interval': 0.3,
        'color_scheme': 'emerald',
        'camera': [22.0, 12.0, 18.0],
        'box_center_offset': True,
    },
    {
        'id': 'associative',
        'title': 'Associative Network Turnover',
        'subtitle': 'fix bond/create + fix bond/break in steady state',
        'description': (
            '48 short bifunctional chains (sticker–spacer³–sticker), packed '
            'densely on a grid. Crosslink bonds (type 2) form stochastically '
            'between stickers within 1.5σ and break stochastically beyond '
            '2.1σ. After the initial gelation transient the crosslink count '
            'plateaus while individual bonds keep turning over — the wrapper '
            'reports both the steady-state count and the per-step '
            'formation/breaking event counts.'
        ),
        'script': build_associative_script(),
        'n_snapshots': 30,
        'interval': 0.4,
        'color_scheme': 'rose',
        'camera': [18.0, 12.0, 15.0],
        'box_center_offset': True,
    },
    {
        'id': 'pruning',
        'title': 'Programmed Atom Removal',
        'subtitle': (
            'Mid-run pruning via the new <code>atoms_to_remove</code> input port'
        ),
        'description': (
            'Same self-assembly polymerisation as experiment 1, but at '
            'snapshot 15 the runner feeds the new <code>atoms_to_remove</code> '
            'input port the IDs of every atom currently in the largest '
            'cluster. The wrapper translates that into '
            '<code>group ... id …; delete_atoms group …; group … delete</code> '
            'and LAMMPS removes the atoms (and their bonds) before the next '
            'integration window. The bonds and clusters time-series shows '
            'a single sharp drop at t = 3.0 — the wrapper-level proof that '
            'the new input port works without needing a Composite.'
        ),
        'script': build_polymerization_script(box=8.0, n_atoms=320, seed=24680),
        'n_snapshots': 30,
        'interval': 0.2,
        'color_scheme': 'amber',
        'camera': [11.0, 4.0, 11.0],
        'box_center_offset': True,
        'prune_at_step': 15,
    },
]


COLOR_SCHEMES = {
    'indigo': {'primary': '#6366f1', 'light': '#e0e7ff', 'dark': '#4338ca'},
    'emerald': {'primary': '#10b981', 'light': '#d1fae5', 'dark': '#059669'},
    'rose': {'primary': '#f43f5e', 'light': '#ffe4e6', 'dark': '#e11d48'},
    'amber': {'primary': '#d97706', 'light': '#fef3c7', 'dark': '#b45309'},
}


# ────────────────────────────────────────────────────────────────────
# Simulation runner
# ────────────────────────────────────────────────────────────────────


def run_simulation(cfg_entry):
    """Run a single simulation; return snapshots and wall-clock runtime.

    Honors the optional ``prune_at_step`` config: at that step we feed
    the wrapper's new ``atoms_to_remove`` input port the IDs of every
    atom currently in the largest cluster, exercising the
    `delete_atoms group ...` path inside ``CASPULEProcess`` directly
    (no Composite required).
    """
    core = allocate_core()
    core.register_link('CASPULEProcess', CASPULEProcess)

    t0 = time.perf_counter()
    proc = CASPULEProcess(config={'input_script': cfg_entry['script']}, core=core)
    state0 = proc.initial_state()

    snapshots = [_snap(0.0, state0)]
    last = state0
    t = 0.0
    prune_step = cfg_entry.get('prune_at_step')
    pruning_event = None
    for step in range(cfg_entry['n_snapshots']):
        atoms_to_remove = []
        if prune_step is not None and step == prune_step:
            atoms_to_remove = _atom_ids_in_largest_cluster(last)
            pruning_event = {
                'step': step,
                'time': round(t + cfg_entry['interval'], 4),
                'count': len(atoms_to_remove),
            }
        result = proc.update(
            {'atoms_to_remove': atoms_to_remove},
            interval=cfg_entry['interval'])
        t += cfg_entry['interval']
        snapshots.append(_snap(round(t, 4), result))
        last = result

    runtime = time.perf_counter() - t0
    proc.close()
    return snapshots, runtime, pruning_event


def _atom_ids_in_largest_cluster(state):
    """Return atom IDs in the largest connected component of the bond graph."""
    bonds = state.get('bonds') or []
    if not bonds:
        return []
    adj = {}
    for triple in bonds:
        _btype, a, b = int(triple[0]), int(triple[1]), int(triple[2])
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    visited = set()
    largest = []
    for start in list(adj.keys()):
        if start in visited:
            continue
        component = []
        stack = [start]
        while stack:
            x = stack.pop()
            if x in visited:
                continue
            visited.add(x)
            component.append(x)
            stack.extend(adj.get(x, []))
        if len(component) > len(largest):
            largest = component
    return sorted(largest)


def _snap(t, s):
    return {
        'time': t,
        'positions': s['positions'],
        'atom_types': s['atom_types'],
        'bonds': s['bonds'],
        'box_dimensions': s['box_dimensions'],
        'num_atoms': s['num_atoms'],
        'num_bonds': s['num_bonds'],
        'bonds_by_type': s['bonds_by_type'],
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
    from bigraph_viz import plot_bigraph

    doc = {
        'caspule': {
            '_type': 'process',
            'address': 'local:CASPULEProcess',
            'config': {'input_script': '<script>'},
            'interval': cfg_entry['interval'],
            'inputs': {
                'atoms_to_remove': ['stores', 'atoms_to_remove'],
            },
            'outputs': {
                'num_bonds': ['stores', 'num_bonds'],
                'bonds_by_type': ['stores', 'bonds_by_type'],
                'formed_bonds': ['stores', 'formed_bonds'],
                'broken_bonds': ['stores', 'broken_bonds'],
                'num_clusters': ['stores', 'num_clusters'],
            },
        },
        'stores': {},
        'emitter': {
            '_type': 'step',
            'address': 'local:ram-emitter',
            'config': {'emit': {
                'num_bonds': 'integer',
                'num_clusters': 'integer',
                'time': 'float',
            }},
            'inputs': {
                'num_bonds': ['stores', 'num_bonds'],
                'num_clusters': ['stores', 'num_clusters'],
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
    """Full PBG document for the JSON tree, with the real LAMMPS script."""
    return make_caspule_document(
        input_script=cfg_entry['script'],
        interval=cfg_entry['interval'],
    )


def script_stats(script):
    """Return a small summary of the script for the UI header."""
    lines = script.splitlines()
    n_total = sum(1 for ln in lines if ln.strip() and not ln.lstrip().startswith('#'))
    n_atoms = sum(1 for ln in lines if ln.lstrip().startswith('create_atoms'))
    n_bonds = sum(1 for ln in lines if ln.lstrip().startswith('create_bonds'))
    n_fix = sum(1 for ln in lines if ln.lstrip().startswith('fix '))
    return {
        'lines': len(lines),
        'effective_lines': n_total,
        'create_atoms': n_atoms,
        'create_bonds': n_bonds,
        'fixes': n_fix,
        'bytes': len(script),
    }


# ────────────────────────────────────────────────────────────────────
# Cluster-size histogram (bin counts per snapshot's final cluster sizes)
# ────────────────────────────────────────────────────────────────────


def cluster_histogram(cluster_sizes, max_bin=None):
    """Bin cluster sizes geometrically — small chains dominate so log-ish bins
    keep all the structure visible without being washed out by singletons."""
    if not cluster_sizes:
        return [], []
    # Count actual sizes 1..max
    largest = max(cluster_sizes)
    if max_bin is None:
        max_bin = largest
    bins = list(range(1, max_bin + 1))
    counts = [cluster_sizes.count(b) for b in bins]
    return bins, counts


# ────────────────────────────────────────────────────────────────────
# HTML rendering
# ────────────────────────────────────────────────────────────────────


pbg_docs = {}


def generate_html(sim_results, output_path):
    sections_html = []
    js_data = {}

    for idx, (cfg, (snapshots, runtime, pruning_event)) in enumerate(sim_results):
        sid = cfg['id']
        cs = COLOR_SCHEMES[cfg['color_scheme']]
        first = snapshots[0]
        last = snapshots[-1]

        times = [s['time'] for s in snapshots]
        n_bonds = [s['num_bonds'] for s in snapshots]
        formed = [s['formed_bonds'] for s in snapshots]
        broken = [s['broken_bonds'] for s in snapshots]
        bond_e = [s['bond_energy'] for s in snapshots]
        pot_e = [s['potential_energy'] for s in snapshots]
        kin_e = [s['kinetic_energy'] for s in snapshots]
        n_clusters = [s['num_clusters'] for s in snapshots]
        largest = [s['largest_cluster'] for s in snapshots]

        # Bond counts split by type — lets us show backbone vs crosslink
        bond_types_seen = sorted({k for s in snapshots for k in s['bonds_by_type']})
        bond_type_series = {
            t: [s['bonds_by_type'].get(t, 0) for s in snapshots]
            for t in bond_types_seen
        }

        hist_bins, hist_counts = cluster_histogram(last['cluster_sizes'])

        # Camera target — center on the simulation box
        box = first['box_dimensions']
        target = [box[0] / 2, box[1] / 2, box[2] / 2]

        js_data[sid] = {
            'snapshots': [{
                'time': s['time'],
                'positions': s['positions'],
                'atom_types': s['atom_types'],
                'bonds': s['bonds'],
            } for s in snapshots],
            'box': box,
            'target': target,
            'camera': cfg['camera'],
            'charts': {
                'times': times,
                'num_bonds': n_bonds,
                'bond_type_series': bond_type_series,
                'formed': formed,
                'broken': broken,
                'bond_energy': bond_e,
                'potential_energy': pot_e,
                'kinetic_energy': kin_e,
                'num_clusters': n_clusters,
                'largest': largest,
                'hist_bins': hist_bins,
                'hist_counts': hist_counts,
            },
            'palette': {
                'primary': cs['primary'],
                'dark': cs['dark'],
            },
        }

        print(f'  Generating bigraph diagram for {sid}...')
        bigraph_img = generate_bigraph_image(cfg)
        pbg_docs[sid] = build_pbg_document(cfg)

        bonds0 = first['num_bonds']
        bonds1 = last['num_bonds']
        formed_total = sum(formed)
        broken_total = sum(broken)

        # Crosslink count summary (bond type 2 if present)
        if '2' in last['bonds_by_type']:
            crosslink_label = 'Crosslinks (type 2)'
            crosslink_value = last['bonds_by_type']['2']
        else:
            crosslink_label = 'Total bonds'
            crosslink_value = bonds1

        # LAMMPS input script — escaped for HTML, displayed as <pre>
        stats = script_stats(cfg['script'])
        import html as _html
        escaped_script = _html.escape(cfg['script'].strip('\n'))

        prune_metric = ''
        if pruning_event:
            prune_metric = (
                f'<div class="metric"><span class="metric-label">Pruned</span>'
                f'<span class="metric-value">{pruning_event["count"]}</span>'
                f'<span class="metric-sub">at t = {pruning_event["time"]}</span>'
                f'</div>'
            )

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
        <div class="metric"><span class="metric-label">Atoms (start)</span><span class="metric-value">{first['num_atoms']:,}</span></div>
        <div class="metric"><span class="metric-label">Atoms (end)</span><span class="metric-value">{last['num_atoms']:,}</span></div>
        <div class="metric"><span class="metric-label">Bonds (start &rarr; end)</span><span class="metric-value">{bonds0} &rarr; {bonds1}</span></div>
        <div class="metric"><span class="metric-label">{crosslink_label}</span><span class="metric-value">{crosslink_value}</span></div>
        <div class="metric"><span class="metric-label">Formed (total)</span><span class="metric-value">{formed_total}</span></div>
        <div class="metric"><span class="metric-label">Broken (total)</span><span class="metric-value">{broken_total}</span></div>
        <div class="metric"><span class="metric-label">Final clusters</span><span class="metric-value">{last['num_clusters']}</span><span class="metric-sub">largest = {last['largest_cluster']}</span></div>
        {prune_metric}
        <div class="metric"><span class="metric-label">Snapshots</span><span class="metric-value">{len(snapshots)}</span></div>
        <div class="metric"><span class="metric-label">Runtime</span><span class="metric-value">{runtime:.2f}s</span></div>
      </div>

      <h3 class="subsection-title">LAMMPS Input Script</h3>
      <details class="script-wrap" style="border-left: 3px solid {cs['primary']};">
        <summary class="script-summary">
          <span class="script-summary-label">Click to expand the full script the wrapper fed to LAMMPS</span>
          <span class="script-summary-stats">
            {stats['lines']} lines &middot;
            {stats['create_atoms']} <code>create_atoms</code> &middot;
            {stats['create_bonds']} <code>create_bonds</code> &middot;
            {stats['fixes']} <code>fix</code>
          </span>
        </summary>
        <pre class="script-pre"><code id="script-{sid}">{escaped_script}</code></pre>
      </details>

      <h3 class="subsection-title">3D Bond Network Viewer</h3>
      <div class="viewer-wrap">
        <canvas id="canvas-{sid}" class="mesh-canvas"></canvas>
        <div class="viewer-info" id="legend-{sid}">
          <span class="atom-dot" style="background:#f97316;"></span>Sticker (type 1)<br>
          <span class="atom-dot" style="background:#3b82f6;"></span>Spacer (type 2)<br>
          <span class="bond-dash" style="background:#1e293b;"></span>Backbone<br>
          <span class="bond-dash" style="background:#dc2626;"></span>Crosslink<br>
          <span style="color:#94a3b8; font-size:.7rem;">Drag to rotate &middot; Scroll to zoom</span>
        </div>
        <div class="slider-controls">
          <button class="play-btn" style="border-color:{cs['primary']}; color:{cs['primary']};" onclick="togglePlay('{sid}')">Play</button>
          <label>Time</label>
          <input type="range" class="time-slider" id="slider-{sid}" min="0" max="{len(snapshots)-1}" value="0" step="1"
                 style="accent-color:{cs['primary']};">
          <span class="time-val" id="tval-{sid}">t = 0</span>
        </div>
      </div>

      <h3 class="subsection-title">Bond Network &amp; Cluster Dynamics</h3>
      <div class="charts-row">
        <div class="chart-box"><div id="chart-bonds-{sid}" class="chart"></div></div>
        <div class="chart-box"><div id="chart-events-{sid}" class="chart"></div></div>
        <div class="chart-box"><div id="chart-clusters-{sid}" class="chart"></div></div>
        <div class="chart-box"><div id="chart-hist-{sid}" class="chart"></div></div>
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
.page-header p { color:#64748b; font-size:.95rem; max-width:820px; }
.page-header code { background:#e2e8f0; padding:0 .25em; border-radius:4px;
                    font-family:'SF Mono',Menlo,monospace; font-size:.85em; }
.nav { display:flex; gap:.8rem; padding:1rem 3rem; background:#f8fafc;
        border-bottom:1px solid #e2e8f0; position:sticky; top:0; z-index:100; flex-wrap:wrap; }
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
.sim-description { color:#475569; font-size:.9rem; margin-bottom:1.5rem; max-width:840px; }
.subsection-title { font-size:1.05rem; font-weight:600; color:#334155;
                     margin:1.5rem 0 .8rem; }
.metrics-row { display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr));
                gap:.8rem; margin-bottom:1.5rem; }
.metric { background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px;
           padding:.8rem; text-align:center; }
.metric-label { display:block; font-size:.7rem; text-transform:uppercase;
                 letter-spacing:.06em; color:#94a3b8; margin-bottom:.2rem; }
.metric-value { display:block; font-size:1.25rem; font-weight:700; color:#1e293b; }
.metric-sub { display:block; font-size:.7rem; color:#94a3b8; }
.viewer-wrap { position:relative; background:#0f172a; border:1px solid #1e293b;
                border-radius:14px; overflow:hidden; margin-bottom:1rem; }
.mesh-canvas { width:100%; height:520px; display:block; cursor:grab; }
.mesh-canvas:active { cursor:grabbing; }
.viewer-info { position:absolute; top:.8rem; left:.8rem; background:rgba(15,23,42,.85);
                border:1px solid #1e293b; border-radius:8px; padding:.6rem .8rem;
                font-size:.75rem; color:#cbd5e1; backdrop-filter:blur(6px); line-height:1.7; }
.atom-dot { display:inline-block; width:9px; height:9px; border-radius:50%;
             margin-right:.4rem; vertical-align:middle; }
.bond-dash { display:inline-block; width:14px; height:3px; border-radius:2px;
              margin-right:.4rem; vertical-align:middle; }
.slider-controls { position:absolute; bottom:0; left:0; right:0;
                    background:linear-gradient(transparent,rgba(15,23,42,.97));
                    padding:1.5rem 1.5rem 1rem; display:flex; align-items:center;
                    gap:.8rem; color:#cbd5e1; }
.slider-controls label { font-size:.8rem; color:#94a3b8; }
.time-slider { flex:1; height:5px; }
.time-val { font-size:.9rem; font-weight:600; color:#f1f5f9; min-width:200px; text-align:right; }
.play-btn { background:rgba(15,23,42,.6); border:1.5px solid; padding:.3rem .8rem;
             border-radius:7px; cursor:pointer; font-size:.8rem; font-weight:600;
             transition:all .15s; }
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
.script-wrap { background:#0f172a; border:1px solid #1e293b; border-radius:10px;
               margin-bottom:1rem; overflow:hidden; }
.script-wrap summary { padding:.7rem 1rem; cursor:pointer; user-select:none;
                       display:flex; flex-wrap:wrap; gap:.6rem; align-items:center;
                       justify-content:space-between; color:#e2e8f0; }
.script-wrap summary:hover { background:#1e293b; }
.script-wrap summary::marker { color:#64748b; }
.script-summary-label { font-size:.85rem; font-weight:600; }
.script-summary-stats { font-size:.72rem; color:#94a3b8; font-family:'SF Mono',Menlo,monospace; }
.script-summary-stats code { background:#1e293b; padding:0 .25em; border-radius:3px;
                              color:#cbd5e1; font-size:.95em; }
.script-pre { background:#020617; color:#e2e8f0; padding:1rem 1.2rem;
              max-height:420px; overflow:auto; margin:0;
              font-family:'SF Mono',Menlo,Monaco,'Courier New',monospace;
              font-size:.75rem; line-height:1.55; border-top:1px solid #1e293b; }
.script-pre code { font-family:inherit; }
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
  adds <code>fix bond/create/random</code> on top of LAMMPS's built-in
  <code>fix bond/create</code> and <code>fix bond/break</code> — as a
  process-bigraph Process. The wrapper queries the live bond list at every
  update step and reports it through dedicated PBG ports: the bond list,
  per-type counts, formed/broken event counts, bond energy, and
  connected-component cluster sizes. It also accepts an
  <code>atoms_to_remove</code> input port: any list of LAMMPS atom IDs
  written to that store before an update is deleted (with their bonds)
  via <code>group … id …; delete_atoms group …</code> in the next
  integration window. The four experiments below show polymer
  self-assembly, sticker-spacer condensation, steady-state associative
  bond turnover, and a programmed mid-run pruning event that exercises
  the new input port.</p>
</div>

<div class="nav">__NAV_ITEMS__</div>

__SECTIONS__

<div class="footer">
  Generated by <strong>pbg-caspule</strong> &mdash;
  CASPULE / LAMMPS + process-bigraph &mdash;
  Sticker-spacer polymer dynamics with first-class bond ports
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

const ATOM_COLORS = { 1: 0xf97316, 2: 0x3b82f6, 3: 0x10b981, default: 0x64748b };
const BOND_COLORS = { 1: 0xe2e8f0, 2: 0xef4444, 3: 0xfbbf24, default: 0xf1f5f9 };

function initViewer(sid) {
  const d = DATA[sid];
  const canvas = document.getElementById('canvas-' + sid);
  const W = canvas.parentElement.clientWidth;
  const H = 520;

  const renderer = new THREE.WebGLRenderer({canvas, antialias:true});
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(W, H);
  renderer.setClearColor(0x0f172a);

  const scene = new THREE.Scene();
  scene.fog = new THREE.Fog(0x0f172a, 25, 80);

  const cam = new THREE.PerspectiveCamera(40, W/H, 0.01, 500);
  cam.position.set(d.camera[0], d.camera[1], d.camera[2]);

  const controls = new THREE.OrbitControls(cam, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.autoRotate = true;
  controls.autoRotateSpeed = 0.5;
  controls.target.set(d.target[0], d.target[1], d.target[2]);

  scene.add(new THREE.AmbientLight(0xffffff, 0.55));
  const dl1 = new THREE.DirectionalLight(0xffffff, 0.8);
  dl1.position.set(20, 25, 15); scene.add(dl1);
  const dl2 = new THREE.DirectionalLight(0x93c5fd, 0.4);
  dl2.position.set(-12, -8, -10); scene.add(dl2);

  // Box wireframe for context
  const box = d.box;
  const boxGeom = new THREE.BoxGeometry(box[0], box[1], box[2]);
  const boxEdges = new THREE.EdgesGeometry(boxGeom);
  const boxMat = new THREE.LineBasicMaterial({color:0x334155, transparent:true, opacity:0.4});
  const boxLines = new THREE.LineSegments(boxEdges, boxMat);
  boxLines.position.set(box[0]/2, box[1]/2, box[2]/2);
  scene.add(boxLines);

  // Atoms via instanced spheres
  const snap0 = d.snapshots[0];
  const N = snap0.positions.length;
  const sphereGeom = new THREE.SphereGeometry(0.34, 14, 14);
  const sphereMat = new THREE.MeshPhongMaterial({ shininess: 60, vertexColors: true });
  const inst = new THREE.InstancedMesh(sphereGeom, sphereMat, N);
  const colorAttr = new Float32Array(N * 3);
  inst.instanceColor = new THREE.InstancedBufferAttribute(colorAttr, 3);
  scene.add(inst);

  // Per-type bond LineSegments — one geometry per bond type so we can
  // color them independently via separate materials
  const bondLayers = {};

  function getBondLayer(bondType) {
    if (bondLayers[bondType]) return bondLayers[bondType];
    const geom = new THREE.BufferGeometry();
    const color = BOND_COLORS[bondType] !== undefined
      ? BOND_COLORS[bondType] : BOND_COLORS.default;
    const mat = new THREE.LineBasicMaterial({
      color: color,
      transparent: true,
      opacity: bondType === 1 ? 0.6 : 0.95,
      linewidth: 2,
    });
    const lines = new THREE.LineSegments(geom, mat);
    scene.add(lines);
    bondLayers[bondType] = { geom, lines };
    return bondLayers[bondType];
  }

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

    // Group bonds by type
    const byType = {};
    for (let k = 0; k < snap.bonds.length; k++) {
      const b = snap.bonds[k];
      const bt = b[0];
      if (!byType[bt]) byType[bt] = [];
      byType[bt].push([b[1] - 1, b[2] - 1]);
    }

    // Update each layer's segment buffer
    for (const layerType in bondLayers) {
      bondLayers[layerType].geom.setAttribute(
        'position', new THREE.BufferAttribute(new Float32Array(0), 3));
    }
    for (const bt in byType) {
      const layer = getBondLayer(parseInt(bt));
      const pairs = byType[bt];
      const seg = new Float32Array(pairs.length * 6);
      for (let k = 0; k < pairs.length; k++) {
        const a = pairs[k][0], b = pairs[k][1];
        if (a < 0 || b < 0 || a >= N || b >= N) continue;
        const pa = snap.positions[a], pb = snap.positions[b];
        seg[k*6]   = pa[0]; seg[k*6+1] = pa[1]; seg[k*6+2] = pa[2];
        seg[k*6+3] = pb[0]; seg[k*6+4] = pb[1]; seg[k*6+5] = pb[2];
      }
      layer.geom.setAttribute('position', new THREE.BufferAttribute(seg, 3));
      layer.geom.computeBoundingSphere();
    }
  }

  updateSnapshot(0);

  const slider = document.getElementById('slider-' + sid);
  const tval = document.getElementById('tval-' + sid);
  function setLabel(idx) {
    const s = d.snapshots[idx];
    tval.textContent = 't=' + s.time + '  bonds=' + s.bonds.length;
  }
  slider.addEventListener('input', () => {
    const idx = parseInt(slider.value);
    updateSnapshot(idx);
    setLabel(idx);
  });
  setLabel(0);

  viewers[sid] = { renderer, scene, cam, controls, updateSnapshot, slider, setLabel };
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
      v.setLabel(idx);
    }, 240);
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

const BOND_TYPE_COLORS = { '1':'#475569', '2':'#dc2626', '3':'#f59e0b' };
const BOND_TYPE_LABEL = { '1':'Type 1 (backbone)', '2':'Type 2 (crosslink)', '3':'Type 3' };

Object.keys(DATA).forEach(sid => {
  const c = DATA[sid].charts;
  const palette = DATA[sid].palette;

  // Bonds by type — stacked-area per bond type
  const bondTraces = [];
  Object.keys(c.bond_type_series).forEach(bt => {
    bondTraces.push({
      x: c.times, y: c.bond_type_series[bt], type:'scatter', mode:'lines',
      stackgroup:'one', name: BOND_TYPE_LABEL[bt] || ('Type ' + bt),
      line:{ color: BOND_TYPE_COLORS[bt] || '#94a3b8', width:1.5 },
      fillcolor: BOND_TYPE_COLORS[bt] || '#94a3b8',
    });
  });
  // If only one bond type, lighter fill so the line is visible
  if (bondTraces.length === 1) bondTraces[0].opacity = 0.6;

  Plotly.newPlot('chart-bonds-'+sid, bondTraces, Object.assign({}, pLayout, {
    title:{ text:'Bonds by type', font:{ size:12, color:'#334155' } },
    xaxis: Object.assign({}, pLayout.xaxis, { title:{ text:'Time', font:{ size:10 } } }),
    yaxis: Object.assign({}, pLayout.yaxis, { title:{ text:'Bonds', font:{ size:10 } } }),
    legend:{ font:{ size:9 }, bgcolor:'rgba(0,0,0,0)' }, showlegend:bondTraces.length > 1,
  }), pCfg);

  // Bond formation/breaking events
  Plotly.newPlot('chart-events-'+sid, [
    { x:c.times, y:c.formed, type:'bar', name:'Formed',
      marker:{ color:'#10b981' } },
    { x:c.times, y:c.broken.map(v => -v), type:'bar', name:'Broken',
      marker:{ color:'#ef4444' } },
  ], Object.assign({}, pLayout, {
    title:{ text:'Per-step bond events (formed vs broken)', font:{ size:12, color:'#334155' } },
    xaxis: Object.assign({}, pLayout.xaxis, { title:{ text:'Time', font:{ size:10 } } }),
    yaxis: Object.assign({}, pLayout.yaxis, { title:{ text:'Δ bonds', font:{ size:10 } } }),
    barmode:'relative',
    legend:{ font:{ size:9 }, bgcolor:'rgba(0,0,0,0)' }, showlegend:true,
  }), pCfg);

  // Cluster connectivity (dual y-axis: # clusters and largest size)
  Plotly.newPlot('chart-clusters-'+sid, [
    { x:c.times, y:c.num_clusters, type:'scatter', mode:'lines',
      line:{ color:'#8b5cf6', width:2 }, name:'# clusters', yaxis:'y' },
    { x:c.times, y:c.largest, type:'scatter', mode:'lines',
      line:{ color:'#f59e0b', width:2 }, name:'Largest', yaxis:'y2' },
  ], Object.assign({}, pLayout, {
    title:{ text:'Cluster connectivity', font:{ size:12, color:'#334155' } },
    xaxis: Object.assign({}, pLayout.xaxis, { title:{ text:'Time', font:{ size:10 } } }),
    yaxis: Object.assign({}, pLayout.yaxis, {
      title:{ text:'# clusters', font:{ size:10, color:'#8b5cf6' } } }),
    yaxis2:{ title:{ text:'Largest cluster size', font:{ size:10, color:'#f59e0b' } },
             gridcolor:'rgba(0,0,0,0)', overlaying:'y', side:'right' },
    legend:{ font:{ size:9 }, bgcolor:'rgba(0,0,0,0)' }, showlegend:true,
  }), pCfg);

  // Final cluster-size distribution histogram
  Plotly.newPlot('chart-hist-'+sid, [{
    x: c.hist_bins, y: c.hist_counts, type:'bar',
    marker:{ color: palette.primary, line:{ color: palette.dark, width: 0.5 } },
  }], Object.assign({}, pLayout, {
    title:{ text:'Final cluster-size distribution', font:{ size:12, color:'#334155' } },
    xaxis: Object.assign({}, pLayout.xaxis, {
      title:{ text:'Cluster size (atoms)', font:{ size:10 } } }),
    yaxis: Object.assign({}, pLayout.yaxis, {
      title:{ text:'# clusters', font:{ size:10 } } }),
    showlegend:false,
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
        snapshots, runtime, pruning_event = run_simulation(cfg)
        sim_results.append((cfg, (snapshots, runtime, pruning_event)))
        last = snapshots[-1]
        bt = last['bonds_by_type']
        extra = ''
        if pruning_event:
            extra = (f' | pruned {pruning_event["count"]} atoms at '
                     f't={pruning_event["time"]}')
        print(f'  Runtime: {runtime:.2f}s | snapshots: {len(snapshots)} | '
              f'final bonds: {last["num_bonds"]} (by type: {bt}) | '
              f'clusters: {last["num_clusters"]}, largest: '
              f'{last["largest_cluster"]}{extra}')

    print('Generating HTML report...')
    generate_html(sim_results, output_path)

    import subprocess
    subprocess.run(['open', '-a', 'Safari', output_path])


if __name__ == '__main__':
    run_demo()
