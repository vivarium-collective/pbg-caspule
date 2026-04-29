# pbg-caspule

A [process-bigraph](https://github.com/vivarium-collective/process-bigraph)
wrapper for [CASPULE](https://caspule.github.io/caspule/) — a modified
LAMMPS that adds dynamic bond formation/breaking via
`fix bond/create/random` (random-partner bond formation) on top of
LAMMPS' built-in `fix bond/create` and `fix bond/break`.

The wrapper drives any bond-aware LAMMPS or CASPULE build forward in
time as a process-bigraph `Process`, and exposes the **live bond
network** through dedicated PBG ports — bond list, formed/broken event
counts per step, per-type bond counts, bond energy, and connected-
component cluster sizes — alongside the standard thermodynamic state.

## What it does

```
                ┌──────────────────────────────────────────────┐
                │  CASPULEProcess  (wraps lammps Python lib)   │
                │                                              │
   interval ──▶ │  push run N command  →  read state via       │
                │  numpy/extract_atom + gather_bonds           │
                └──────────────┬───────────────────────────────┘
                               │
                  ports        ▼
       ┌─────────────────────────────────────────────────────┐
       │ thermo:    temperature, potential_energy, ...       │
       │ atoms:     positions, velocities, atom_types        │
       │ bonds:     num_bonds, bonds, bonds_by_type          │
       │ dynamics:  formed_bonds, broken_bonds (per step)    │
       │ topology:  num_clusters, largest_cluster, sizes     │
       └─────────────────────────────────────────────────────┘
```

The wrapper is identical for stock LAMMPS and the CASPULE-patched
build — the patched build adds the `bond/create/random` fix, but the
live-bond extraction goes through LAMMPS' standard `gather_bonds()`
API regardless.

## Installation

```bash
git clone <this-repo>
cd pbg-caspule
uv venv .venv
source .venv/bin/activate
uv pip install -e .
# Optional: dev tools (tests + demo)
uv pip install -e ".[dev]"
```

The `lammps` Python wheel is pulled in automatically. On macOS you
may also need `mpich` (run `uv pip install mpich`) to satisfy the
wheel's `libmpi` rpath.

For the **CASPULE-patched LAMMPS** (with `bond/create/random` enabled),
follow the upstream installer:

```bash
curl -O https://raw.githubusercontent.com/caspule/caspule/main/install_caspule_requirements.sh
bash install_caspule_requirements.sh
```

The patched build is API-compatible with the wrapper — point your
script at `bond/create/random` instead of `bond/create` and the same
PBG ports keep working.

## Quick start

```python
from process_bigraph import Composite, gather_emitter_results
from pbg_caspule.composites import register_caspule, make_caspule_document

# Sticker (type 1) + spacer (type 2) atoms, dynamic bond formation
script = """
units lj
atom_style bond
boundary p p p
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

core = register_caspule()
doc = make_caspule_document(input_script=script, interval=0.05)
sim = Composite({'state': doc}, core=core)
sim.run(0.5)

results = gather_emitter_results(sim)[('emitter',)]
for emit in results:
    print(f"t={emit['time']:.3f}  bonds={emit['num_bonds']}  "
          f"formed={emit['formed_bonds']}  clusters={emit['num_clusters']}")
```

## API reference

### `pbg_caspule.processes.CASPULEProcess`

Process subclass driven by an interval. Build it with either an inline
script or a path to a `.in` file. `run` / `rerun` commands are stripped
at startup — the wrapper drives integration via `update(state, interval)`.

| Config key            | Type      | Default | Description |
|-----------------------|-----------|---------|-------------|
| `input_file`          | string    | `''`    | Path to a LAMMPS / CASPULE `.in` file |
| `input_script`        | string    | `''`    | Inline LAMMPS / CASPULE script |
| `working_directory`   | string    | `''`    | Resolves relative paths in `read_data` etc. |
| `cluster_max_atoms`   | integer   | 200000  | Skip cluster computation when natoms exceeds this |

#### Output ports

Thermodynamic + structural state (`overwrite[...]` = absolute, not delta):

| Port              | Type             | Notes |
|-------------------|------------------|-------|
| `temperature`     | `float`          | LAMMPS thermo `temp` |
| `potential_energy`| `float`          | thermo `pe` |
| `kinetic_energy`  | `float`          | thermo `ke` |
| `total_energy`    | `float`          | thermo `etotal` |
| `pressure`        | `float`          | thermo `press` |
| `volume`          | `float`          | thermo `vol` |
| `box_dimensions`  | `list`           | `[lx, ly, lz]` |
| `num_atoms`       | `integer`        | global `natoms` |
| `positions`       | `list`           | per-atom `[x, y, z]` |
| `velocities`      | `list`           | per-atom `[vx, vy, vz]` |
| `atom_types`      | `list`           | per-atom integer type |
| `num_bonds`       | `integer`        | global bond count |
| `bonds`           | `list`           | `[type, atom_lo, atom_hi]` triples (canonicalised) |
| `bonds_by_type`   | `map[integer]`   | string-keyed counts (`{'1': 47, '2': 3}`) |
| `bond_energy`     | `float`          | thermo `ebond` |
| `formed_bonds`    | `integer`        | bonds added since previous update |
| `broken_bonds`    | `integer`        | bonds removed since previous update |
| `num_clusters`    | `integer`        | connected components in bond graph |
| `largest_cluster` | `integer`        | size of largest component |
| `cluster_sizes`   | `list`           | sorted descending list of all component sizes |

### `pbg_caspule.composites.make_caspule_document`

Returns a Composite document with `CASPULEProcess` wired to a `RAMEmitter`
that records the scalar thermo + bond ports. Pass to `Composite({'state': doc}, core=core)`.

### `pbg_caspule.composites.register_caspule(core=None)`

Returns a freshly-allocated core with `CASPULEProcess` and `ram-emitter`
registered. Pass an existing core to register onto it.

## Architecture

The wrapper instantiates a `lammps.lammps` process per
`CASPULEProcess`, feeds it the user's input script (with `run` /
`rerun` lines stripped), and on each `update()`:

1. Issues `run N pre no post no` for `N = round(interval / dt)`.
2. Reads thermo via `get_thermo()`, atoms via `numpy.extract_atom()`,
   and the global bond list via `gather_bonds()`.
3. Diffs the canonical bond set against the previous step to compute
   `formed_bonds` and `broken_bonds`.
4. Runs a small union-find over the bond list to produce
   `num_clusters`, `largest_cluster`, and `cluster_sizes`.

Because LAMMPS' Python module is the same regardless of which
`bond/create*` fix is in use, the wrapper works without modification
on stock LAMMPS, the CASPULE-patched LAMMPS, or any other custom
LAMMPS build with bond support.

## Demo

```bash
source .venv/bin/activate
uv pip install bigraph-viz matplotlib
python demo/demo_report.py
```

This runs three configurations — a 50-bead polymer chain, dynamic
sticker/spacer crosslinking, and a `bond/create` + `bond/break`
equilibrium — and writes a single self-contained `demo/report.html`
with interactive Three.js bond-network viewers, Plotly time-series
charts, colored bigraph-viz architecture diagrams, and a collapsible
JSON tree of each composite document. The report opens in Safari
automatically.

## Tests

```bash
source .venv/bin/activate
python -m pytest tests/ -q
```

The test suite verifies:

- Process instantiation and config validation
- Initial state correctness on static and dynamic systems
- Bond list canonicalisation and connected-component analysis
- Filtering of `run` / `rerun` directives from user scripts
- End-to-end Composite assembly and emitter timeseries
- Dynamic bond formation via `fix bond/create`
