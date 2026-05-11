"""CASPULE composite documents + composite-spec discovery.

Two flavors of composite construction live in this package:

1. **Hand-coded factories** — `make_caspule_document(input_script=…)` builds a
   PBG state-dict programmatically for callers that want full control over
   the LAMMPS script + wiring. Used by `demo/demo_report.py` for the four
   bond-network experiments.

2. **Declarative `*.composite.yaml`** — sibling files in this directory follow
   the pbg-superpowers composite-spec convention. `build_composite()` loads
   one by name and instantiates `process_bigraph.Composite` with parameter
   substitution. The dashboard's composite explorer discovers these
   automatically once the package is installed in a workspace.

Both flavors are equivalent — pick the one that fits your use case.
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Any

import yaml
from process_bigraph import allocate_core
from process_bigraph.emitter import RAMEmitter

from pbg_caspule.processes import CASPULEProcess


# ---------------------------------------------------------------------------
# Hand-coded composite factories (legacy / programmatic API)
# ---------------------------------------------------------------------------

# Process output ports.
ALL_PORTS = (
    'temperature', 'potential_energy', 'kinetic_energy', 'total_energy',
    'pressure', 'volume', 'box_dimensions',
    'num_atoms', 'positions', 'velocities', 'atom_types',
    'num_bonds', 'bonds', 'bonds_by_type', 'bond_energy',
    'formed_bonds', 'broken_bonds',
    'num_clusters', 'largest_cluster', 'cluster_sizes',
)

# Scalar ports — these are safe to wire into the RAM emitter, which
# expects concrete typed slots. Larger structured outputs (positions,
# bonds, cluster_sizes) flow through stores but aren't emitted by
# default, since the emitter wants typed scalar values.
SCALAR_EMIT_TYPES = {
    'temperature': 'float',
    'potential_energy': 'float',
    'kinetic_energy': 'float',
    'total_energy': 'float',
    'pressure': 'float',
    'volume': 'float',
    'bond_energy': 'float',
    'num_atoms': 'integer',
    'num_bonds': 'integer',
    'formed_bonds': 'integer',
    'broken_bonds': 'integer',
    'num_clusters': 'integer',
    'largest_cluster': 'integer',
}


def register_caspule(core=None):
    """Return a core with CASPULEProcess and the RAM emitter registered."""
    if core is None:
        core = allocate_core()
    core.register_link('CASPULEProcess', CASPULEProcess)
    core.register_link('ram-emitter', RAMEmitter)
    return core


def make_caspule_document(input_script='', input_file='',
                          working_directory='', interval=10.0,
                          emit_scalars=True):
    """Build a Composite document wiring CASPULEProcess to a RAM emitter.

    Returns a state-dict suitable for `Composite({'state': doc}, core=...)`.

    Provide either `input_script` (inline LAMMPS/CASPULE script) or
    `input_file` (path to a `.in` file). `run` / `rerun` commands are
    stripped at process startup — the orchestrator drives integration
    via the supplied `interval`.

    By default the emitter records the scalar thermo + bond ports
    (`SCALAR_EMIT_TYPES`). Set `emit_scalars=False` to skip the
    emitter and just leave outputs in the stores.
    """
    if not input_script and not input_file:
        raise ValueError(
            'make_caspule_document requires input_script or input_file')

    config = {
        'input_file': input_file,
        'input_script': input_script,
        'working_directory': working_directory,
    }

    process_outputs = {p: ['stores', p] for p in ALL_PORTS}

    doc = {
        'caspule': {
            '_type': 'process',
            'address': 'local:CASPULEProcess',
            'config': config,
            'interval': interval,
            'inputs': {},
            'outputs': process_outputs,
        },
        'stores': {},
    }

    if emit_scalars:
        emit_schema = dict(SCALAR_EMIT_TYPES)
        emit_schema['time'] = 'float'
        emitter_inputs = {p: ['stores', p] for p in SCALAR_EMIT_TYPES}
        emitter_inputs['time'] = ['global_time']
        doc['emitter'] = {
            '_type': 'step',
            'address': 'local:ram-emitter',
            'config': {'emit': emit_schema},
            'inputs': emitter_inputs,
        }

    return doc


# ---------------------------------------------------------------------------
# Declarative composite-spec loader (*.composite.yaml)
# ---------------------------------------------------------------------------

_COMPOSITES_DIR = Path(__file__).parent

_FULL_PLACEHOLDER = re.compile(r"^\$\{([a-zA-Z_][a-zA-Z0-9_]*)\}$")
_INLINE_PLACEHOLDER = re.compile(r"\$\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def _cast(value: Any, declared_type: str | None) -> Any:
    if declared_type is None:
        return value
    if declared_type == "float":
        return float(value)
    if declared_type == "int":
        return int(value)
    if declared_type in ("string", "str"):
        return str(value)
    if declared_type == "bool":
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes")
        return bool(value)
    return value


def _substitute(state: Any, params: dict, overrides: dict) -> Any:
    if isinstance(state, dict):
        return {k: _substitute(v, params, overrides) for k, v in state.items()}
    if isinstance(state, list):
        return [_substitute(v, params, overrides) for v in state]
    if isinstance(state, str):
        m = _FULL_PLACEHOLDER.match(state)
        if m:
            pname = m.group(1)
            pdef = params.get(pname, {})
            raw = overrides.get(pname, pdef.get("default"))
            return _cast(raw, pdef.get("type"))
        if _INLINE_PLACEHOLDER.search(state):
            return _INLINE_PLACEHOLDER.sub(
                lambda mm: str(overrides.get(mm.group(1), params.get(mm.group(1), {}).get("default", ""))),
                state,
            )
    return state


def list_composite_specs() -> list[str]:
    """Return short names of every `*.composite.yaml` shipped in this package."""
    out: list[str] = []
    for path in sorted(_COMPOSITES_DIR.glob("*.composite.yaml")):
        out.append(path.name[: -len(".composite.yaml")])
    return out


def load_composite_spec(name: str) -> dict:
    """Load and parse a named composite spec. `name` is the stem (no suffix)."""
    path = _COMPOSITES_DIR / f"{name}.composite.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"composite spec not found: {path}")
    return yaml.safe_load(path.read_text())


def build_composite(name: str, *, overrides: dict | None = None, core=None):
    """Load a *.composite.yaml by name and instantiate process_bigraph.Composite.

    overrides: parameter overrides (keys must match spec.parameters)
    core:      optional pre-built core; otherwise register_caspule() is used
    """
    from process_bigraph import Composite

    spec = load_composite_spec(name)
    if not isinstance(spec, dict) or "state" not in spec or "name" not in spec:
        raise ValueError(f"composite '{name}' missing required keys (name, state)")

    if core is None:
        core = register_caspule()

    params = spec.get("parameters") or {}
    state = _substitute(spec.get("state") or {}, params, overrides or {})
    return Composite({"state": state}, core=core)
