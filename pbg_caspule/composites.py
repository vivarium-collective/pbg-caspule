"""Pre-built composite document factory for CASPULE simulations."""

from process_bigraph import allocate_core
from process_bigraph.emitter import RAMEmitter

from pbg_caspule.processes import CASPULEProcess


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
