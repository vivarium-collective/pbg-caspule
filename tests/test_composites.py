"""Integration tests for the CASPULE composite document factory."""

import pytest
from process_bigraph import Composite, gather_emitter_results
from pbg_caspule.composites import (
    register_caspule, make_caspule_document, ALL_PORTS, SCALAR_EMIT_TYPES,
)


SHORT_BOND_SCRIPT = """
units lj
atom_style bond
boundary p p p
neighbor 0.4 bin
neigh_modify every 1 delay 0
region box block 0 8 0 8 0 8
create_box 1 box bond/types 1 extra/bond/per/atom 4
mass 1 1.0
create_atoms 1 single 1.0 4.0 4.0
create_atoms 1 single 2.0 4.0 4.0
create_atoms 1 single 3.0 4.0 4.0
create_bonds single/bond 1 1 2
create_bonds single/bond 1 2 3
pair_style lj/cut 1.5
pair_coeff * * 1.0 1.0 1.5
bond_style harmonic
bond_coeff 1 100.0 1.0
fix nve all nve
fix lan all langevin 0.4 0.4 1.0 12345
timestep 0.005
thermo 1
"""


@pytest.fixture
def core():
    return register_caspule()


def test_make_document_requires_input():
    with pytest.raises(ValueError):
        make_caspule_document()


def test_composite_assembly(core):
    doc = make_caspule_document(input_script=SHORT_BOND_SCRIPT, interval=0.05)
    sim = Composite({'state': doc}, core=core)
    assert sim is not None


def test_composite_short_run_populates_stores(core):
    doc = make_caspule_document(input_script=SHORT_BOND_SCRIPT, interval=0.05)
    sim = Composite({'state': doc}, core=core)
    sim.run(0.1)
    stores = sim.state['stores']
    assert stores['num_atoms'] == 3
    assert stores['num_bonds'] == 2
    assert stores['num_clusters'] == 1
    assert stores['largest_cluster'] == 3
    assert isinstance(stores['total_energy'], float)


def test_emitter_collects_scalar_timeseries(core):
    doc = make_caspule_document(input_script=SHORT_BOND_SCRIPT, interval=0.05)
    sim = Composite({'state': doc}, core=core)
    sim.run(0.2)
    raw = gather_emitter_results(sim)
    series = raw[('emitter',)]
    # We get >=2 emits; first one fires before the first process update,
    # so look at the last emit for populated state.
    last = series[-1]
    assert last['num_bonds'] == 2
    assert last['num_clusters'] == 1
    assert last['largest_cluster'] == 3
    # Every scalar port we declared should be present in each emit
    for port in SCALAR_EMIT_TYPES:
        assert port in last


def test_all_ports_wired(core):
    doc = make_caspule_document(input_script=SHORT_BOND_SCRIPT, interval=0.05)
    outputs = doc['caspule']['outputs']
    for port in ALL_PORTS:
        assert port in outputs


def test_emit_scalars_false_skips_emitter():
    doc = make_caspule_document(
        input_script=SHORT_BOND_SCRIPT, interval=0.05, emit_scalars=False)
    assert 'emitter' not in doc
