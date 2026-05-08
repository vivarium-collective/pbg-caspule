"""Unit tests for CASPULEProcess."""

import pytest
from process_bigraph import allocate_core
from pbg_caspule.processes import CASPULEProcess


# Minimal 5-atom linear chain with 4 harmonic bonds. No dynamics
# fixes that would form/break bonds, so the bond list is static and
# the connectivity result is deterministic.
STATIC_CHAIN_SCRIPT = """
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
create_atoms 1 single 4.0 4.0 4.0
create_atoms 1 single 5.0 4.0 4.0
create_bonds single/bond 1 1 2
create_bonds single/bond 1 2 3
create_bonds single/bond 1 3 4
create_bonds single/bond 1 4 5
pair_style lj/cut 1.5
pair_coeff * * 1.0 1.0 1.5
bond_style harmonic
bond_coeff 1 100.0 1.0
fix nve all nve
fix lan all langevin 0.4 0.4 1.0 12345
timestep 0.005
thermo 1
"""


# Two disconnected 3-atom chains: the connectivity test should report
# 2 clusters of size 3 plus zero singletons.
TWO_CHAINS_SCRIPT = """
units lj
atom_style bond
boundary p p p
neighbor 0.4 bin
neigh_modify every 1 delay 0
region box block 0 12 0 8 0 8
create_box 1 box bond/types 1 extra/bond/per/atom 4
mass 1 1.0
create_atoms 1 single 1.0 4.0 4.0
create_atoms 1 single 2.0 4.0 4.0
create_atoms 1 single 3.0 4.0 4.0
create_atoms 1 single 8.0 4.0 4.0
create_atoms 1 single 9.0 4.0 4.0
create_atoms 1 single 10.0 4.0 4.0
create_bonds single/bond 1 1 2
create_bonds single/bond 1 2 3
create_bonds single/bond 1 4 5
create_bonds single/bond 1 5 6
pair_style lj/cut 1.5
pair_coeff * * 1.0 1.0 1.5
bond_style harmonic
bond_coeff 1 100.0 1.0
fix nve all nve
fix lan all langevin 0.4 0.4 1.0 12345
timestep 0.005
thermo 1
"""


# Minimal dynamic-bond system: type-1 + type-2 atoms in a soft potential
# with `fix bond/create` (a built-in LAMMPS fix; CASPULE adds the
# /random variant on top of this). At least one bond should form
# during the first run window.
DYNAMIC_BOND_SCRIPT = """
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


@pytest.fixture
def core():
    c = allocate_core()
    c.register_link('CASPULEProcess', CASPULEProcess)
    return c


def test_process_instantiation(core):
    proc = CASPULEProcess(config={'input_script': STATIC_CHAIN_SCRIPT}, core=core)
    assert proc.config['input_script']
    proc.close()


def test_initial_state_reports_static_bonds(core):
    proc = CASPULEProcess(config={'input_script': STATIC_CHAIN_SCRIPT}, core=core)
    state = proc.initial_state()
    assert state['num_atoms'] == 5
    assert state['num_bonds'] == 4
    # canonicalised pairs are (lo,hi), so bonds should be exactly:
    pairs = {(b[1], b[2]) for b in state['bonds']}
    assert pairs == {(1, 2), (2, 3), (3, 4), (4, 5)}
    assert state['bonds_by_type'] == {'1': 4}
    assert state['num_clusters'] == 1
    assert state['largest_cluster'] == 5
    assert state['cluster_sizes'] == [5]
    proc.close()


def test_two_disconnected_chains(core):
    proc = CASPULEProcess(config={'input_script': TWO_CHAINS_SCRIPT}, core=core)
    state = proc.initial_state()
    assert state['num_atoms'] == 6
    assert state['num_bonds'] == 4
    assert state['num_clusters'] == 2
    assert state['largest_cluster'] == 3
    assert state['cluster_sizes'] == [3, 3]
    proc.close()


def test_static_bonds_are_stable_after_step(core):
    """A passive chain with no bond/create or bond/break fix should
    keep the same bond list across an update step (formed=0, broken=0)."""
    proc = CASPULEProcess(config={'input_script': STATIC_CHAIN_SCRIPT}, core=core)
    proc.initial_state()
    out = proc.update({}, interval=0.01)
    assert out['num_bonds'] == 4
    assert out['formed_bonds'] == 0
    assert out['broken_bonds'] == 0
    proc.close()


def test_filter_run_strips_run_commands():
    script = "fix nve all nve\nrun 1000\nfix lan all langevin 1 1 1 1\nrerun foo\n"
    filtered = CASPULEProcess._filter_run_commands(script)
    assert 'run 1000' not in filtered
    assert 'rerun foo' not in filtered
    assert 'fix nve' in filtered
    assert 'fix lan' in filtered


def test_dynamic_bond_formation_via_create_fix(core):
    """At least one bond should form during the first step window."""
    proc = CASPULEProcess(config={'input_script': DYNAMIC_BOND_SCRIPT}, core=core)
    state = proc.initial_state()
    assert state['num_bonds'] == 0  # no bonds at t=0
    out = proc.update({}, interval=0.05)  # 10 timesteps of bond/create
    assert out['num_bonds'] > 0
    assert out['formed_bonds'] == out['num_bonds']  # all are new since prev=0
    assert out['broken_bonds'] == 0
    # Some atoms should now sit in 2-clusters
    assert max(out['cluster_sizes']) >= 2
    proc.close()


def test_missing_input_raises(core):
    """No input_script and no input_file should raise on first build."""
    proc = CASPULEProcess(config={}, core=core)
    with pytest.raises(ValueError, match='input_file or input_script'):
        proc.initial_state()


def test_atoms_to_remove_deletes_named_atoms(core):
    """Passing atoms_to_remove=[2,3] should delete two atoms and one bond
    each at both ends of the missing pair (chain becomes 1-?-4-5 → atoms
    {1,4,5} with one bond 4-5). Connectivity stats should reflect this."""
    proc = CASPULEProcess(config={'input_script': STATIC_CHAIN_SCRIPT}, core=core)
    proc.initial_state()
    out = proc.update({'atoms_to_remove': [2, 3]}, interval=0.01)
    assert out['num_atoms'] == 3
    remaining = {(b[1], b[2]) for b in out['bonds']}
    assert remaining == {(4, 5)}
    assert out['num_bonds'] == 1
    proc.close()


def test_atoms_to_remove_empty_is_noop(core):
    """Empty atoms_to_remove must not perturb the simulation."""
    proc = CASPULEProcess(config={'input_script': STATIC_CHAIN_SCRIPT}, core=core)
    proc.initial_state()
    out = proc.update({'atoms_to_remove': []}, interval=0.01)
    assert out['num_atoms'] == 5
    assert out['num_bonds'] == 4
    proc.close()


def test_atoms_to_remove_skips_already_removed(core):
    """Re-issuing the same removal list on a later step is a no-op
    (already-deleted IDs are silently skipped, never raise)."""
    proc = CASPULEProcess(config={'input_script': STATIC_CHAIN_SCRIPT}, core=core)
    proc.initial_state()
    proc.update({'atoms_to_remove': [3]}, interval=0.01)
    out = proc.update({'atoms_to_remove': [3]}, interval=0.01)
    assert out['num_atoms'] == 4  # still gone, not an error
    proc.close()


def test_cluster_max_atoms_disables_computation(core):
    proc = CASPULEProcess(
        config={'input_script': STATIC_CHAIN_SCRIPT, 'cluster_max_atoms': 1},
        core=core,
    )
    state = proc.initial_state()
    # Limit (1) is below the 5 atoms so cluster stats short-circuit
    assert state['num_clusters'] == 0
    assert state['largest_cluster'] == 0
    assert state['cluster_sizes'] == []
    # But the bond list itself still works
    assert state['num_bonds'] == 4
    proc.close()
