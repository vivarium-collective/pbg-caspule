"""Unit tests for the pbg-caspule Visualization Steps.

These tests instantiate the Visualization classes directly and drive
``update(state)`` with hand-crafted snapshots, then assert the rendered
HTML contains the expected markers. They don't require LAMMPS at all
— exercising the viz contract in isolation from the heavy process.

Visualization is a process_bigraph Edge, whose __init__ requires a
``core``. To exercise the pure render contract without booting a full
core we bypass __init__ via ``object.__new__`` and seed only the
attributes the render path actually touches.
"""
from __future__ import annotations

from pbg_caspule.visualizations import BondNetwork3D, BondNetworkPlots


def _new_bond_plots(config=None):
    inst = object.__new__(BondNetworkPlots)
    inst.config = config or {}
    inst.times = []
    inst.history = {
        'temperature': [],
        'total_energy': [],
        'num_bonds': [],
        'bond_energy': [],
    }
    return inst


def _new_bond_3d(config=None):
    inst = object.__new__(BondNetwork3D)
    inst.config = config or {}
    inst._latest = None
    return inst


def test_bond_network_plots_renders_plotly():
    viz = _new_bond_plots({'title': 'Test'})
    out = viz.update({
        'temperature': 0.4,
        'total_energy': -1.0,
        'num_bonds': 2,
        'bond_energy': 0.1,
        'time': 0.05,
    })
    html = out['html']
    assert isinstance(html, str)
    assert 'Plotly.newPlot' in html
    assert 'Test' in html


def test_bond_network_3d_renders_with_three_js():
    """BondNetwork3D must emit a three.js scene with the atom positions baked in."""
    viz = _new_bond_3d({'title': 'Test 3D'})
    state = {
        # Three atoms along the x axis
        'positions': [[1.0, 4.0, 4.0], [2.0, 4.0, 4.0], [3.0, 4.0, 4.0]],
        'types': [1, 1, 1],
        # CASPULE-style triples: [bond_type, atom1_id, atom2_id] (1-based)
        'bonds': [[1, 1, 2], [1, 2, 3]],
    }
    out = viz.update(state)
    html = out['html']
    assert isinstance(html, str)
    # three.js asset reference
    assert 'three.module.js' in html
    assert 'OrbitControls' in html
    # The title flows into the rendered footer
    assert 'Test 3D' in html
    # Atom x-coordinates should appear baked into the JSON payload
    assert '1.0' in html
    assert '2.0' in html
    assert '3.0' in html
    # Bonds should be converted from 1-based [type,a,b] to 0-based [i,j]
    # i.e. [1,1,2] -> [0,1] and [1,2,3] -> [1,2]
    assert '[0, 1]' in html or '[0,1]' in html
    assert '[1, 2]' in html or '[1,2]' in html


def test_bond_network_3d_handles_empty_snapshot():
    """An empty snapshot must still produce valid HTML (no atoms, no bonds)."""
    viz = _new_bond_3d()
    out = viz.update({'positions': [], 'types': [], 'bonds': []})
    html = out['html']
    assert 'three.module.js' in html
