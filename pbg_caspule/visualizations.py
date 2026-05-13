"""Visualization Step subclasses for pbg-caspule.

Visualizations follow the pbg-superpowers convention (v0.4.15+):
each subclass overrides `update()` to consume per-step state via wires
(like an Emitter), accumulates history internally, and returns
``{'html': '<rendered figure>'}`` each step. The composite spec wires
the input ports to store paths.

See pbg_superpowers.visualization for the base-class contract.
"""
from __future__ import annotations

from pbg_superpowers.visualization import Visualization


class BondNetworkPlots(Visualization):
    """Time-series HTML plot of CASPULE's scalar bond-network outputs.

    Consumes the four core CASPULE scalars (temperature, total_energy,
    num_bonds, bond_energy) at each step, accumulates them across calls,
    and emits a Plotly HTML figure on every update. Downstream consumers
    (dashboards, notebook viewers) read the latest 'html' from the wired
    store.
    """

    config_schema = {
        'title': {'_type': 'string', '_default': 'CASPULE bond network'},
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # One list per consumed scalar; aligned by index across all signals.
        self.times: list[float] = []
        self.history: dict[str, list[float]] = {
            'temperature': [],
            'total_energy': [],
            'num_bonds': [],
            'bond_energy': [],
        }

    def inputs(self):
        return {
            'temperature': 'float',
            'total_energy': 'float',
            'num_bonds': 'integer',
            'bond_energy': 'float',
            'time': 'float',
        }

    def update(self, state, interval=1.0):
        self.times.append(float(state.get('time', len(self.times) * (interval or 1.0))))
        for key in self.history:
            v = state.get(key)
            self.history[key].append(float(v) if v is not None else 0.0)

        title = (self.config or {}).get('title', 'CASPULE bond network')
        traces = []
        for key, ys in self.history.items():
            traces.append(
                '{"x":' + repr(self.times) + ',"y":' + repr(ys) +
                ',"type":"scatter","mode":"lines","name":"' + key + '"}'
            )
        html = (
            f'<div id="bnp" style="height:380px"></div>'
            f'<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>'
            f'<script>Plotly.newPlot("bnp",[{",".join(traces)}],'
            f'{{title:"{title}",margin:{{l:55,r:15,t:35,b:40}},'
            f'xaxis:{{title:"time"}},'
            f'legend:{{orientation:"h",y:-0.2}}}},'
            f'{{responsive:true,displayModeBar:false}});</script>'
        )
        return {'html': html}


class BondNetwork3D(Visualization):
    """three.js-based 3D viewer for atoms + bonds.

    Atoms render as colored spheres at their last-known positions; bonds
    render as thin cylinders between atom pairs. Camera is orbitable
    (mouse-drag rotate, scroll zoom). Accumulates the most-recent state
    on each update; render only shows the final/current snapshot for
    visual cleanliness — a future 'timeline' version can scrub through.
    """

    config_schema = {
        'title': {'_type': 'string', '_default': 'Bond network'},
        'sphere_radius': {'_type': 'float', '_default': 0.4},
        'bond_radius': {'_type': 'float', '_default': 0.08},
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Latest snapshot only — full trajectory will balloon the HTML
        self._latest = None

    def inputs(self):
        # positions: list of [x,y,z] per atom; types: int per atom; bonds: list of [i,j,type]
        return {
            'positions': 'list[list[float]]',
            'types': 'list[integer]',
            'bonds': 'list[list[integer]]',
        }

    def update(self, state, interval=1.0):
        self._latest = {
            'positions': state.get('positions') or [],
            'types': state.get('types') or [],
            'bonds': state.get('bonds') or [],
        }
        return {'html': self._render()}

    def _render(self) -> str:
        import json
        title = (self.config or {}).get('title', 'Bond network')
        data = self._latest or {'positions': [], 'types': [], 'bonds': []}
        # Limit to first 2000 atoms + 3000 bonds to keep HTML reasonable
        positions = (data.get('positions') or [])[:2000]
        types = (data.get('types') or [])[:2000]
        bonds = (data.get('bonds') or [])[:3000]
        # Normalise bond rows to plain [i, j] pairs into the positions list.
        # CASPULE emits [bond_type, atom1_id, atom2_id] triples with 1-based
        # LAMMPS atom IDs; positions are 0-indexed. Subtract 1 to align.
        # Users may also pass plain [i, j] pairs (already 0-indexed).
        bond_pairs = []
        for b in bonds:
            if not b:
                continue
            if len(b) >= 3:
                bond_pairs.append([int(b[1]) - 1, int(b[2]) - 1])
            elif len(b) == 2:
                bond_pairs.append([int(b[0]), int(b[1])])
        data_json = json.dumps({
            'positions': positions,
            'types': types,
            'bonds': bond_pairs,
        })
        sphere_r = float((self.config or {}).get('sphere_radius', 0.4))
        _bond_r = float((self.config or {}).get('bond_radius', 0.08))  # noqa: F841
        return (
            '<div id="viz" style="width:100%;height:480px;border:1px solid #e5e7eb;border-radius:4px"></div>'
            '<script type="importmap">'
            '{"imports": {"three": "https://unpkg.com/three@0.158.0/build/three.module.js",'
            ' "three/addons/": "https://unpkg.com/three@0.158.0/examples/jsm/"}}'
            '</script>'
            '<script type="module">'
            'import * as THREE from "three";'
            'import { OrbitControls } from "three/addons/controls/OrbitControls.js";'
            'const data = ' + data_json + ';'
            'const container = document.getElementById("viz");'
            'const renderer = new THREE.WebGLRenderer({antialias:true});'
            'renderer.setSize(container.clientWidth, 480);'
            'renderer.setClearColor(0xffffff, 1);'
            'container.appendChild(renderer.domElement);'
            'const scene = new THREE.Scene();'
            'const camera = new THREE.PerspectiveCamera(60, container.clientWidth/480, 0.1, 1000);'
            'camera.position.set(20, 20, 20);'
            'const controls = new OrbitControls(camera, renderer.domElement);'
            'scene.add(new THREE.AmbientLight(0xffffff, 0.6));'
            'const sun = new THREE.DirectionalLight(0xffffff, 0.7);'
            'sun.position.set(10,20,10);'
            'scene.add(sun);'
            'const palette = [0x6366f1,0xef4444,0x10b981,0xf59e0b,0x8b5cf6,0xec4899,0x14b8a6,0xf97316];'
            'const sphereGeom = new THREE.SphereGeometry(' + str(sphere_r) + ', 16, 12);'
            'data.positions.forEach((p, i) => {'
            '  const t = data.types[i] || 0;'
            '  const mat = new THREE.MeshLambertMaterial({color: palette[t % palette.length]});'
            '  const m = new THREE.Mesh(sphereGeom, mat);'
            '  m.position.set(p[0], p[1], p[2] || 0);'
            '  scene.add(m);'
            '});'
            'const bondMat = new THREE.LineBasicMaterial({color: 0x6b7280});'
            'data.bonds.forEach(b => {'
            '  const a = data.positions[b[0]], c = data.positions[b[1]];'
            '  if (!a || !c) return;'
            '  const geom = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(...a), new THREE.Vector3(...c)]);'
            '  scene.add(new THREE.Line(geom, bondMat));'
            '});'
            'function animate() { requestAnimationFrame(animate); controls.update(); renderer.render(scene, camera); }'
            'animate();'
            '</script>'
            '<div style="font-size:0.85em;color:#6b7280;margin-top:4px">' + title + ' — drag to rotate, scroll to zoom</div>'
        )
