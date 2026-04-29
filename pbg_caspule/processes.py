"""CASPULE Process wrapper for process-bigraph.

CASPULE is a modified LAMMPS that adds `fix bond/create/random`
(random-partner bond formation) on top of LAMMPS's built-in
`fix bond/create` and `fix bond/break`. This wrapper drives any
bond-aware LAMMPS build forward in time and reports the live bond
network through dedicated process-bigraph ports — bond list, per-type
counts, formed/broken deltas, bond energy, and connected-component
cluster sizes.
"""

import os
from process_bigraph import Process


class CASPULEProcess(Process):
    """Bridge Process wrapping CASPULE / bond-aware LAMMPS.

    Configure with a standard LAMMPS input script (a path to a `.in`
    file or an inline string). `run` and `rerun` commands are stripped
    out at load time; the bridge issues `run N` calls itself based on
    the requested update interval. CASPULE-specific commands such as
    `fix ID grp bond/create/random ...` and `fix ID grp bond/break ...`
    are passed through to LAMMPS untouched.

    On every `update()` the process reads the current global bond list
    via `gather_bonds()` and emits it through the `bonds`, `bonds_by_type`,
    `num_bonds`, `formed_bonds`, `broken_bonds`, and `cluster_sizes`
    ports, alongside the standard LAMMPS thermo state.

    Config:
        input_file: path to a LAMMPS / CASPULE `.in` input file
        input_script: inline LAMMPS / CASPULE script (alternative to
            input_file)
        working_directory: directory used for resolving relative paths
            (e.g. `read_data`) — defaults to the directory of input_file,
            or CWD for input_script
        cluster_max_atoms: skip cluster-distribution computation when
            the system has more atoms than this (the connected-component
            cost is linear in atoms+bonds but the dict construction can
            grow noticeable for very large systems). Default 200000.
    """

    config_schema = {
        'input_file': {'_type': 'string', '_default': ''},
        'input_script': {'_type': 'string', '_default': ''},
        'working_directory': {'_type': 'string', '_default': ''},
        'cluster_max_atoms': {'_type': 'integer', '_default': 200000},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self._lmp = None
        self._first_run = True
        self._dt = None
        self._prev_num_bonds = 0
        self._prev_bond_set = set()

    def inputs(self):
        return {}

    def outputs(self):
        return {
            # Thermodynamic state (overwrite[...] = absolute, not delta)
            'temperature': 'overwrite[float]',
            'potential_energy': 'overwrite[float]',
            'kinetic_energy': 'overwrite[float]',
            'total_energy': 'overwrite[float]',
            'pressure': 'overwrite[float]',
            'volume': 'overwrite[float]',
            'box_dimensions': 'overwrite[list]',
            # Per-atom state
            'num_atoms': 'overwrite[integer]',
            'positions': 'overwrite[list]',
            'velocities': 'overwrite[list]',
            'atom_types': 'overwrite[list]',
            # Bond state — the CASPULE-relevant ports
            'num_bonds': 'overwrite[integer]',
            'bonds': 'overwrite[list]',
            'bonds_by_type': 'overwrite[map[integer]]',
            'bond_energy': 'overwrite[float]',
            # Bond dynamics — counts since the previous update()
            'formed_bonds': 'overwrite[integer]',
            'broken_bonds': 'overwrite[integer]',
            # Connectivity — cluster sizes from the live bond graph
            'num_clusters': 'overwrite[integer]',
            'largest_cluster': 'overwrite[integer]',
            'cluster_sizes': 'overwrite[list]',
        }

    @staticmethod
    def _filter_run_commands(script):
        """Strip `run` / `rerun` commands so the bridge can drive integration."""
        out = []
        for line in script.split('\n'):
            stripped = line.split('#', 1)[0].strip()
            tokens = stripped.split()
            if tokens and tokens[0] in ('run', 'rerun'):
                continue
            out.append(line)
        return '\n'.join(out)

    def _resolve_script(self):
        cfg = self.config
        if cfg['input_file']:
            path = cfg['input_file']
            with open(path) as f:
                script = f.read()
            wd = cfg['working_directory'] or os.path.dirname(os.path.abspath(path))
            return script, wd
        if cfg['input_script']:
            return cfg['input_script'], cfg['working_directory']
        raise ValueError(
            'CASPULEProcess requires either input_file or input_script')

    def _build_simulation(self):
        if self._lmp is not None:
            return

        from lammps import lammps

        script, wd = self._resolve_script()
        script = self._filter_run_commands(script)

        original_cwd = os.getcwd()
        if wd:
            os.chdir(wd)
        try:
            self._lmp = lammps(cmdargs=['-nocite', '-log', 'none', '-screen', 'none'])
            self._lmp.commands_string(script)
        finally:
            if wd:
                os.chdir(original_cwd)

        self._dt = self._lmp.extract_global('dt')

    def _read_bonds(self):
        """Pull the global bond list from LAMMPS.

        gather_bonds returns (nbonds, flat list of [type, atom1, atom2, ...]).
        We canonicalise each pair to (min,max) so bond identity is
        order-independent and we can reliably diff against the previous step.
        """
        nbonds, raw = self._lmp.gather_bonds()
        bonds = []
        bond_set = set()
        bonds_by_type = {}
        for i in range(nbonds):
            btype = int(raw[3 * i])
            a1 = int(raw[3 * i + 1])
            a2 = int(raw[3 * i + 2])
            lo, hi = (a1, a2) if a1 <= a2 else (a2, a1)
            bonds.append([btype, lo, hi])
            bond_set.add((lo, hi))
            # bigraph-schema maps key by string by default — serialise the
            # bond-type integer so it round-trips through the type system.
            key = str(btype)
            bonds_by_type[key] = bonds_by_type.get(key, 0) + 1
        return nbonds, bonds, bond_set, bonds_by_type

    def _cluster_stats(self, num_atoms, bond_set):
        """Build the bond graph and return (#clusters, largest, sorted sizes).

        Uses a small union-find so we don't pull in networkx for the
        Process itself — keeps the runtime dependency surface minimal.
        Isolated atoms each count as a singleton cluster.
        """
        if num_atoms > self.config['cluster_max_atoms']:
            return 0, 0, []

        parent = list(range(num_atoms + 1))  # 1-indexed atom IDs

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for a, b in bond_set:
            if 1 <= a <= num_atoms and 1 <= b <= num_atoms:
                union(a, b)

        sizes = {}
        for atom in range(1, num_atoms + 1):
            r = find(atom)
            sizes[r] = sizes.get(r, 0) + 1

        size_list = sorted(sizes.values(), reverse=True)
        largest = size_list[0] if size_list else 0
        return len(size_list), largest, size_list

    def _read_state(self):
        lmp = self._lmp
        natoms = lmp.get_natoms()
        nlocal = lmp.extract_setting('nlocal')

        x = lmp.numpy.extract_atom('x')[:nlocal].copy()
        v = lmp.numpy.extract_atom('v')[:nlocal].copy()
        types = lmp.numpy.extract_atom('type')[:nlocal].copy()

        boxlo, boxhi, _xy, _yz, _xz, _periodicity, _box_change = lmp.extract_box()
        lx = boxhi[0] - boxlo[0]
        ly = boxhi[1] - boxlo[1]
        lz = boxhi[2] - boxlo[2]

        nbonds, bonds, bond_set, bonds_by_type = self._read_bonds()

        formed = len(bond_set - self._prev_bond_set)
        broken = len(self._prev_bond_set - bond_set)
        self._prev_num_bonds = nbonds
        self._prev_bond_set = bond_set

        num_clusters, largest, cluster_sizes = self._cluster_stats(int(natoms), bond_set)

        try:
            bond_energy = float(lmp.get_thermo('ebond'))
        except Exception:
            bond_energy = 0.0

        return {
            'temperature': float(lmp.get_thermo('temp')),
            'potential_energy': float(lmp.get_thermo('pe')),
            'kinetic_energy': float(lmp.get_thermo('ke')),
            'total_energy': float(lmp.get_thermo('etotal')),
            'pressure': float(lmp.get_thermo('press')),
            'volume': float(lmp.get_thermo('vol')),
            'box_dimensions': [lx, ly, lz],
            'num_atoms': int(natoms),
            'positions': x.tolist(),
            'velocities': v.tolist(),
            'atom_types': types.tolist(),
            'num_bonds': int(nbonds),
            'bonds': bonds,
            'bonds_by_type': bonds_by_type,
            'bond_energy': bond_energy,
            'formed_bonds': int(formed),
            'broken_bonds': int(broken),
            'num_clusters': int(num_clusters),
            'largest_cluster': int(largest),
            'cluster_sizes': cluster_sizes,
        }

    def initial_state(self):
        self._build_simulation()
        self._lmp.command('run 0')
        self._first_run = False
        # Initialise the previous-bond bookkeeping from the post-setup state
        _, _, self._prev_bond_set, _ = self._read_bonds()
        return self._read_state()

    def update(self, state, interval):
        self._build_simulation()

        n_steps = max(1, int(round(interval / self._dt)))

        if self._first_run:
            self._lmp.command(f'run {n_steps}')
            self._first_run = False
        else:
            self._lmp.command(f'run {n_steps} pre no post no')

        return self._read_state()

    def close(self):
        """Explicitly close the LAMMPS instance."""
        if self._lmp is not None:
            self._lmp.close()
            self._lmp = None

    def __del__(self):
        try:
            self.close()
        except (ImportError, TypeError):
            pass
