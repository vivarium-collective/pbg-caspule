"""Custom bigraph-schema types for CASPULE.

CASPULE's first-class data is the global bond list — a list of
(bond_type, atom_id_i, atom_id_j) triples. We express it through
existing built-in types (`list`, `map[integer]`) so no custom type
registration is strictly required, but this module is provided as a
hook for future type extensions (e.g. a typed `bond` record or a
sparse adjacency matrix).
"""


def register_types(core):
    """Register CASPULE-specific bigraph-schema types onto `core`.

    Currently a no-op — CASPULE outputs use built-in types only. Kept
    as a stable extension point so callers can opt in to richer types
    without changing wiring code.
    """
    return core
