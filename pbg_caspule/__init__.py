"""pbg-caspule — process-bigraph wrapper for CASPULE.

CASPULE is a modified LAMMPS that supports dynamic bond formation
and breaking via `fix bond/create/random` and `fix bond/break`. This
package exposes a `CASPULEProcess` Process that drives a CASPULE (or
any bond-aware LAMMPS) simulation forward in time and reports bond
state through dedicated process-bigraph ports.
"""

from pbg_caspule.processes import CASPULEProcess
from pbg_caspule.composites import make_caspule_document

__all__ = ['CASPULEProcess', 'make_caspule_document']
