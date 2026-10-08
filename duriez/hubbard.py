"""HfO2 extended Hubbard input from the Duriez paper's public repository.

Energies are in eV. PAPER_TENSOR_CONVENTION reproduces its preprocessing
(eri[p,p,p,p]=2U and eri[p,p,q,q]=2V). Set it to False for the conventional
spin-independent representation of Eq. (1) with onsite energy U.
"""

from pathlib import Path
import numpy as np


def hubbard_integrals(hopping, onsite_u, intersite_v):
    """Return real, spin-independent PySCF/SQD chemist-order tensors.

    The usual 1/2-prefactored spin sum gives eri[p,p,p,p] = U[p].
    Equal-spin onsite terms vanish by Pauli exclusion. For the ordered
    intersite sum in Eq. (1), eri[p,p,q,q] = 2*V[p,q]. This avoids
    doubling U when using the spin-independent solver API.
    """
    if any(np.iscomplexobj(x) for x in (hopping, onsite_u, intersite_v)):
        raise ValueError("This RHF/SQD workflow requires real Gamma-point tensors")
    hopping, onsite_u, intersite_v = map(
        lambda x: np.asarray(x, dtype=float), (hopping, onsite_u, intersite_v)
    )
    if hopping.ndim != 2:
        raise ValueError("Expected hopping (n,n), U (n,), V (n,n)")
    n = hopping.shape[0]
    if hopping.shape != (n, n) or onsite_u.shape != (n,) or intersite_v.shape != (n, n):
        raise ValueError("Expected hopping (n,n), U (n,), V (n,n)")
    if not all(np.all(np.isfinite(x)) for x in (hopping, onsite_u, intersite_v)):
        raise ValueError("Model parameters must be finite")
    if not np.allclose(hopping, hopping.T) or not np.allclose(intersite_v, intersite_v.T):
        raise ValueError("hopping and V must be symmetric")
    if not np.allclose(np.diag(intersite_v), 0):
        raise ValueError("V diagonal must be zero; use U for onsite interactions")
    eri = np.zeros((n, n, n, n))
    for p in range(n):
        for q in range(n):
            eri[p, p, q, q] = onsite_u[p] if p == q else 2 * intersite_v[p, q]
    return hopping.copy(), eri


# Source: neumannrf/sqd-band-gaps, runs/hafnium_2/dft_data.
# Orbital indices were transcribed from the source hubbard_params.npy.
PAPER_TENSOR_CONVENTION = True
hopping = np.load(Path(__file__).parent / "data/hfo2/hopping_matrix.npy", allow_pickle=False)
num_orbitals = hopping.shape[0]
nelec = (12, 12)
onsite_u = np.zeros(num_orbitals)
intersite_v = np.zeros_like(hopping)
hf_5d = [4, 5, 6, 7, 8]
oxygen_2p = [11, 12, 13, 15, 16, 17]
onsite_u[hf_5d] = 2.6590
for p in hf_5d:
    for q in oxygen_2p:
        intersite_v[p, q] = intersite_v[q, p] = 0.5672


def material_integrals():
    hcore, eri = hubbard_integrals(hopping, onsite_u, intersite_v)
    if PAPER_TENSOR_CONVENTION:
        for p in range(num_orbitals):
            eri[p, p, p, p] = 2 * onsite_u[p]
    return hcore, eri


def prepare_hubbard(build_ucj=True):
    """Solve lattice RHF/CCSD and return tensors in the same MO basis as UCJ."""
    from pyscf import ao2mo, cc, gto, scf

    site_hcore, site_eri = material_integrals()
    if nelec[0] != nelec[1]:
        raise ValueError("The spin-balanced RHF/UCJ workflow requires Nalpha=Nbeta")
    mol = gto.M(verbose=0)
    mol.nelectron = sum(nelec)
    mol.incore_anyway = True
    mf = scf.RHF(mol)
    mf.get_hcore = lambda *args: site_hcore
    mf.get_ovlp = lambda *args: np.eye(num_orbitals)
    mf.energy_nuc = lambda *args: 0.0
    mf._eri = ao2mo.restore(8, site_eri, num_orbitals)
    mf.max_cycle = 200
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("Hubbard RHF did not converge")
    orbitals = mf.mo_coeff
    hcore = orbitals.T @ site_hcore @ orbitals
    eri = ao2mo.restore(1, ao2mo.incore.full(site_eri, orbitals), num_orbitals)
    # Full FCI for 18 orbitals / 24 electrons is too large for this workflow.
    exact_energy = None
    coupled_cluster = cc.CCSD(mf)
    coupled_cluster.max_cycle = 200
    coupled_cluster.kernel()
    if not coupled_cluster.converged:
        raise RuntimeError("Hubbard CCSD did not converge; adjust model/initial state")
    if not build_ucj:
        return hcore, eri, mf.e_tot, coupled_cluster.e_tot
    import ffsim

    ucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
        t2=coupled_cluster.t2,
        t1=coupled_cluster.t1,
        n_reps=1,
        interaction_pairs=(
            [(p, p + 1) for p in range(num_orbitals - 1)],
            [(p, p) for p in range(num_orbitals)],
        ),
    )
    return hcore, eri, exact_energy, ucj_op
