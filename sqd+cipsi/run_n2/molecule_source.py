import warnings
import ffsim
import pyscf
import pyscf.cc
import pyscf.mcscf

warnings.filterwarnings("ignore")
open_shell = False
spin_sq = 0

mol = pyscf.gto.Mole()
mol.build(
    atom=[["N", (0, 0, 0)], ["N", (1.0, 0, 0)]],
    basis="6-31g",
    symmetry="Dooh",
)

print(mol.atom)
print("=" * 26)

n_frozen = 2
active_space = range(n_frozen, mol.nao_nr())

print("exact energy:")

scf = pyscf.scf.RHF(mol).run()

print("molecular integrals with CASCI:")

num_orbitals = len(active_space)

n_electrons = int(sum(scf.mo_occ[active_space]))
num_elec_a = (n_electrons + mol.spin) // 2
num_elec_b = (n_electrons - mol.spin) // 2

nelec = (num_elec_a, num_elec_b)

cas = pyscf.mcscf.CASCI(
    scf,
    num_orbitals,
    nelec,
)

mo = cas.sort_mo(active_space, base=0)

hcore, nuclear_repulsion_energy = cas.get_h1cas(mo)

eri = pyscf.ao2mo.restore(
    1,
    cas.get_h2cas(mo),
    num_orbitals,
)

exact_energy = cas.run(mo).e_tot


print("amp calc on ccsd:")

ccsd = pyscf.cc.CCSD(
    scf,
    frozen=[i for i in range(mol.nao_nr()) if i not in active_space],
)

ccsd.run()

t1 = ccsd.t1
t2 = ccsd.t2


print("constructing ucj factors")

n_reps = 2

alpha_alpha_indices = [(p, p + 1) for p in range(num_orbitals - 1)]
alpha_beta_indices = [(p, p) for p in range(0, num_orbitals, 4)]

ucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
    t2=t2,
    t1=t1,
    n_reps=n_reps,
    interaction_pairs=(alpha_alpha_indices, alpha_beta_indices),
)

print("going into qc")
