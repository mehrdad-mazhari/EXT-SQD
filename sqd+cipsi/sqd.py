"""Explicit command-line entry point for quantum sampling and SQD."""

def main():
    import argparse
    import importlib
    from functools import partial
    import ffsim
    import matplotlib.pyplot as plt
    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    from qiskit_addon_sqd.fermion import (
        diagonalize_fermionic_hamiltonian,
        solve_sci_batch,
    )
    from datetime import datetime

    parser = argparse.ArgumentParser(description="Prepare a molecule and run SQD on IBM hardware.")
    parser.add_argument("--molecule", choices=("H2", "N2", "CO"), default="H2")
    parser.add_argument("--backend", default="ibm_marrakesh")
    args = parser.parse_args()
    model = importlib.import_module(args.molecule).prepare()
    eri = model["eri"]
    exact_energy = model["exact_energy"]
    hcore = model["hcore"]
    nelec = model["nelec"]
    nuclear_repulsion_energy = model["nuclear_repulsion_energy"]
    num_orbitals = model["num_orbitals"]
    ucj_op = model["ucj_op"]

    # hf+ucj
    circuit = QuantumCircuit(2 * num_orbitals)
    circuit.append(
        ffsim.qiskit.PrepareHartreeFockJW(num_orbitals, nelec), circuit.qubits
    )
    circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(ucj_op), circuit.qubits)
    circuit.measure_all()

    service = QiskitRuntimeService()
    backend = service.backend(args.backend)
    pass_manager = generate_preset_pass_manager(optimization_level=1, backend=backend)
    pass_manager.pre_init = ffsim.qiskit.PRE_INIT
    isa_circuit = pass_manager.run(circuit)

    # submit
    job = SamplerV2(mode=backend).run([isa_circuit], shots=1000)
    print("Job ID:", job.job_id())
    samples = job.result()[0].data.meas


    print("diag started")
    print(datetime.now())

    # Store the energy from every batch at every configuration-recovery iteration.
    # The callback receives a list with one SCIResult per batch.
    iteration_energies = []


    def record_energies(batch_results):
        electronic_energies = np.array(
            [float(batch_result.energy) for batch_result in batch_results]
        )
        iteration_energies.append(electronic_energies)
        total_energies = electronic_energies + nuclear_repulsion_energy
        print(
            f"Iteration {len(iteration_energies):2d}: "
            f"batch total energies = {total_energies}; "
            f"best = {total_energies.min():.12f} Eh"
        )


    result = diagonalize_fermionic_hamiltonian(
        one_body_tensor=hcore,
        two_body_tensor=eri,
        bit_array=samples,
        samples_per_batch=350,
        norb=num_orbitals,
        nelec=nelec,
        num_batches=3,
        max_iterations=10,
        energy_tol=1e-4,
        occupancies_tol=1e-4,
        sci_solver=partial(solve_sci_batch, spin_sq=0.0, max_cycle=200),
        symmetrize_spin=True,
        carryover_threshold=1e-5,
        callback=record_energies,
        seed=243,
    )

    # Plot all batch energies, the best energy in each iteration, and the running
    # minimum. The final point of the running minimum is the returned SQD energy.
    energies = np.asarray(iteration_energies) + nuclear_repulsion_energy
    iterations = np.arange(1, len(energies) + 1)
    best_per_iteration = energies.min(axis=1)
    best_so_far = np.minimum.accumulate(best_per_iteration)

    fig, ax = plt.subplots(figsize=(8, 5))
    for batch in range(energies.shape[1]):
        ax.plot(
            iterations,
            energies[:, batch],
            "o--",
            alpha=0.45,
            label=f"Batch {batch + 1}",
        )
    ax.plot(iterations, best_per_iteration, "o-", label="Best in iteration")
    ax.plot(iterations, best_so_far, "k.-", linewidth=2, label="Best so far")
    ax.axhline(exact_energy, color="tab:red", linestyle=":", label="CASCI")
    ax.set(xlabel="Configuration-recovery iteration", ylabel="Total energy (Eh)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("sqd_energy_convergence.png", dpi=200)
    plt.show()

    total_energy = float(result.energy) + nuclear_repulsion_energy
    print(f"SQD energy: {total_energy:.12f} Eh")
    print(f"CASCI energy: {exact_energy:.12f} Eh")
    print(f"SQD error: {(total_energy - exact_energy) * 1000:+.3f} mEh")

if __name__ == "__main__":
    main()
