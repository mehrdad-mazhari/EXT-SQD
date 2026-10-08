"""Explicit command-line entry point for quantum sampling and SQD."""

def main():
    import argparse
    import json
    from pathlib import Path
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
    from hubbard import nelec, num_orbitals, prepare_hubbard
    from datetime import datetime

    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="Use approximate Aer MPS sampling")
    parser.add_argument("--backend", default="ibm_marrakesh")
    args = parser.parse_args()
    hcore, eri, exact_energy, ucj_op = prepare_hubbard()
    nuclear_repulsion_energy = 0.0  # lattice Hamiltonian has no molecular nuclear term
    print("HfO2 extended Hubbard: 18 orbitals, 24 electrons; energies in eV")

    # hf+ucj
    circuit = QuantumCircuit(2 * num_orbitals)
    circuit.append(
        ffsim.qiskit.PrepareHartreeFockJW(num_orbitals, nelec), circuit.qubits
    )
    circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(ucj_op), circuit.qubits)
    circuit.measure_all()

    if args.local:
        from qiskit_aer import AerSimulator
        from qiskit.primitives.containers import BitArray

        backend = AerSimulator(
            method="matrix_product_state",
            matrix_product_state_max_bond_dimension=64,
            matrix_product_state_truncation_threshold=1e-10,
            max_parallel_threads=2,
        )
        pass_manager = generate_preset_pass_manager(optimization_level=1, backend=backend)
        pass_manager.pre_init = ffsim.qiskit.PRE_INIT
        isa_circuit = pass_manager.run(circuit)
        print("Sampling locally with MPS (max bond dimension 64), 1000 shots", flush=True)
        counts = backend.run(isa_circuit, shots=1000, seed_simulator=243).result().get_counts()
        samples = BitArray.from_counts(counts, num_bits=2 * num_orbitals)
        Path("hfo2_local_counts.json").write_text(json.dumps(counts, indent=2))
    else:
        service = QiskitRuntimeService()
        backend = service.backend(args.backend)
        pass_manager = generate_preset_pass_manager(optimization_level=1, backend=backend)
        pass_manager.pre_init = ffsim.qiskit.PRE_INIT
        isa_circuit = pass_manager.run(circuit)
        job = SamplerV2(mode=backend).run([isa_circuit], shots=1000)
        print("Job ID:", job.job_id(), flush=True)
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
            f"best = {total_energies.min():.12f} eV"
        )


    result = diagonalize_fermionic_hamiltonian(
        one_body_tensor=hcore,
        two_body_tensor=eri,
        bit_array=samples,
        samples_per_batch=50 if args.local else 350,
        norb=num_orbitals,
        nelec=nelec,
        num_batches=3,
        max_iterations=3 if args.local else 10,
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
    if exact_energy is not None:
        ax.axhline(exact_energy, color="tab:red", linestyle=":", label="Exact Hubbard (FCI)")
    ax.set(xlabel="Configuration-recovery iteration", ylabel="Hubbard energy (eV)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("sqd_energy_convergence.png", dpi=200)
    if not args.local:
        plt.show()

    total_energy = float(result.energy) + nuclear_repulsion_energy
    print(f"SQD energy: {total_energy:.12f} eV")
    if exact_energy is not None:
        print(f"Exact Hubbard energy: {exact_energy:.12f} eV")
        print(f"SQD error: {total_energy - exact_energy:+.6f} eV")

    Path("hfo2_run_result.json").write_text(json.dumps({
        "sampling": "local_mps_bond64" if args.local else args.backend,
        "shots": 1000,
        "energy_eV": total_energy,
        "iteration_energies_eV": energies.tolist(),
    }, indent=2))


if __name__ == "__main__":
    main()
