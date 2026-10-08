# HfO2 data provenance

Source: https://github.com/neumannrf/sqd-band-gaps/tree/6105530ed2b4a46b8ba2f333789a7a51182b9a0c/sqd-lattice/runs/hafnium_2/dft_data

`hopping_matrix.npy` is copied unchanged (18 x 18, real Gamma-point matrix).
SHA256: 28780c9ec8be7253cd0dc769a8e45dac834e10402aed630d81a8b527f517833e

Interaction assignments transcribed from `hubbard_params.npy`:
- U = 2.6590 eV on Hf 5d indices 4,5,6,7,8.
- V = 0.5672 eV between those indices and O 2p indices 11,12,13,15,16,17.
- Neutral electron sector: (12,12), from `hubbard_params.py`.

The upstream `pre_process.make_two_body_tensor` sets onsite tensor elements
 to 2U and both directions of intersite elements to 2V. This is reproduced
by default. For a spin-independent PySCF tensor this gives a double-occupancy
energy 2U; the conventional Eq. (1) onsite interpretation instead uses U.
`PAPER_TENSOR_CONVENTION=False` selects that alternative. Do not mix benchmark
energies calculated with these different conventions.

Upstream Apache-2.0 license is included as `LICENSE`.
