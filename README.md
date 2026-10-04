# Distributed Bregman Primal–Dual Algorithms for vGNE Seeking with Nonlinear Coupled Inequalities

MATLAB implementation and reproducibility package for the paper:

**Distributed Bregman Primal–Dual Algorithms for vGNE Seeking with Nonlinear Coupled Inequalities**  
Sichen Qian, Hongzhe Liu, Wenwu Yu, and Wei Xing Zheng.

**Paper status:** Submitted to *IEEE Transactions on Automatic Control*.

## Numerical examples

| Script | Example | Figures in the manuscript |
| --- | --- | --- |
| `example1_convex.m` | Ten-agent game with a convex coupled inequality | Figs. 2–5 |
| `example2_nonconvex.m` | Ten-agent game with a structured composite nonconvex coupled inequality, plus a separate two-player annular diagnostic | Figs. 6–9 |

Both scripts are self-contained. Model parameters, communication graphs,
equilibrium targets, and deterministic initial conditions are defined in the
scripts. Numerical checks of the theorem inequalities run before integration.

The two-player diagnostic in the nonconvex script compares stationary and
global equilibrium branches using a unilateral global best-response gap.
Both flows start from the same primal profile, with different multiplier and
canonical-state initializations. This diagnostic is separate from the
ten-agent theorem certificate.

## Requirements

- MATLAB; validated with **R2024a**.
- No additional MATLAB toolboxes, Python packages, or external input files are required.

## Usage

Open either script in MATLAB and click **Run**, or execute the following
commands from the folder containing the scripts:

```matlab
example1_convex
example2_nonconvex
```

The scripts can also be run from another folder using MATLAB's `run` function
with the script's location. All outputs are saved relative to the script's
folder, so no output paths need to be edited:

```text
results/
  example1_convex/
  example2_nonconvex/
```

Running a script again overwrites its corresponding output files. Results
from the two examples are stored in separate folders.

## Outputs

Each script saves figures in `.fig`, `.png`, and `.eps` formats, CSV tables
of numerical checks and equilibrium data, and a MAT file containing model
parameters, initialization, trajectories, and numerical metrics:

- `results/example1_convex/convex_run_data.mat`
- `results/example2_nonconvex/nonconvex_run_data.mat`

The nonconvex script also runs a second simulation with tighter solver
tolerances and a smaller maximum step size. It saves solver-sensitivity
tables, an annular-diagnostic summary, and `globality_gap_report.txt`.

The scripts stop with an assertion error if a required check fails.

## Figure correspondence

The following stems identify the exported `.fig`, `.png`, and `.eps` files:

| Figure | Output folder | File stem | Content |
| --- | --- | --- | --- |
| Fig. 2 | `results/example1_convex/` | `shared_topology` | Shared communication graph |
| Fig. 3 | `results/example1_convex/` | `convex_tra` | Primal and decision-estimate errors |
| Fig. 4 | `results/example1_convex/` | `convex_aux` | Multipliers and auxiliary states |
| Fig. 5 | `results/example1_convex/` | `convex_exp` | Normalized logarithmic error and shared-constraint residual |
| Fig. 6 | `results/example2_nonconvex/` | `nonconvex_tra` | Primal and decision-estimate errors |
| Fig. 7 | `results/example2_nonconvex/` | `nonconvex_aux` | Multiplier deviations and auxiliary states |
| Fig. 8 | `results/example2_nonconvex/` | `nonconvex_exp` | Aggregate, canonical-state, and consistency errors; constraint residuals |
| Fig. 9 | `results/example2_nonconvex/` | `nonconvex_globality` | Annular-diagnostic trajectories and global best-response gaps |

## Citation

If you use this code in your research, please cite the accompanying paper:

```bibtex
@unpublished{qian_bregman_vgne,
  author = {Qian, Sichen and Liu, Hongzhe and Yu, Wenwu and Zheng, Wei Xing},
  title  = {Distributed {Bregman} Primal--Dual Algorithms for {vGNE} Seeking with Nonlinear Coupled Inequalities},
  note   = {Submitted to {IEEE} Transactions on Automatic Control}
}
```
