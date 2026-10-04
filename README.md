# Distributed Bregman Primal–Dual Algorithms for vGNE Seeking with Nonlinear Coupled Inequalities

Python and MATLAB implementations and reproducibility package for the paper:

**Distributed Bregman Primal–Dual Algorithms for vGNE Seeking with Nonlinear Coupled Inequalities**  
Sichen Qian, Hongzhe Liu, Wenwu Yu, and Wei Xing Zheng.

**Paper status:** Submitted to *IEEE Transactions on Automatic Control*.

## Numerical examples

| Script | Example | Figures in the manuscript |
| --- | --- | --- |
| `main.py` | Feasible sets, potential contours, and equilibrium points for the two-player convex and nonconvex examples | Fig. 1 |
| `example_1_convex.m` | Ten-agent game with a convex coupled inequality | Figs. 2–5 |
| `example2_nonconvex.m` | Ten-agent game with a structured composite nonconvex coupled inequality, plus a separate two-player annular diagnostic | Figs. 6–9 |

Both MATLAB scripts are self-contained. Model parameters, communication graphs,
equilibrium targets, and deterministic initial conditions are defined in the
scripts. Numerical checks of the theorem inequalities run before integration.

The two-player diagnostic in the nonconvex script compares stationary and
global equilibrium branches using a unilateral global best-response gap.
Both flows start from the same primal profile, with different multiplier and
canonical-state initializations. This diagnostic is separate from the
ten-agent theorem certificate.

## Requirements

### Python

- Python **3.12 or later**; validated with **3.13.15**.
- NumPy and Matplotlib, with versions specified in `requirements.txt`.

The default Python run uses Matplotlib's built-in mathematical text rendering
and bundled STIX fonts. No external LaTeX installation is required.

### MATLAB

- MATLAB; validated with **R2024a**.
- No additional MATLAB toolboxes, Python packages, or external input files are required.

## Usage

### Python: Fig. 1

From the repository folder, install the dependencies and run:

```sh
python -m pip install -r requirements.txt
python main.py
```

The script saves the two panels as `convex_case.eps`, `convex_case.png`,
`nonconvex_case.eps`, and `nonconvex_case.png` in `results/figure1/`.

An alternative output folder can be selected with `--output-dir`:

```sh
python main.py --output-dir path/to/results
```

For the manuscript's LaTeX font configuration, use:

```sh
python main.py --usetex
```

This optional mode requires `latex`, `dvips`, and `dvipng` on `PATH`, together
with the LaTeX packages `amsmath`, `amssymb`, `newtxtext`, and `newtxmath`.
Ghostscript is used for EPS distillation when available. See the
[Matplotlib LaTeX rendering documentation](https://matplotlib.org/stable/users/explain/text/usetex.html).
The default and LaTeX modes use the same plotted data; the fonts differ.

For Fig. 1(b), the script sums the two local constraints
`g_i(x_i) = 0.5 * (x_i**2 - 5/8)**2 - 9/128`.
The aggregate constant is therefore `9/64`.

### MATLAB: Figs. 2–9

Open either MATLAB script and click **Run**, or execute the following
commands from the folder containing the scripts:

```matlab
example_1_convex
example2_nonconvex
```

The MATLAB scripts can also be run from another folder using MATLAB's `run`
function with the script's location. All three programs save their default
outputs relative to their own location, so no output paths need to be edited:

```text
results/
  figure1/
  example1_convex/
  example2_nonconvex/
```

Running a program again overwrites its corresponding output files. Results
from the Python figure and the MATLAB examples are stored in separate folders.

## Outputs

Each MATLAB script saves figures in `.fig`, `.png`, and `.eps` formats, CSV tables
of numerical checks and equilibrium data, and a MAT file containing model
parameters, initialization, trajectories, and numerical metrics:

- `results/example1_convex/convex_run_data.mat`
- `results/example2_nonconvex/nonconvex_run_data.mat`

The nonconvex script also runs a second simulation with tighter solver
tolerances and a smaller maximum step size. It saves solver-sensitivity
tables, an annular-diagnostic summary, and `globality_gap_report.txt`.

The scripts stop with an assertion error if a required check fails.

## Figure correspondence

The Python panels are exported as `.eps` and `.png`; MATLAB figures are
exported as `.fig`, `.png`, and `.eps`:

| Figure | Output folder | File stem | Content |
| --- | --- | --- | --- |
| Fig. 1(a) | `results/figure1/` | `convex_case` | Convex feasible set, potential contours, and vGNE |
| Fig. 1(b) | `results/figure1/` | `nonconvex_case` | Nonconvex feasible set, potential contours, global vGNE, and nonglobal KKT point |
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
