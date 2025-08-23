# CGM-RGF Experimental Framework

This directory contains experiments to test the predictions and theorems of the **Common Governance Model**.

## Setup

The project uses a Python virtual environment (.venv) with the following dependencies:

- `numpy` - For numerical computations and gyrovector operations
- `matplotlib` - For creating visualizations of experimental results
- `scipy` - For advanced mathematical functions (optional)
- `jupyter` - For interactive notebooks (optional)

## Quick Start

### Windows Batch Script (Recommended)
Double-click `run_cgm_experiments.bat` to automatically:
1. Activate the virtual environment
2. Install/update dependencies
3. Run all CGM experiments

### PowerShell Script
Run `.\run_cgm_experiments.ps1` in PowerShell to:
1. Activate the virtual environment
2. Run all CGM experiments

### Manual Setup
If you prefer manual control:

1. **Activate the virtual environment:**
   ```powershell
   .venv\Scripts\Activate.ps1
   ```

2. **Install dependencies:**
   ```powershell
   python -m pip install -r requirements.txt
   ```

3. **Run experiments:**
   ```powershell
   python run_experiments.py
   ```

## Experimental Framework

The experiments test the four fundamental theorems of CGM-RGF:

### 1. CS Axiom: The Source is Common
- Tests chiral asymmetry at the origin
- Verifies left gyration dominance
- Measures primitive chiral distinction

### 2. UNA Theorem: Unity is Non-Absolute
- Tests right gyration activation
- Verifies emergence of observable structure
- Checks chiral memory preservation
- Generates orthogonal spin axes (SU(2) frame)

### 3. ONA Theorem: Opposition is Non-Absolute
- Tests maximal non-associativity
- Verifies translational DoF activation
- Measures opposition non-absoluteness
- Checks bi-gyroassociativity

### 4. BU Theorem: Balance is Universal
- Tests return to identity gyrations
- Verifies coaddition commutativity/associativity
- Checks global closure constraint
- Validates amplitude threshold: A = 1/(2√(2π))

### 5. Gyrotriangle Closure
- Tests defect-free closure with angles (π/2, π/4, π/4)
- Verifies defect asymmetry between positive/negative sequences
- Confirms threshold uniqueness

## Output

The experiments generate:
- **Console output** with detailed test results
- **Results file** (`cgm_experiment_results.npy`) containing all numerical data
- **Summary statistics** showing pass/fail status for each theorem

## Key Predictions Tested

1. **3D Structure Emergence**: Verifies exactly 3 spatial dimensions emerge from recursive constraints
2. **6 Degrees of Freedom**: Confirms 3 rotational + 3 translational DoF from gyrogroup structure
3. **Defect Asymmetry**: Tests that positive angle sequence achieves closure while negative sequence incurs 2π defect
4. **Chiral Memory**: Validates primordial left-bias preservation through all stages
5. **Recursive Infinity**: Tests phase-gradient flattening as saturation depth ℓ* is approached
6. **Universal Balance**: Verifies that non-associativity cancels globally while memory is preserved

## Expected Results

When experiments pass, you should see:
- Gyrotriangle defect δ ≈ 0 (within numerical precision)
- Global closure constraint satisfied
- Observable distinction emerging at UNA
- Peak non-associativity at ONA
- Coaddition commutativity/associativity at BU
- Defect asymmetry favoring positive sequence

## Troubleshooting

### Common Issues:

1. **Virtual Environment Not Activated**
   - Make sure you're running the launcher scripts
   - Or manually activate with `.venv\Scripts\Activate.ps1`

2. **Missing Dependencies**
   - Run `python -m pip install -r requirements.txt`
   - Ensure you're in the activated virtual environment

3. **Python Path Issues**
   - Make sure Python 3.7+ is installed
   - Check that virtual environment is properly created

### Performance Notes:
- Some experiments involve nested loops over test vectors
- Expect 1-2 minutes for complete test suite
- Numerical precision affects some closure tests

## Next Steps

After running the core experiments, you can:
1. **Examine the saved results** in `cgm_experiment_results.npy`
2. **Create visualizations** using the visualization modules
3. **Extend experiments** to test additional predictions
4. **Run threshold searches** for alternative angle combinations

## Architecture

```
CGM_Experiments/
├── .venv/                 # Python virtual environment
├── core/                  # Core mathematical operations
│   ├── gyrovector_ops.py  # Gyrovector space implementation
│   └── gyrotriangle.py    # Defect calculations
├── stages/                # CGM stage implementations
│   ├── cs_stage.py        # Common Source
│   ├── una_stage.py       # Unity Non-Absolute
│   ├── ona_stage.py       # Opposition Non-Absolute
│   └── bu_stage.py        # Balance Universal
├── experiments/           # Experimental test suites
│   └── core_experiments.py # Core theorem tests
├── visualizations/        # (Future) Visualization modules
├── utils/                 # (Future) Utility functions
├── run_experiments.py     # Main experimental runner
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── run_cgm_experiments.bat # Windows launcher
└── run_cgm_experiments.ps1 # PowerShell launcher
```
