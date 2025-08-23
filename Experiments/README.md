# CGM Experimental Framework

**Common Governance Model (CGM)** - Mathematical and experimental framework for understanding fundamental physics through gyrogroup geometry and recursive memory structures.

The experiments test the four fundamental theorems of CGM:

1. **Dimensional Homomorphism**: The map from dimension vectors to units is a group homomorphism
2. **Unique G Monomial**: Gravitational constant requires a dimensionless coupling κ
3. **Base-Unit Identities**: M₀ = m⋆, L₀ = ħ/(m⋆c), T₀ = ħ/(m⋆c²)
4. **Geometric Theorems**: Gyrotriangle defect = area/c², Thomas-Wigner rotation

## 🚀 Quick Start

### Windows
Double-click `run_cgm_experiments.bat` to automatically:
- Set up the Python environment
- Install dependencies
- Run all experiments
- Display results

### PowerShell
Run `.\run_cgm_experiments.ps1` in PowerShell to:
- Execute the same workflow
- See detailed progress
- Handle any errors

### Manual Execution
```bash
cd Experiments
python run_experiments.py
```

## 📁 Project Structure

```
Experiments/
├── core/                    # Core mathematical operations
│   ├── dimensions.py       # Dimensional calibration engine
│   ├── gyrovector_ops.py  # Gyrovector operations
│   ├── gyrotriangle.py    # Gyrotriangle implementation
│   └── recursive_memory.py # Recursive memory structure
├── experiments/            # Experimental modules
│   ├── physical_constants.py      # Constants derivation
│   ├── gravity_coupling.py        # Gravitational coupling
│   ├── fine_structure_focus.py    # Fine structure analysis
│   └── core_experiments.py        # Core experiment runner
├── theorems/               # Mathematical proofs
│   ├── run_proofs.py      # Theorem proof runner
│   ├── gyrogeometry.py    # Geometric theorems
│   └── dimensional_engine.py # Dimensional engine
├── stages/                 # CGM stage implementations
│   ├── cs_stage.py        # Common Source stage
│   ├── una_stage.py       # Unity Non-Absolute stage
│   ├── ona_stage.py       # Opposition Non-Absolute stage
│   └── bu_stage.py        # Balance Universal stage
├── tests/                  # Test suite
├── run_experiments.py      # Main experiment runner
├── run_cgm_experiments.bat # Windows launcher
└── run_cgm_experiments.ps1 # PowerShell launcher
```

## 🔬 Key Experiments

### 1. Physical Constants Derivation
- Tests dimensional calibration engine
- Validates base-unit identities
- Checks c-invariance preservation

### 2. Gravity Coupling Analysis
- Anchor mass sweep experiments
- α_G scaling law verification
- Planck mass inference consistency

### 3. Geometric Theorem Validation
- Gyrotriangle defect = area/c²
- Thomas-Wigner small-velocity gyration
- Property-based homomorphism testing

### 4. Recursive Memory Structure
- Coherence field accumulation
- Monodromy residue computation
- Phase gradient analysis

## 📊 Results

All experiments generate:
- **Numerical results** (`.npy` files)
- **JSON summaries** for machine readability
- **Human-readable reports** in console output
- **Validation status** for each theorem

## 🛠️ Dependencies

- Python 3.8+
- NumPy
- SciPy (for advanced mathematical functions)
- Matplotlib (for plotting, optional)

## 📚 Documentation

- **README_SCORECARD.md**: What's proven vs. diagnostic
- **TECH_NOTE_OUTLINE.md**: Technical publication outline
- **CHANGELOG.md**: Development history and changes

## 🔍 Troubleshooting

### Common Issues
1. **Import errors**: Ensure you're in the Experiments directory
2. **Missing dependencies**: Run `pip install -r requirements.txt`
3. **Path issues**: Use absolute paths or ensure proper working directory

### Getting Help
- Check the console output for detailed error messages
- Verify all dependencies are installed
- Ensure Python path includes the Experiments directory

## 🎯 Next Steps

The framework is ready for:
- **κ prediction** from recursive memory
- **Stage transition observables** implementation
- **Physical validation** against experimental data
- **Publication** of proven mathematical foundations

---

*This experimental framework provides the mathematical foundation for the Common Governance Model, establishing rigorous proofs and validation methods for fundamental physics through gyrogroup geometry.*
