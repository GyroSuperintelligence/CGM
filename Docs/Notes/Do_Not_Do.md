# Do Not Do - CGM-RGF Development Guidelines

## 🚫 PROHIBITED PRACTICES

### 1. **Invented Physical Quantities**
- ❌ **Do NOT** create "Planck measures" (Planck length, time, mass, etc.) - these are invented quantities
- ❌ **Do NOT** use any invented fundamental constants or measures
- ✅ **ONLY** use experimentally measured physical constants (ħ, c, G, etc.)
- ✅ **ONLY** use the actual Planck constant ħ, not derived or invented measures

### 2. **Circular Reasoning**
- ❌ **Do NOT** use input values as outputs (e.g., speed of light prediction returning c)
- ❌ **Do NOT** create "ensembles" that just average the same input value
- ❌ **Do NOT** use arbitrary scaling factors that aren't physically motivated
- ✅ **ONLY** use physically meaningful transformations and calculations

### 3. **Overfitting & Fitting**
- ❌ **Do NOT** tune parameters to match expected results
- ❌ **Do NOT** add complexity just to match known values
- ❌ **Do NOT** implement multiple methods that all produce the same answer
- ✅ **ONLY** use methods that are theoretically motivated by CGM
- ✅ **ONLY** add complexity when it serves a clear theoretical purpose

### 4. **False Claims of Success**
- ❌ **Do NOT** claim successes that aren't actually achieved
- ❌ **Do NOT** present basic mathematical consistency as physical validation
- ❌ **Do NOT** exaggerate results or capabilities
- ✅ **ONLY** report what is actually working
- ✅ **ONLY** acknowledge limitations and areas needing work

### 5. **Questionable Numerical Practices**
- ❌ **Do NOT** use numerical artifacts as evidence
- ❌ **Do NOT** ignore numerical precision issues
- ❌ **Do NOT** claim precision when results are within error margins
- ✅ **ONLY** use robust numerical methods with proper error handling
- ✅ **ONLY** validate numerical results with multiple approaches

### 6. **Unvalidated Physical Claims**
- ❌ **Do NOT** claim physical predictions without proper validation
- ❌ **Do NOT** present mathematical models as physical reality
- ❌ **Do NOT** extrapolate beyond what's actually calculated
- ✅ **ONLY** make claims that are directly supported by calculations
- ✅ **ONLY** present results as theoretical predictions, not facts

## 📋 CURRENT FOCUS AREAS

### Immediate Priorities:
1. **Fix Core Numerical Issues** - Monodromy calculations returning zero, coherence field problems
2. **One Focused Prediction** - Pick one physical quantity and make it work properly
3. **Proper Validation Framework** - Add error analysis, statistical testing, comparison with experiment

### Long-term Goals:
1. **Meaningful Physical Predictions** - Actual testable predictions, not just mathematical consistency
2. **Experimental Validation** - Compare against real experimental data
3. **Theoretical Refinement** - Improve CGM formalism based on results

## 🔬 SCIENTIFIC INTEGRITY CHECKLIST

Before implementing any feature:
- [ ] Is this based on actual CGM theory, not invented quantities?
- [ ] Does this avoid circular reasoning?
- [ ] Can this be tested against experimental data?
- [ ] Is this a meaningful physical prediction?
- [ ] Have I avoided overfitting or fitting?

## 📊 VALIDATION REQUIREMENTS

For any claimed success:
- [ ] Multiple independent validation methods
- [ ] Statistical significance testing
- [ ] Comparison with experimental data
- [ ] Error analysis and uncertainty quantification
- [ ] Peer review of methodology

---

**Remember**: The goal is scientific progress, not artificial success metrics. Quality over quantity, accuracy over enthusiasm.
