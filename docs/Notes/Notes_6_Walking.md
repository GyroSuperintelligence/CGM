Basil — below is a clean extraction of the theory and mathematics from Herr & Popović, framed so you can plug it straight into CGM and your proto-units. I leave GyroSI aside and keep the language concrete.

---

## Core quantities (about the whole-body centre of mass)

1. **Whole-body angular momentum about the CM (vector $L$)**
   Sum of segment contributions (16-segment model), split into “orbital” and “spin” terms for each segment $i$:

$$
L \;=\; \sum_{i=1}^{16}\Big[(r_i^{CM}-r^{CM})\times m_i\,(v_i - v^{CM})\;+\;I_i\,\omega_i\Big].
$$

Here $r^{CM}, v^{CM}$ are the whole-body CM position/velocity; $r_i^{CM}, v_i$ the $i$-th segment CM position/velocity; $m_i$ the segment mass; $I_i$ the segment inertia tensor about its CM; $\omega_i$ the segment angular velocity.&#x20;

2. **Moment about the CM and its link to $L$**
   Horizontal component of the CM moment $T_{hor}=(T_x,T_y)$ (about the body CM) equals the time derivative of the horizontal component of $L$:

$$
T_{hor} \;=\; \big[(r_{CP}-r^{CM})\times F\big]_{hor} \;=\; \frac{dL}{dt}\Big|_{hor},
$$

where $F$ is the ground reaction force and $r_{CP}$ is the centre of pressure (CP) on the ground.&#x20;

3. **How CP is measured from platform signals**
   With vertical force $F_z$ and horizontal platform moments $M_x, M_y$ about a known lab reference point on the force plate:

$$
x_{CP} = -\frac{M_y}{F_z},\qquad y_{CP} = \frac{M_x}{F_z}.
$$

(Signs per the paper’s frame.)&#x20;

---

## Decomposition of horizontal ground-reaction force (key control split)

From the identity in (2), the measured horizontal forces $F_x, F_y$ decompose into:

$$
F_x \;=\; \underbrace{\frac{F_z}{z_{CM}}\,(x_{CM} - x_{CP})}_{\text{zero-moment term}}\; \underbrace{-\,\frac{T_y}{z_{CM}}}_{\text{moment force}},\qquad
F_y \;=\; \underbrace{\frac{F_z}{z_{CM}}\,(y_{CM} - y_{CP})}_{\text{zero-moment term}}\; \underbrace{+\,\frac{T_x}{z_{CM}}}_{\text{moment force}}.
$$

* The **zero-moment term** is what you would get if the net horizontal CM moment were zero ($T_x=T_y=0$).
* The **moment force** term arises directly from non-zero horizontal CM moments.
  This split is the backbone of their analysis and evaluation.&#x20;

---

## CMP (centroidal moment pivot) — the geometric “no-moment” point

Define the **CMP** as the ground point where a line through the CM, parallel to the ground reaction force, intersects the ground. Imposing zero horizontal CM moment at that point gives:

$$
\big[(r_{CMP}-r^{CM})\times F\big]_{hor}=0.
$$

Expanding yields explicit coordinates:

$$
x_{CMP} = x_{CM} - z_{CM}\,\frac{F_x}{F_z},\qquad
y_{CMP} = y_{CM} - z_{CM}\,\frac{F_y}{F_z}.
$$

If **CMP = CP**, the horizontal CM moment is zero; any separation indicates non-zero $T_{hor}$. CMP can move outside the support base; CP cannot.&#x20;

---

## Dimensionless units and normalisations

* **Normalising $L$** to reduce inter-subject variance:

$$
N_{\text{subject}} = M_{\text{subject}}\,V_{\text{subject}}\,H_{\text{subject}},\quad
\tilde{L} = L/N_{\text{subject}}.
$$

Here $M$ is body mass, $H$ CM height in quiet standing, $V$ mean self-selected gait speed across seven trials. This renders angular momentum dimensionless and comparable across participants.&#x20;

* **Moments** are often reported scaled by $M_{\text{subject}}\,g\,H_{\text{subject}}$ (dimensionless).&#x20;

* **Model/estimation checks**: computing $L$ by (i) kinematic summation (eqn above) and (ii) time-integrating the measured moment about the CM agree well (R² ≈ 0.97–0.98 for components), and errors were bounded using the flight phase of running (where $L$ should be conserved).&#x20;

---

## Main empirical results (steady walking at self-selected speed)

* **Whole-body $L$ is small and tightly regulated** about all three axes during the gait cycle (means ±1 s.d., dimensionless):
  $ |\tilde{L}_x|\lesssim 0.05,\; |\tilde{L}_y|\lesssim 0.03,\; |\tilde{L}_z|\lesssim 0.01.$&#x20;

* **CM moments are small** (dimensionless, means +1 s.d.):
  $|T_x| \lesssim 0.07,\; |T_y| \lesssim 0.03,\; |T_z| \lesssim 0.014.$&#x20;

* **Zero-moment forces predict measured horizontal forces well**:
  Across 10 participants, $R^2$ ≈ 0.91 (medio-lateral $x$), 0.90 (anterior–posterior $y$); no significant difference between directions.&#x20;

* **CMP stays within the support base and near CP** in steady walking; mean CMP–CP separation $\bar{\varepsilon} \approx 14\%$ of foot length (across participants).&#x20;

* **Segmental cancellation is large** (PC-based estimate): about **95%** of angular momentum cancels in the medio-lateral direction, **\~70%** in anterior–posterior, and **\~80%** in vertical. A small number of principal components suffice (NE≈4) to explain >90% of the segmental momentum data; tuning coefficients quantify how PCs are weighted through the gait cycle. &#x20;

* **Plane dependence**: strongest inter-participant variability appears in the **coronal (x–z) plane**, consistent with the notion that medio-lateral balance is the least passively stable and thus most actively controlled.&#x20;

---

## When angular momentum is intentionally not regulated

* **Hula-hoop (double support)** and **exaggerated walking** show angular momentum magnitudes roughly an order of magnitude larger than normal walking; **zero-moment predictions fail** (e.g., exaggerated walking $R^2_x \approx 0.01$, $R^2_y \approx -1.6$), and **CMP diverges from CP** (mean normalised separation ≈ **50%** of foot length). These tasks **use moment forces deliberately** to move/stop the CM. &#x20;

* **One-leg balance example (sign analysis of eqn for $F_x$)**: if $x_{CM}$ moves outside the lateral edge of the stance foot, the zero-moment term pushes you further out; recovery requires generating a **positive CM moment $T_y$** (by rotating arms/trunk/leg) so that the **moment force** term $-T_y/z_{CM}$ pulls the CM back.&#x20;

---

## The PC framework used to quantify cancellation (what you’d actually compute)

* Build segmental angular momentum vectors per axis and apply PCA:
  $ \Lambda_j(t)\,P^j = \sum_{i=1}^{16} C_i^j(t)\,P_i^j,$
  where $P_i^j$ are unit PC vectors, $C_i^j(t)$ time-dependent weights; define normalised tuning coefficients $c_i^j(t)$ so that $\sum_i \big[c_i^j(t)\big]^2 = 1$ for each axis $j$. &#x20;

* Cancellation metric (per axis $j$) aggregates the largest PCs (NE≈4) weighted by their explained variance $DE_i^j$:

  $$
  S_j \;=\; 1 \;-\; \sum_{i=1}^{NE} DE_i^j\;\Big(\sum_{q=1}^{16} P_{iq}^j\Big)^2
  \quad\text{(as used to report ~95%, ~70%, ~80%).}
  $$

  (The paper’s definition is expressed as a weighted sum of orthogonal PC contributions; the exact discrete form is given around their Eqn (19).)&#x20;

---

## How this maps to CGM (so you can use it)

I will be explicit about what is directly supported by the paper and what is an interpretive bridge to CGM.

1. **Zero-moment ≈ local rotational closure (BU-like regime).**
   When $T_{hor}\approx 0$ and CMP≈CP, the derivative of horizontal angular momentum is near zero, and the horizontal forces reduce to the geometric term proportional to the CM–CP offset. This is a **local closure** condition during steady walking. In CGM language, that is your “balanced, associative” regime where residual twist is negligible; the body achieves near-commutative behaviour at the whole-body level. (Paper support: the decomposition and CMP=CP condition; your CGM mapping is conceptual.) &#x20;

2. **Moment-force ≈ controlled departure from closure (ONA-like excursion).**
   When tasks demand manoeuvrability or recovery, the system injects horizontal CM moments; CMP moves away from CP and the moment term dominates. In CGM terms, you are **intentionally re-introducing non-associativity** (gyration) to achieve new states (turning, quick repositioning). (Paper support: hula-hoop/exaggerated walking results; mapping is interpretive.)&#x20;

3. **Segmental cancellation ≈ “balance through opposition” not “balance by stillness”.**
   The whole remains quiet while parts are active; large, coordinated counter-rotations cancel at the top level. That is exactly a **non-absolute opposition** principle realised biomechanically. (Directly supported numerically; mapping is interpretive.)&#x20;

4. **Dimensionless practice aligns with your proto-units.**
   The paper’s normalisations (for $L$ by $MVH$; for $T$ by $MgH$) are precisely the sort of **dimensionless scaling** your unit system prioritises. You can treat these as the **human locomotion analogues** of CGM’s dimensionless invariants: they separate geometry/control from scale. (This is a methodological alignment rather than a claim from the paper.) &#x20;

5. **A concrete “defect” observable:**
   Use **CMP–CP separation** (normalised by foot length) as the **measured defect from rotational closure**. In steady walking the mean defect is small (\~14%); in intentional manoeuvres it can be large (\~50%). That is a clean, unitless signal for “how much non-closure” the system is using. &#x20;

---

## Assumptions and scope (so you don’t over-generalise)

* Rigid-segment model (16 segments), standard force-plate CP definitions, and steady, level walking. &#x20;

* Gravity appears implicitly via $F_z$ and the $MgH$ scaling; the work is not about gravitational field modelling but about how **vertical loading couples into horizontal control** through geometry and CM moment.&#x20;

* The “closure” statements are **empirical** for steady walking; humans **deliberately violate** zero-moment when needed (manoeuvres, recovery), in which case CMP moves away from CP and moment forces dominate. &#x20;

---
