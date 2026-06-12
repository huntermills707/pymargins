# Computation graph — clean-break rewrite plan (0.4.0)

2026-06-11 · **rev. 2** (same day, post-adoption plan review — correctness
re-anchored from legacy byte-identity to external oracles; see §0.2)

Execution-grade plan. **Supersedes the Phase-2 workstreams (W2.x) and the
coexistence machinery of
[`061026_computation_graph_plan.md`](061026_computation_graph_plan.md);
Phases 3–5 of that plan remain authoritative with the deltas listed in §6.
Rev. 2 additionally supersedes the byte-identity anchor language wherever
it appears upstream (design §2.2 scope fence and its echoes; req. §1
tier-2 "byte-identical" no-regress; req. §9 anchor harness) — replaced by
the §4 validation protocol. The architectural content of those documents
stands.** Authority chain otherwise unchanged: this plan says *do*,
[`061026_computation_graph_requirements.md`](061026_computation_graph_requirements.md)
says *what precisely*,
[`060926_computation_graph_design.md`](060926_computation_graph_design.md)
says *why*. When ambiguous, escalate up the chain; if still ambiguous,
**stop and ask — do not improvise statistics**. Implementation companion:
[`061126_rewrite_implementation_guide.md`](061126_rewrite_implementation_guide.md)
(signatures, test specs, pitfalls — subordinate to this plan; its pinned
decisions are listed in its Appendix C).

## 0. The decisions (2026-06-11) and why this plan exists

### 0.1 Clean break

The first v0.4.0 implementation built the architecture inverted:
`GComputation` as a kwargs-translation facade over `Margins`
(audited 2026-06-10/11; findings 1–7 fixed, 8–13 open). Root cause was
structural, not incidental — every silent-wrong bug lived in the
translation layer, and doctrine could only be patched from above the
delegation boundary, one hole at a time. The user's verdict, adopted here:

> 0.0.0–0.3.0 was based on a flaw of only one fit, which AIPW and Rubin's
> rules invalidate. The existing work is only well suited for single-fit
> use cases.

**Adopted: clean break at 0.4.0.** The legacy orchestration (`Margins`
session, `MarginsResult`, the glue, the fallback dispatch) is deleted, not
shimmed. No shim, no deprecation cycle (0.x semver; user accepts the
break).

### 0.2 Re-anchor correctness to external oracles (rev. 2)

Rev. 1 inherited the design note's anchor: v0.3.0 reproduced
byte-identically (`np.array_equal`, never `allclose`), kernels frozen
read-only to make that achievable, the test suite ported with asserted
numbers carried verbatim. The user's verdict at plan review, adopted here:

> This is a young package, and I'm happy to ditch code (mostly vibed) as
> needed. I'd rather build a healthier foundation. Happy to keep `Margins`
> as a possible oracle for tests — but there are also options in R.

Byte-identity to 0.3.0 anchors the rewrite to the *old code's outputs* —
precisely the authority this verdict withdraws. A healthier foundation
anchors to statistical truth the package does not own. Adopted
consequences:

1. **Correctness authority moves outside the package.** The anchor is an
   oracle stack (§4): analytic closed forms first, R reference
   implementations second (`marginaleffects`, `survey`, …), legacy
   `Margins` demoted to a corroborating instrument during construction.
2. **Keep is earned, not granted.** The §2 KEEP list becomes
   *presumptive*: kept code passes a review gate (I3″) — formulas traced
   to citations, outputs traced through the oracle suite — before R7
   declares it permanent. "Mostly vibed" code that fails review is fixed
   or rewritten, with the finding recorded.
3. **Dual-run is an instrument, not an invariant.** New-vs-legacy
   divergence is *arbitrated* by the oracle stack, never auto-blamed on
   the new path. Legacy can be wrong.
4. **Legacy defects are fixed, not reproduced.** Oracle-visible defects in
   ≤0.3.0 numbers go to a ledger
   (`notes/061126_legacy_defect_ledger.md`) and the 0.4.0 CHANGELOG gains
   a corrections section. 0.4.0 ships the *correct* number.
5. **Seed/draw derivations are unfrozen.** Determinism and a recorded
   derivation are required; reproducing v0.3.0's exact streams is not.
   Same-seed sim/boot draws MAY differ from 0.3.0 — a breaking-release
   note (R8).

### Amendments to previously adopted verdicts (recorded, not improvised)

| Verdict / rule | Status | Replacement |
|---|---|---|
| Verdict 2 — `Margins` becomes a shim at Phase-2 end | **RESCINDED** (rev. 1) | Legacy is a frozen in-tree *oracle* during construction (§4 layer 3), deleted before the 0.4.0 tag. |
| I3 — legacy behavior sacred in legacy mode | **RESCINDED** (rev. 1) | There is no legacy mode. The doctrine is the only behavior. Fallback code paths are deleted, not gated. |
| Req. §8 coexistence table (0.4 "Margins delegates internally", 1.0 deprecation flip) | **SUPERSEDED** (rev. 1) | 0.4.0 = the break. 1.0 = stability declaration, nothing to deprecate. |
| `from_posterior` (design §3.9: "stays on legacy session only") | **DROPPED with the legacy session** (rev. 1) | If demand returns it re-enters as its own small front-end (design §3.9), not as a survivor. |
| I1 — v0.3.0 reproduced byte-identically (design §2.2 scope fence; req. §9 anchor harness) | **REPLACED** (rev. 2) | I1′ — external oracle anchor (§0.2, §4). Legacy agreement is corroboration-grade only. |
| I3′ — kernels and atoms read-only ("consumed, never rewritten") | **REPLACED** (rev. 2) | I3″ — keep is earned via the review gate (§2). Read-only sanctity dissolved with I1. |
| Verdict 5 — M=1 bank derivations frozen; sim/boot byte-identical | **RESCINDED** (rev. 2) | Determinism + recorded derivation + §4 validation. Redesign freedom exercised once, at R1, or not at all (§7 trap 2). |
| Rev.-1 R7 rule — "asserted numbers carried verbatim — never re-derived" | **AMENDED** (rev. 2) | Reference-derived assertions port as-is; engine-derived expectations are re-anchored to an oracle or dropped, per ledger (R7). The anti-self-anchor principle survives in new form: no expectation may be regenerated *from the new engine without oracle authority*. |
| Verdict 1 (eager C2), Verdict 3 as amended 06-10 (no-fan adapter replay / fan products-only, byte-budget dropped), Verdict 4 (result-op boundary) | **UNCHANGED** | — |

### Invariants (violating any = stop, revert, report)

- **I1′ — The oracle anchor.** Every numeric surface validates against the
  strongest available oracle: analytic closed form where one exists
  (exact), an R reference implementation otherwise (recorded tolerance and
  alignment spec), legacy `Margins` as corroboration during construction.
  Resampling paths, where streams cannot cross languages, validate by
  determinism goldens plus slow-lane calibration. Oracle disagreement
  beyond tolerance ⇒ stop and ask — never silently pick a side. Enforced
  per §4.
- **I2 — Tier-2 no-regress.** The FD-wrapped adapters (lifelines,
  linearmodels, MixedLM) keep delta/simulation working unchanged through
  the new engine. No new requirement imposed on them. Validated through
  the same §4 stack where an oracle reaches them, else legacy
  corroboration + regression goldens.
- **I3″ — Keep is earned (review gate).** Legacy code the new engine
  consumes (kernels, atoms, adapters) passes review at first consumption:
  read-through with every formula traced to a design citation (I4) and
  outputs traced through the §4 suites. Review failures are fixed or
  rewritten — recorded in the defect ledger, never silent. The inverse
  guard: **no gratuitous rewrites** — churn without an oracle-visible
  defect, a citation gap, or a structural need is reverted.
- **I4 — Never invent methodology.** Every formula traces to a design-note
  citation or to shipped, review-passed code being moved. Unspecified ⇒
  stop and ask. (Now load-bearing for rewrites, not only for moves.)
- **I5 — Severity texts are spec.** Design §6 verdicts and steers verbatim;
  *(future)* markers on steers at unshipped machinery.
- **I6 — No delegation to legacy.** The new engine never imports from
  `margins/_session.py`, `margins/_inference_glue.py`, or
  `_result/_margins.py`. Legacy is reachable from *test code only* (the
  corroboration oracle), until R7 deletes it. This makes the facade
  failure mode unrepresentable.

**Working rules** (carried from the 061026 plan): pytest + ruff before any
workstream is declared done; new modules cite the design section + an
`Added in 0.4.0 (RN)` line; slow-marked sims run in the weekly CI lane and
gate release tags; commits per-workstream; surface-API signatures
validated by the facade round-trip are **frozen** — the rewrite changes
what is behind them, not what they look like.

---

## 1. Disposition of the current working tree (uncommitted changes)

**KEEP — commit now, in structured checkpoint commits on `graph_api`.**
Nothing is dropped. Rationale: Phase 1 and the surface are audit-validated;
the facade pieces keep the kept tests importable until each R-workstream
replaces them, and replacing them *with history* beats a working-tree
purge. Suggested commit slices:

1. Phase 1 — `_adapter.py`, four adapter `influence()` impls,
   `tests/test_influence_contract.py`. (Permanent.)
2. Soundness — `_soundness/` + its two test files. (Permanent.)
3. Graph surface — `_graph/`, `steps/`, their tests. (Permanent, evolves.)
4. Interim scaffolding — `estimators/`, `_result/_graphresult.py`,
   `_engine/`, `tests/anchor/`, `tests/test_graphresult.py`,
   `tests/test_compile.py`. Commit message marks them
   `interim — replaced by R1–R6 (see 061126 plan)`.
5. Metadata — `__init__.py` exports, CHANGELOG (rewritten at R8),
   `pyproject.toml` (leave at 0.4.0 on this branch; the *tag* is gated,
   not the string), this plan (rev. 2).

Docs (`docs/explanations/computation_graph.md`,
`docs/tutorials/graph_quickstart.md`) are skeletal; keep as stubs, rewrite
at R8.

---

## 2. Keep / delete inventory (the surgical list)

### KEEP (presumptive) — subject to the review gate (I3″)

"Keep" means *consumed by the new engine after passing review at first
consumption*, not read-only sanctity. The review gate, applied by the
R-workstream that first consumes a module: (a) read-through with formulas
traced to citations; (b) the paths exercised covered by the §4 oracle
suites; (c) findings dispositioned in
`notes/061126_legacy_defect_ledger.md` — *fix in place* / *rewrite* /
*accept with rationale*. A module that passes is kept as-is — the cheapest
healthy outcome, and the expected one for most rows.

| What | Why / review note |
|---|---|
| `_adapter.py` + all of `_adapters/` (incl. Phase-1 `influence()`, `data_fingerprint()`) | The source contract; the capability lattice is already its shape. Phase-1 surface audit-validated; numeric paths reviewed as consumed |
| `_inference/_linearization.py`, `_kappa.py` (package root; incl. `session_kappa:283`), `_delta.py` / `_simulation.py` / `_bootstrap.py` numerics, `_inference/_config.py` | The numbers today; the §4 suites decide whether they stay the numbers. Reviewed at R2/R3 consumption |
| `_generate_resample_indices` (`_inference/_bootstrap.py`), `_generate_simulation_draws` (`_inference/_simulation.py`) | M=1 bank derivations — plain RNG engineering, reviewed at R1; unfrozen by rev. 2 but redesigned once-or-never (§7 trap 2) |
| `margins/_estimands.py`, `margins/_atoms.py` | Estimand definitions are statistics (I4). Citation-reviewed at R2 (first consumption); relocate wholesale at R7; numeric edits only as ledgered review findings |
| `_transforms/` stages, `scenarios`, `survey.py` (`SurveyDesign`), `matching/` | Fit-count-agnostic library; reviewed as consumed |
| `_result/_pooling.py` Rubin arithmetic | Re-pointed at R4; math reviewed against its citations there |
| `_soundness/`, `_graph/_node.py`, `steps/`, `Plan` | New, audit-validated |
| `adjust()` / `AdjustedResults` | Result-layer utility, carries over (req. §7) |

### DELETE — at R7, after the parity gate

| What | Successor |
|---|---|
| `margins/_session.py` (incl. `from_posterior:720`) | `estimators/` nouns; `from_posterior` dropped (recorded above) |
| `margins/_inference_glue.py` | `_engine/_queries.py` (R2) |
| `_inference/_dispatch.py` fallback policy (`:36` κ-flip caller, `:64–98` non-differentiable reroute) | doctrine dispatch (R3); refusals at compile/query, never reroutes |
| `_result/_margins.py` (`MarginsResult`, 2,435 lines) | new result (R4); its interval/test/sup-t math is review-then-moved there |
| Facade internals: `estimators/_base.py` delegation body, `_extract_legacy_kwargs`, `_resolve_model`, `_graphresult.py` wrapper body | R5–R6 |
| `_engine/_banks.py` `RetentionPolicy` byte-budget | amended verdict 3 retention (R1) |
| Legacy exports in `__init__.py` (`Margins`, `MarginsResult`, …) | new surface only |

### Migration-map closure (req. §7 under a clean break)

With legacy deleted, every req. §7 row MUST have a new home **or an
explicitly recorded drop** — there is no "keep using `Margins`" escape.
Deltas from the 06-10 reading:

- `formula=`/`data=` (`from_formula`) → **spec-form `outcome=` lands in
  R6** (it was deferrable when `Margins` survived; it is not now).
- `from_posterior` → dropped (recorded above).
- `strict=` → gone; the constructor doctrine is strict mode (unchanged).
- `diagnostics=` → gone; always-on, severity-routed (unchanged).
- Everything else: per the req. §7 table, now mandatory in R6's
  coverage checklist (R6 acceptance includes a row-by-row audit).

---

## 3. Module map (end state)

```
pymargins/
  steps/__init__.py        # wiring verbs (kept)
  _graph/_node.py          # Node, hashing (kept)
  _graph/_plan.py          # Plan (kept)
  _graph/_compile.py       # C1/C2 — C2 rebuilt (R5)
  _soundness/              # constants, predicates + SOUNDNESS_ROWS (R5)
  _engine/_seeds.py        # deterministic seed tree — reviewed derivations,
                           #   own regression goldens (R1)
  _engine/_banks.py        # BankSet, amended retention (R1)
  _engine/_queries.py      # query spec → h/h_factory/metadata (R2)
  _engine/_execute.py      # doctrine dispatch + executor (R3)
  estimators/_base.py      # noun base, rebuilt on the engine (R6)
  estimators/__init__.py   # GComputation (R6); IPW/AIPW in 0.6.0
  _result/_graphresult.py  # real result, §7.1 field set (R4)
  _result/_intervals.py    # interval/test/sup-t math, review-then-moved (R4)
  _result/_pooling.py      # pool_imputations re-pointed (R4)
  margins/, _result/_margins.py   # FROZEN, test-reachable only, until R7

tools/oracle/              # R scripts that generate committed goldens (R1)
tests/oracle/              # analytic suite + R-golden comparisons (R1)
tests/anchor/              # legacy dual-run corroboration — construction
                           #   window only, retired at R7
tests/golden/              # new-engine regression goldens, recorded at R7

notes/061126_legacy_defect_ledger.md   # oracle-visible ≤0.3.0 defects
notes/061126_test_port_ledger.md       # non-mechanical port dispositions
```

---

## 4. The validation protocol (I1′ enforcement; replaces the anchor protocol)

Five layers. Authority order: **analytic > consensus of independent
oracles > single R oracle > legacy corroboration**. Disagreement between
layers beyond recorded tolerance ⇒ stop and ask (I4); the resolution is a
ledger entry, never a silent tolerance bump.

1. **Analytic suite** (`tests/oracle/test_analytic.py`) — exact identities
   with closed forms: OLS AME ≡ β̂; identity-link predictions; logit AME
   = mean(p(1−p))·β; delta-method SEs hand-derivable on tiny fixtures;
   weight/cluster algebra on toy data. Exact: `np.array_equal` where the
   derivation is shared, `rtol ≤ 1e-12` where only float association
   order differs. Highest authority; costs nothing to keep forever.
2. **R reference suite** — `tools/oracle/*.R` scripts generate goldens
   committed under `tests/oracle/golden/` (JSON);
   `tests/oracle/test_r_golden.py` compares. The default test lane never
   shells out to R; the weekly slow lane MAY re-run the scripts to detect
   upstream drift. Oracles and their targets:
   - `marginaleffects` (0.32.0 installed): predictions, slopes
     (AME/MEM/`at`), comparisons/contrasts, hypotheses — estimates,
     delta-method SEs, CIs.
   - `survey` (4.5 installed): `svyglm` Taylor-linearized SEs for the
     survey-design paths.
   - `sandwich` (3.1.1 installed): HC/cluster vcov alignment.
   - Installed on demand when their cases land: `multcomp` (sup-t /
     max-t adjusted intervals), `survRM2` (RMST), `emmeans`
     (corroboration), `margins` 0.3.28 (Leeper; corroboration only —
     superseded upstream by `marginaleffects`).
   Local stack today: R 4.6.0.
   **Alignment discipline (per golden, recorded in the file):** the R
   call verbatim; package versions; vcov convention (HC type, df
   correction); CI convention (z vs t, df); fit-convergence settings on
   both sides; tolerance + one-line justification. Defaults: estimates
   `rtol 1e-6`, SEs/CIs `rtol 1e-5`. Cross-fitter drift (statsmodels vs R
   IRLS, ~1e-8 on β̂) is handled by *tightening convergence in the oracle
   script*, never by loosening tolerance.
3. **Legacy corroboration** (`tests/anchor/`, construction window only) —
   dual-run new engine vs frozen in-tree legacy across the matrix
   (adapter fixtures × delta/sim/boot × predict/dydx/contrasts/evaluate ×
   {plain, survey, cluster, block, matching, transforms}),
   `np.array_equal`, with the localization diagnostic (max |a−b|, dtypes,
   shapes, strides) on failure. **Role under rev. 2: instrument, not
   judge.** Perfect byte-alignment (shared kernels) makes it the fastest
   divergence localizer; layers 1–2 arbitrate who is wrong. Legacy-wrong
   ⇒ defect ledger + an expected-divergence marker on that cell;
   new-wrong ⇒ fix. Retired (not converted) at R7.
4. **Regression goldens** (`tests/golden/`, recorded at R7 from the
   *validated* new engine) — estimates/SEs/CIs/draws per matrix cell plus
   the seed-derivation goldens from R1. Self-recorded is sound here
   because correctness authority lives in layers 1–2; these protect
   against *regression*, including the determinism of resampling streams
   no external oracle can check. Regeneration requires a recorded
   justification (same discipline as the plan-hash recipe bump).
5. **Calibration lane** (weekly, slow-marked) — sim/boot SE vs delta SE
   agreement within Monte Carlo error on smooth cases; the req. §9
   coverage simulations. This is the only validation resampling
   *statistics* can get, and it is genuinely statistical — failures here
   are method bugs, not float noise.

Lifecycle: construction runs layers 1–3 (1–2 are the gate, 3 the
debugger); the R7 parity gate per §5; the deletion commit records layer 4
and retires layer 3; steady state = layers 1, 2, 4 in the default lane,
5 weekly.

---

## 5. Workstreams (order: R0 → R1 → R2 → R3 → (R4 ∥ R5) → R6 → R7 → R8)

### R0 — Checkpoint

Commit the working tree per §1 (including this plan revision). Confirm
`pytest -m "not slow"` and `ruff` green at the checkpoint (install ruff in
the dev env — the 06-10 audit found it missing, so the "ruff before done"
rule was unenforceable).

### R1 — Validation harness, seeds, banks (replaces W2.4)

- **Oracle harness first** (§4 layers 1–2): build the analytic suite and
  the R-golden pipeline, then run both against the *current tree* (the
  facade, i.e. effectively legacy numbers). This baselines legacy before
  the engine swap: every failure is triaged to the defect ledger as
  *legacy/kernel defect* (defines a corrected target for the rewrite) or
  *facade defect* (no fix needed — the facade dies at R5–R6; note and
  move on). Initial matrix: {OLS, logit, probit, Poisson} ×
  {predict, dydx(AME/`at`), contrasts, evaluate} ×
  {nonrobust, HC1, cluster} × {unweighted, weights, survey}. sup-t and
  RMST cells land when `multcomp`/`survRM2` are installed.
- `_engine/_seeds.py`: review the existing derivations
  (`_generate_resample_indices` / `_generate_simulation_draws` — plain
  numpy RNG engineering) per I3″; keep them if they pass (expected), or
  redesign **now, once** — this is the only window (§7 trap 2). Add
  spawn-tree derivation for branch/task independence (deterministic under
  shuffled task order). `tests/test_engine_seeds.py` records the
  *reviewed* derivations' outputs for three seeds × {iid, cluster, block,
  stratified} × sim draws as regression goldens (§4 layer 4 discipline —
  these are the one golden set recorded before R7, because nothing
  external can validate streams). Verify wrapper-vs-derivation agreement
  before recording; same-seed streams MAY differ from 0.3.0 → R8 note.
- `_engine/_banks.py`: `BankSet` per (estimator, branch); keys
  `(plan_hash, branch_id, seed)`. Retention per amended verdict 3: no fan
  ⇒ retain refit adapters (today's replay semantics); fan ⇒ products-only
  (0.5.0 activates this). **Delete `RetentionPolicy` and its
  byte-budget.** `BankRetentionError` message: issue queries together;
  re-run is deterministic (same seed tree).
- Acceptance: oracle suites green against the current tree (or failures
  ledgered with disposition); seed/bank goldens green.

### R2 — Query layer (`_engine/_queries.py`)

- Move (not copy) the query-construction logic out of
  `margins/_inference_glue.py` semantics: a session-free function per query
  kind (`predict/dydx/eyex/eydx/dyex/contrasts/evaluate/rmst/wtp`) mapping
  (query spec, adapter, Plan) → (`h`, `h_factory`, estimand metadata,
  `InferenceConfig`). Consume `margins/_estimands.py` / `_atoms.py` in
  place — **their citation review (I3″) happens here**, at first
  consumption; they relocate at R7. `wtp` = declared ratio estimand via
  the evaluate path (design §4.8), not result composition.
- The `InferenceConfig` it builds is doctrine-shaped: `kappa_threshold=inf`
  always (there is no other mode), method = the Plan's resolved method.
- **This module is where the facade's bugs lived as a class; it gets the
  heaviest validation coverage.** Every query kind × every option that
  changes input construction (at, scale/phi, weights, vcov spec) covered
  by §4 layers 1–2 where an oracle reaches it, and corroborated by
  layer 3 dual-run, before R3 builds on it.
- Note: legacy `_inference_glue.py` stays untouched and running (I6); R2
  is a parallel implementation whose equality is proven, not assumed —
  the construction-window duplication is accepted and is exactly what the
  validation stack polices, then R7 deletes the legacy copy.

### R3 — Doctrine dispatch and executor (`_engine/_execute.py`)

- Calls `_run_delta` / `_run_simulation` / `_run_bootstrap` kernels
  directly (kernels reviewed per I3″ at this consumption). **No fallback
  branches exist**: method is already resolved (Plan); a
  non-JAX-differentiable `h` under `method="delta"` is a `CompileError`
  with the steer to simulation (design §6.1) raised at compile (posture
  probe) or at query time (`evaluate` compose) — never a reroute. κ
  computed per query, recorded on the result, never steering (design
  §5.2).
- Owns bank build/replay through R1: M=1 path calls the R1-reviewed
  derivations, always. Replicate failure policy per §6.7 constants.
- Survey design and cluster/block declarations read from the `input`
  node's params (the steps fixes of 06-11) and routed to both consumers:
  VarianceScheme-side (kernel args) and resampler-side. Trap 5 of the old
  plan applies: the design always drives resampling even when analytic Σ
  is dead code. Survey SEs validate against `svyglm` (§4 layer 2).

### R4 — The real result (`_result/_graphresult.py` rebuilt)

- Field set per req. §6: estimates + labels; per-method payload (delta:
  gradient + Σ̂; sim/boot: draws); **stored ψ^h when tier-1** (computed
  `adapter.influence() @ ∇h` — the W1.3-pinned identity); plan copy;
  population notes; diagnostics (κ, G, ESS, replicate failures); declared
  scale/level/ci. No live references to estimator/session in any method.
- **Review-then-move** the interval/test/sup-t math from
  `_result/_margins.py` into `_result/_intervals.py`: read-through with
  citations (I3″), oracle coverage (CIs and hypothesis tests against
  `marginaleffects`; sup-t against `multcomp` max-t when installed; plain
  z/t against scipy closed forms), then the new result consumes it.
  Rewrite only on review failure, ledgered. Legacy keeps its own copy
  untouched until R7 deletes it (construction-window duplication accepted
  — the validation stack polices divergence, and the deletion commit ends
  it).
- Doctrine surface (signatures frozen from the validated facade):
  `conf_int(correction=None|"bonferroni"|"sidak"|"sup-t")` — no `level=`,
  and the `TypeError` for a `level=` attempt carries the re-declaration
  steer; corrections allocate the declared budget and only widen; `test`,
  `joint_test`, `summary` (footer: plan hash, population note, κ),
  `to_frame/to_latex/to_html`, `outcome()`, `scaled()`, `contrast()`,
  `pairwise_contrasts()`, `influence()` (attribute read of stored ψ^h),
  `to_disk`/`from_disk` lossless (drop the dead `format=` param until a
  second format exists).
- `pool_imputations` re-pointed: consumes and returns the new result type
  (`_rubin_pool` arithmetic citation-reviewed, expected kept); its
  existing tests port in R7 — their assertions derive from scipy t-dist
  formulas in-test, so they keep their own authority.

### R5 — Compile C2 for real (replaces W2.5 internals) + soundness spine

- C2 per req. §3: point execution through the node executor (topological
  stage application); template fingerprint check (refusal names both
  fingerprints; no exception-swallowing skip path); `method="auto"`
  resolution = differentiability probe + κ pre-pass via `session_kappa`
  on tier-1/autodiff, `delta_simulation_disagreement` on tier-2 (design
  §11.8), reason recorded; adequacy predicates (tail counts, G, lonely
  PSU — now reachable, the design object flows); population notes.
- `SOUNDNESS_ROWS` registry: every design-§6 row enumerated with a stable
  id, implemented rows mapped, the rest `None`;
  `tests/test_soundness_predicates.py` iterates the registry. Add the
  req-§1 lattice-consistency test (tier vs `supported_inference_methods`,
  parametrized over the adapter registry).
- Constructor strictness: unknown kwargs are `TypeError` (no `**kwargs`
  swallowing — a pre-registration point rejects typos). Callable scale
  pairs accepted (`scale=(phi, phi_inv)`), fingerprinted by
  source/qualname, `unhashable_callable` marked when not retrievable.
- `test_plan_hash_golden` (fixed toy plan → recorded constant; recipe-bump
  rule enforced).

### R6 — Nouns on the new engine (replaces W2.7)

- `GComputation` constructor per design §4.5 (positional model | wiring;
  `outcome=` template **and spec form** — spec form is now mandatory
  coverage, see §2 migration closure: fit via the formula API on the
  wiring's point-execution output, family from kwarg). Compile (C1+C2) →
  Plan → owns `BankSet` → queries through R2/R3. No `Margins` import (I6).
- Noun query scoping per design §4.2; `est.joint()` → stub naming 0.5.0.
- `match` + row-filter stages in one wiring: **keep the C1 refusal** for
  0.4.0 (steer: lands with the fan engine in 0.5.0). The legacy mutual
  exclusivity encoded an unexamined replicate-loop ordering question
  (rematch vs filters); enabling it is statistical design work, not
  plumbing (I4) — revisit in W3.1 where the branch executor is built.
- Acceptance: **req. §7 row-by-row coverage audit** (every row → new home
  or recorded drop) + the §4 layer-1/2 matrix green end-to-end through
  the nouns + the layer-3 corroboration matrix green (minus ledgered
  expected divergences).

### R7 — Test porting, parity gate, deletion (the big one)

Measured scope (2026-06-11): **68 of 91 test files reference `Margins`.**
A survey of assertion styles shows the suite carries essentially no
hardcoded engine-output goldens — numeric expectations are
*reference-derived in-test* (statsmodels native predictions, scipy
formulas, FD↔autodiff agreement, internal-consistency identities). That
authority is independent of the legacy engine and survives the teardown
untouched. The port is therefore a triage, not a verbatim copy:

- **(a) Reference-derived assertions** (the bulk): mechanical surface
  translation (`Margins(model, …).predict()` →
  `GComputation(model, …).predict()`, `steps.input` for dependence
  kwargs, ci/B/seed spellings per req. §7); assertions untouched — their
  authority is the in-test reference.
- **(b) Engine-derived / self-anchored expectations** (numbers that are
  ultimately old-engine output): if the §4 oracle matrix already covers
  the case, drop as redundant; if the coverage is unique, re-anchor the
  expectation through an oracle (analytic or R golden); if no oracle
  reaches it, keep as a legacy-corroborated regression value with a
  ledger note saying exactly that.
- **(c) Semantic-change tests** (κ-flip, fallback-warning, `strict=`,
  `conf_int(level=)`, `from_posterior`): replaced by
  doctrine-refusal-test / ported-with-new-spelling / dropped-with-reason.
- Every non-mechanical disposition recorded in
  `notes/061126_test_port_ledger.md`. The lethal porting failure mode,
  restated for rev. 2: **never regenerate an expectation from the new
  engine without oracle authority** — that anchors the suite to itself.
- A ported assertion that fails is a new-engine bug, a genuine semantic
  change (→ category c), or a legacy defect now fixed (→ defect ledger;
  update the assertion *to the oracle-correct value*, cited).
- **Parity gate:** §4 layers 1–2 green through the nouns + ported suite
  green + layer-3 corroboration matrix green except ledgered expected
  divergences.
- Then, one commit sequence: record `tests/golden/` regression goldens
  from the validated new engine (§4 layer 4) → retire `tests/anchor/` →
  delete `margins/_session.py`, `_inference_glue.py`,
  `_result/_margins.py`, dispatch fallback code, legacy exports →
  relocate `_estimands.py`/`_atoms.py` to their final home → full suite +
  ruff green.

### R8 — Docs and the 0.4.0 release

- CHANGELOG rewritten honestly as a **breaking release**: first line says
  `Margins` is removed; the req. §7 map rendered as the migration table;
  `from_posterior` removal noted with the design §3.9 rationale;
  **a corrections section** — every defect-ledger entry where 0.4.0's
  numbers intentionally differ from ≤0.3.0, each with its oracle evidence;
  **a reproducibility note** — same-seed sim/boot draw streams may differ
  from 0.3.0 (rev.-2 amendment to verdict 5).
- Docs: expand the two stubs (design §2 condensed; worked examples
  8.1–8.3); plan/pre-registration guide (level/ci doctrine);
  `kappa_fallback.md` rewritten **now** (decide-once is the only behavior
  — this moves up from the old Phase 5); `session_precommitment.md`
  superseded by the plan doc; the explanations doc gains a short "how
  pymargins is validated" section (the §4 stack — analytic + R oracles is
  a credibility asset; say it).
- Release gates: full suite + oracle suite + regression goldens + ruff +
  the weekly slow lane (calibration sims) green at the tag.

---

## 6. Deltas to Phases 3–5 of the 061026 plan (which otherwise stand)

- **W3.2:** `pool_imputations` already re-pointed at R4; the "shim +
  golden tests for the old signature" language reduces to "existing
  ported tests stay green."
- **W3.1/W3.2 fan × view refusals:** unchanged, but there is no legacy
  mode to exempt — refusals are unconditional.
- **W4.4:** migration guide shrinks to the req. §7 table + the 0.4.0
  breaking-release notes (no `compose_results` era to migrate from on the
  new surface; `wtp` worked example stays).
- **Phase 5 (1.0):** deprecation flip is moot. 1.0 = surface stability
  declaration + the soundness-registry completeness check (no `None`
  rows) + docs pass.
- **Req. §9 anchor harness:** superseded by §4 (rev. 2). The §9
  statistical-correctness and performance requirements stand.

## 7. Traps

1. **Construction-window fork.** Until R7, interval math and query glue
   exist twice (frozen legacy + new). The §4 stack is the only thing
   policing divergence — do not skip oracle/corroboration coverage for a
   query kind "temporarily."
2. **Seed-derivation churn.** The derivations are unfrozen (rev. 2), but
   every change invalidates recorded goldens and user-visible
   reproducibility. Redesign happens once, at R1, with the review in
   hand — or not at all. Opportunistic mid-build "improvements" to RNG
   derivations are reverted on sight.
3. **Porting self-anchor.** An expectation regenerated from the new
   engine without oracle authority silently anchors the suite to itself.
   Every non-mechanical disposition goes through the ledger; oracle-traced
   beats verbatim-carried beats regenerated.
4. **No delegation to legacy (I6).** The facade failure mode. New-engine
   modules import kernels and atoms, never the session/glue/result.
5. **Scope fence.** No fans, no IPW/AIPW, no result algebra, no new
   estimator nouns in 0.4.0, however natural mid-rewrite. The rewrite's
   only deliverable is: oracle-validated numbers, real engine,
   doctrine-native, legacy gone.
6. **Oracle misalignment.** Comparing quantities under different vcov
   conventions (HC type, df correction), CI conventions (z vs t), or
   fit-convergence settings produces false defects — or worse, false
   passes after a tolerance loosening "to make it work." The §4 alignment
   discipline (record the full spec per golden) is mandatory; tolerance
   bumps require a recorded reason.
7. **Oracle worship.** R packages have defects too, and conventions that
   differ legitimately (df choices, profile vs Wald). Authority order:
   analytic > consensus of independent oracles > single R oracle >
   legacy. Oracles disagreeing beyond tolerance is a stop-and-ask, never
   a coin flip.
8. **Don't reopen settled design verdicts mid-build.** A genuine design
   flaw found during the rewrite becomes a recorded amendment in the
   notes (§0's table is the template — rev. 2 itself is the mechanism in
   action), never an in-flight swerve.
