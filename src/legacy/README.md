# Deprecated characterization baselines

This package preserves the previous implementation exclusively so regression
tests can describe its historical behavior. NeuroIris runtime, training,
calibration, and evaluation do not import it.

The old algorithms do not provide current inspection guarantees. They include
silhouette-only angular alignment, scalar whole-part geometry rejection, F2-led
threshold selection, image-level random splitting, repeated-frame voting, and
small-image statistical comparisons. Their tests are software characterization,
not evidence of industrial inspection accuracy.

Current APIs live in `src.vision`, `src.data`, `src.evaluation.calibration`, and
`src.inspection.session` / `src.inspection.fusion`. Original public module paths
re-export those modern APIs with migration instructions in their docstrings.
No old class names masquerade as compatible replacements.
