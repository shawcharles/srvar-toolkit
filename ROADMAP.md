# Roadmap

## Minnesota Prior Work

- Keep `minnesota_legacy` as the default reproducibility path.
- Keep `minnesota_canonical` semantically stable. Do not silently redefine it.
- Use `minnesota_tempered` only as an explicit experimental mitigation for diagonal-SV cases
  where the canonical scale construction is too loose for high-variance equations.

## Future Principled Redesign

- Design a new named diagonal-SV Minnesota method that regularizes or otherwise rethinks the
  equation-scale propagation behind the current canonical variance ratios.
- Validate that redesign on a broader benchmark matrix, including crisis-window origin
  diagnostics and higher-dimensional panels.
- Ship any redesign as a new method name with explicit migration notes, rather than mutating the
  meaning of `minnesota_canonical`.
