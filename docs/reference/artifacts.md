# `srvar.artifacts`

## Artifact compatibility and security

New fit and forecast artifacts use schema version 1. They contain Unicode variable names and
numeric arrays only, so loading a new artifact does not require pickle deserialisation.

Artifacts produced before this format migration are pickle-backed. The loaders reject them by
default. If, and only if, you have verified an old artifact's source and integrity, use the
explicit compatibility option:

```python
from srvar.artifacts import load_run_dir

fit_res = load_run_dir("outputs/trusted_old_run", allow_legacy_pickle=True)
```

This option can execute pickle code. It is not appropriate for files received from an untrusted
or unknown source. NPZ parsing can still consume resources; this format change does not add
archive size limits or complete payload validation.

```{eval-rst}
.. automodule:: srvar.artifacts
   :members:
   :undoc-members:
   :show-inheritance:
```
