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
or unknown source.

## Availability limits

Every artifact loader applies metadata-based limits before NumPy opens the archive: a maximum
archive size (512 MiB), member count (128), per-member uncompressed size (512 MiB), aggregate
uncompressed size (1 GiB), and ZIP expansion ratio (100:1). These defaults allow ordinary
research runs while bounding accidental or maliciously oversized inputs. The same checks apply
when `allow_legacy_pickle=True`; trusted legacy compatibility does not bypass them.

Use an immutable `ArtifactLoadLimits` value to set stricter or larger local bounds:

```python
from srvar.artifacts import ArtifactLoadLimits, load_fit_npz

limits = ArtifactLoadLimits(max_archive_bytes=64 * 1024 * 1024)
fit = load_fit_npz("outputs/fit_result.npz", limits=limits)
```

These are early availability controls, not complete payload validation. ZIP metadata is untrusted
and is used only to bound work before NumPy access: parsing the ZIP directory and subsequent
`.npy` headers can still consume resources, and a malformed `.npy` shape or header can request an
allocation inconsistent with its ZIP member size. The limits reduce this exposure but cannot
eliminate it; do not treat them as a safe way to load untrusted pickle-backed files.

```{eval-rst}
.. automodule:: srvar.artifacts
   :members:
   :undoc-members:
   :show-inheritance:
```
