def test_import_srvar() -> None:
    import srvar  # noqa: F401


def test_cli_main() -> None:
    from srvar.cli import main

    assert main([]) == 0
