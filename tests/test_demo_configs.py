from srvar.config import load_config


def test_backtest_demo_config_loads() -> None:
    cfg = load_config("config/backtest_demo_config.yaml")
    assert cfg["prior"]["method"] == "minnesota_legacy"
    assert cfg["backtest"]["mode"] == "expanding"
