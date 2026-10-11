from minivlm.config import Config, load_yaml


def test_dataclass_defaults_match_config_yaml():
    assert Config.from_yaml(load_yaml()) == Config()


def test_from_yaml_ignores_keys_that_are_not_run_settings():
    cfg = Config.from_yaml({"model": {"llm": "x"}, "train": {"lr": 0.5, "save_every": 7},
                            "data": {"max_text_tokens": 9, "n_total": 3}})
    assert (cfg.llm, cfg.lr, cfg.max_text_tokens) == ("x", 0.5, 9)
