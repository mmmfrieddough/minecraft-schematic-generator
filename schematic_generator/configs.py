import itertools
import random


def handle_callable_functions(config):
    for key, value in config.items():
        if callable(value):
            config[key] = value()  # Call the function to get the value
    return config


def expand_configs(configs: list[dict]) -> list[dict]:
    random.seed(1337)
    expanded_configs = []
    for config in configs:
        keys = config.keys()
        combinations = list(itertools.product(*config.values()))
        for combination in combinations:
            expanded_config = dict(zip(keys, combination))
            expanded_config = handle_callable_functions(expanded_config)
            expanded_configs.append(expanded_config)
    return expanded_configs
