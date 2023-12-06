import itertools
import random

from common import constants


def handle_callable_functions(config):
    for key, value in config.items():
        if callable(value):
            config[key] = value()  # Call the function to get the value
    return config


def expand_configs(configs: list[dict]) -> list[dict]:
    expanded_configs = []
    for config in configs:
        keys = config.keys()
        combinations = list(itertools.product(*config.values()))
        for combination in combinations:
            expanded_config = dict(zip(keys, combination))
            expanded_config = handle_callable_functions(expanded_config)
            if 'position_offsets' in expanded_config:
                if expanded_config['shape_type'] == 'cube':
                    # Generate offset combinations
                    shape_size = expanded_config['side_length']
                    region_size = min(expanded_config['region_size'])
                    offset_range = (region_size - shape_size) // 2
                    dimension_offsets = range(-offset_range - (1 if shape_size %
                                              2 != 0 else 0), offset_range + 1)
                    all_offset_combinations = list(itertools.product(
                        dimension_offsets, dimension_offsets, dimension_offsets))
                    num_offsets = min(len(all_offset_combinations),
                                      expanded_config['position_offsets'])
                    selected_offsets = random.sample(
                        all_offset_combinations, num_offsets)

                    # Add an expanded config for each offset
                    for offset in selected_offsets:
                        offset_config = expanded_config.copy()
                        offset_config['position_offset'] = offset
                        expanded_configs.append(offset_config)
                else:
                    raise ValueError(
                        "position_offsets is only supported for shape_type 'cube'")
            else:
                expanded_configs.append(expanded_config)
    return expanded_configs
