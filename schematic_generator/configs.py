import itertools
import random


def handle_callable_functions(config, include_random_seed):
    for key, value in config.items():
        if callable(value) and (include_random_seed or key != 'random_seed'):
            config[key] = value()  # Call the function to get the value
    return config


def expand_configs(configs: list[dict]) -> list[dict]:
    expanded_configs = []
    for config in configs:
        keys = config.keys()
        combinations = list(itertools.product(*config.values()))
        for combination in combinations:
            expanded_config = dict(zip(keys, combination))
            expanded_config = handle_callable_functions(expanded_config, False)
            if 'position_offsets' in expanded_config:
                region_size = min(expanded_config['region_size'])
                if expanded_config['shape_type'] == 'cube':
                    shape_size = expanded_config['side_length']
                elif expanded_config['shape_type'] == 'sphere':
                    shape_size = expanded_config['radius'] * 2 + 1
                else:
                    raise ValueError(
                        "position_offsets is only supported for shape_type 'cube'")
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
                    offset_config = handle_callable_functions(
                        offset_config, True)
                    expanded_configs.append(offset_config)
            else:
                expanded_configs.append(expanded_config)
    return expanded_configs
