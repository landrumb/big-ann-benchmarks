"""writes a config to config.yaml for each ablation/dataset pair"""

import os
import yaml
import copy


build_args = [
            #  {"cluster_size": 5000, 
            #   "T": 8,
            #   "cutoff": 10000,
            #   "max_iter": 10,
            #   "weight_classes": [100000, 400000],
            #   "build_params": [{"max_degree": 8,
            #                     "limit": 200,
            #                     "alpha": 1.175},
            #                   {"max_degree": 10,
            #                    "limit": 200,
            #                    "alpha": 1.175},
            #                   {"max_degree": 12,
            #                    "limit": 200,
            #                    "alpha": 1.175}],
            #   "bitvector_cutoff": 10000,
            #   "materialize_joins": "True",
            #   "sorted_queries": "True"
            # },
            {"cluster_size": 5000, 
              "T": 180,
              "cutoff": 10000,
              "max_iter": 10,
              "weight_classes": [100000, 400000],
              "build_params": [{"max_degree": 6,
                                "limit": 200,
                                "alpha": 1.175},
                              {"max_degree": 8,
                               "limit": 200,
                               "alpha": 1.175},
                              {"max_degree": 10,
                               "limit": 200,
                               "alpha": 1.175}],
              "bitvector_cutoff": 10000,
              "materialize_joins": "True",
              "sorted_queries": "True"
            }]

query_args = [
             {"target_points": 5000,
              "tiny_cutoff": 30000,
              "beam_widths": [85, 50, 95],
              "search_limits": [500, 500, 500]
              },
            #   {"target_points": 7500,
            #    "tiny_cutoff": 35000,
            #    "beam_widths": [55, 55, 55]
            #    },
            #    {"target_points": 5000,
            #   "tiny_cutoff": 28000,
            #   "beam_widths": [90, 57, 90],
            #   "search_limits": [500, 500, 500]
            #   },
            #   {"target_points": 15000,
            #    "tiny_cutoff": 100000,
            #    "beam_widths": [60, 60, 60]
            #   },
            #   {"target_points": 15000,
            #    "tiny_cutoff": 60000,
            #    "beam_widths": [90, 90, 90]
            #   },
            #   {"target_points": 15000,
            #    "tiny_cutoff": 100000,
            #    "beam_widths": [90, 90, 90]
            #   },
            #   {"target_points": 15000,
            #    "tiny_cutoff": 60000,
            #    "beam_widths": [50, 50, 50]
            #   },
            #   {"target_points": 15000,
            #    "tiny_cutoff": 100000,
            #    "beam_widths": [50, 50, 50]
            #   },
            #   {"target_points": 15000,
            #    "tiny_cutoff": 60000,
            #    "beam_widths": [40, 40, 40]
            #   },
            #   {"target_points": 15000,
            #    "tiny_cutoff": 100000,
            #    "beam_widths": [40, 40, 40]
            #   }
              ]

base_config = {
    'docker-tag': 'neurips23-filter-parlayivf',
    'module': 'neurips23.filter.parlayivf.parlayivf',
    'constructor': 'ParlayIVF',
    'base-args': ['@metric'],
    'run-groups': {
        'base': {
            'args': copy.deepcopy(build_args),
            'query-args': copy.deepcopy(query_args)
        }
    }
}

datasets = ['yfcc-10M', 'crawl', 'enron', 'gist', 'msong', 'audio', 'sift', 'uqv']


def generate_ablation(build_overrides={}):
    """returns the dict representing everything under the ablation name. elements in the build_overrides dict will be applied to construction """
    config = copy.deepcopy(base_config)
    for build in config['run-groups']['base']['args']:
        build.update(build_overrides)
    return config

def no_ablation(added_query_configs=[], added_build_configs=[]):
    config = copy.deepcopy(base_config)
    config['run-groups']['base']['args'] += added_build_configs
    config['run-groups']['base']['query-args'] += added_query_configs
    return config

def no_material_join(added_query_configs=[], added_build_configs=[]):
    config = generate_ablation({'materialize_joins': 'False'})
    added_build_configs = copy.deepcopy(added_build_configs)
    for build in added_build_configs:
        build['materialize_joins'] = 'False'
    config['run-groups']['base']['args'] += added_build_configs
    config['run-groups']['base']['query-args'] += added_query_configs
    return config

def no_sorted_queries(added_query_configs=[], added_build_configs=[]):
    config = generate_ablation({'sorted_queries': 'False'})
    added_build_configs = copy.deepcopy(added_build_configs)
    for build in added_build_configs:
        build['sorted_queries'] = 'False'
    config['run-groups']['base']['args'] += added_build_configs
    config['run-groups']['base']['query-args'] += added_query_configs
    return config

def no_bitvector(added_query_configs=[], added_build_configs=[]):
    config = generate_ablation({'bitvector_cutoff': 1_000_000_000})
    added_build_configs = copy.deepcopy(added_build_configs)
    for build in added_build_configs:
        build['bitvector_cutoff'] = 1_000_000_000
    config['run-groups']['base']['args'] += added_build_configs
    config['run-groups']['base']['query-args'] += added_query_configs
    return config

def no_weight_classes(added_query_configs=[], added_build_configs=[]):
    config = generate_ablation({'weight_classes': [1, 1]})
    added_build_configs = copy.deepcopy(added_build_configs)
    for build in added_build_configs:
        build['weight_classes'] = [1, 1]
    config['run-groups']['base']['args'] += added_build_configs
    config['run-groups']['base']['query-args'] += added_query_configs
    return config


outer_dict = {}

for ds in datasets:
    outer_dict[ds] = {
        'parlayivf': no_ablation(),
        'parlayivf-no-material-join': no_material_join(),
        'parlayivf-no-sorted-queries': no_sorted_queries(),
        'parlayivf-no-bitvector': no_bitvector(),
        'parlayivf-no-weight-classes': no_weight_classes()
    }

with open('config.yaml', 'w') as f:
    yaml.dump(outer_dict, f, default_flow_style=True, sort_keys=False, indent=1)

print('done')
