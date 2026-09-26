"""V5 option wiring and shared defaults for GPU experiment entrypoints."""
from __future__ import annotations


def option_specs():
    # flag, destination, config key, type, default, choices
    old = [
        ('--sae-ragged-engine', 'sae_ragged_engine', 'ragged_decoder_engine', str, 'triton', ['openai','triton','torch_reference']),
        ('--sae-ragged-main-compute', 'sae_ragged_main_compute', 'ragged_main_compute', str, 'sparse', ['sparse','auto','local_dense','compact_dense']),
        ('--sae-ragged-aux-compute', 'sae_ragged_aux_compute', 'ragged_aux_compute', str, 'sparse', ['sparse','auto','local_dense','compact_dense']),
        ('--sae-ragged-wgrad-split', 'sae_ragged_wgrad_split', 'ragged_wgrad_split', int, 1, [1,2,4,8]),
        ('--sae-ragged-index-backend', 'sae_ragged_index_backend', 'ragged_index_backend', str, 'sort', ['sort','histogram']),
        ('--sae-ragged-openai-page-k', 'sae_ragged_openai_page_k', 'ragged_openai_page_k', int, 512, [32,64,128,256,512]),
        ('--sae-ragged-openai-workspace-mib', 'sae_ragged_openai_workspace_mib', 'ragged_openai_workspace_mib', int, 64, None),
        ('--sae-ragged-openai-forward', 'sae_ragged_openai_forward', 'ragged_openai_forward', str, 'bucketed', ['bucketed','coo']),
        ('--sae-auxk-selection', 'sae_auxk_selection', 'auxk_selection', str, 'auto', ['auto','legacy']),
        ('--sae-auxk-decoder', 'sae_auxk_decoder', 'auxk_decoder_backend', str, 'auto', ['auto','local_dense','compact_dense']),
        ('--sae-auxk-complement', 'sae_auxk_complement', 'auxk_complement', str, 'auto', ['auto','off']),
    ]
    modes = ['inherit','sparse','local_dense','compact_dense','auto']
    result = list(old)
    for branch in ('main','aux'):
        for part in ('compute','forward','dvalues','dweight'):
            flag = f'--sae-{branch}-{part}'
            result.append((flag, flag[2:].replace('-','_'), f'v5_{branch}_{part}', str, 'inherit', modes))
        flag = f'--sae-{branch}-dense-threshold'
        result.append((flag, flag[2:].replace('-','_'), f'v5_{branch}_threshold', int, 512, None))
        for part in ('forward','dvalues','dweight'):
            flag = f'--sae-{branch}-{part}-threshold'
            result.append((flag, flag[2:].replace('-','_'), f'v5_{branch}_{part}_threshold', int, None, None))
    result += [
        ('--sae-aux-k','sae_aux_k','auxk',int,None,None),
        ('--sae-topk-ties','sae_topk_ties','topk_tie_policy',str,'stable_id',['stable_id','torch_tp1']),
        ('--sae-dispatch-k-metric','sae_dispatch_k_metric','v5_k_metric',str,'mean',['mean','max']),
        ('--sae-compact-max-ratio','sae_compact_max_ratio','v5_compact_max_ratio',float,.5,None),
        ('--sae-compact-min-density','sae_compact_min_density','v5_compact_min_density',float,.25,None),
        ('--sae-compute-workspace-mib','sae_compute_workspace_mib','v5_workspace_mib',int,128,None),
    ]
    return result


def add_v5_arguments(parser):
    named = parser._option_string_actions
    for flag, dest, key, typ, default, choices in option_specs():
        if flag in named:
            # Prior v4 optional runner integration is allowed. Do not duplicate
            # or replace defaults on an existing argparse action.
            continue
        parser.add_argument(flag, dest=dest, type=typ, default=default, choices=choices,
                            help=f'SAE config {key}; inherit preserves the V4 route.' if default=='inherit'
                            else f'SAE config {key}. V5 thresholds use actual LOCAL entries, not global K/TP.')
    for branch in ('main','aux'):
        parser.add_argument(f'--sae-{branch}-backward', default='inherit',
                            choices=['inherit','sparse','local_dense','compact_dense','auto'],
                            help='Convenience default for both dvalues/dweight; explicit stage flags take precedence.')
    if '--sae-topk-backend' in named:
        action = named['--sae-topk-backend']
        if action.choices is not None and 'sharded_ragged' not in action.choices:
            action.choices = [*action.choices, 'sharded_ragged']
    parser.add_argument('--sae-profile', action='store_true',
                        help='Run isolated SAE-only profiling instead of normal LLM/SAE training. Use --sae-profile --help.')


def set_experiment_policy_defaults(parser):
    # None distinguishes an omitted branch policy from an explicit inherit.
    parser.set_defaults(sae_ragged_engine='openai', sae_main_compute=None, sae_aux_compute=None)
    for branch, default in (('main', 'sparse'), ('aux', 'compact_dense')):
        parser._option_string_actions[f'--sae-{branch}-compute'].help += (
            f' Experiment default: {default} for sharded_ragged, inherit otherwise.'
        )


def resolve_experiment_policy_defaults(args, argv):
    for branch, default in (('main', 'sparse'), ('aux', 'compact_dense')):
        dest = f'sae_{branch}_compute'
        if getattr(args, dest) is None:
            old_flag = f'--sae-ragged-{branch}-compute'
            explicit_v4 = any(token == old_flag or token.startswith(old_flag + '=') for token in argv)
            value = default if args.sae_topk_backend == 'sharded_ragged' and not explicit_v4 else 'inherit'
            setattr(args, dest, value)
    return args


def v5_config_kwargs(args, exclude=()):
    values = {}
    for flag, dest, key, typ, default, choices in option_specs():
        value = getattr(args, dest, default)
        values[key] = value
    for branch in ('main','aux'):
        bwd = getattr(args, f'sae_{branch}_backward', 'inherit')
        if bwd != 'inherit':
            for stage in ('dvalues','dweight'):
                key = f'v5_{branch}_{stage}'
                if values[key] == 'inherit':
                    values[key] = bwd
    enabled = any(values[f'v5_{branch}_{stage}'] != 'inherit'
                  for branch in ('main','aux') for stage in ('compute','forward','dvalues','dweight'))
    backend = getattr(args,'sae_topk_backend','sharded_ragged')
    if enabled and backend != 'sharded_ragged':
        raise ValueError('Use --sae-topk-backend sharded_ragged with V5 Main/Aux computation policies')
    return {k:v for k,v in values.items() if k not in exclude}
