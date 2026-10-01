"""Shared SAE runner controls, canonical execution policy and legacy CLI migration."""
from __future__ import annotations

import argparse
import copy
from types import SimpleNamespace

from sae_lens.adaptive_sae import (
    COMPUTATIONS, REPRESENTATIONS, DEFAULT_REPRESENTATION,
    DEFAULT_COMPUTE_WORKSPACE_MIB, DENSE_COMPUTE_WORKSPACE_MIB,
    DEFAULT_OPENAI_WORKSPACE_MIB, computation, representation, stage_requests,
    validate_config,
)


def workspace_argument(value):
    if str(value).lower() == 'auto':
        return 'auto'
    try:
        budget = int(value)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError('Expected auto or an integer MiB budget in [1,4096]') from exc
    if not 1 <= budget <= 4096:
        raise argparse.ArgumentTypeError('Workspace must be in [1,4096] MiB')
    return budget


def compute_workspace_for_stages(modes):
    large = 'compact_dense' in modes or ('local_dense' in modes and len(set(modes)) > 1)
    return DENSE_COMPUTE_WORKSPACE_MIB if large else DEFAULT_COMPUTE_WORKSPACE_MIB


def resolve_workspace_defaults(args):
    """Resolve once at the CLI boundary; model/checkpoint budgets remain integers.

    Stage overrides take precedence over branch defaults. These measured
    starting budgets do not promise an optimal choice for every shape/GPU.
    Explicit integers are never enlarged or replaced.
    """
    if not hasattr(args, '_compute_workspace_auto'):
        args._compute_workspace_auto = getattr(args, 'sae_compute_workspace_mib', 'auto') == 'auto'
    if getattr(args, 'sae_compute_workspace_mib', 'auto') == 'auto':
        fields = getattr(args, '_execution_fields', None)
        if fields is not None:
            modes = stage_requests(SimpleNamespace(**fields))
        else:
            base = computation(getattr(args, 'sae_main_compute', None))
            if base == 'none':
                base = computation(getattr(args, 'sae_ragged_main_compute', 'sparse'))
            backward = computation(getattr(args, 'sae_main_backward', 'none'))
            modes = []
            for stage in ('forward', 'dvalues', 'dweight'):
                mode = computation(getattr(args, f'sae_main_{stage}', 'none'))
                if mode == 'none' and stage != 'forward':
                    mode = backward
                mode = base if mode == 'none' else mode
                modes.append({'compact': 'compact_dense', 'dense': 'local_dense'}.get(mode, mode))
        args.sae_compute_workspace_mib = compute_workspace_for_stages(modes)
    if getattr(args, 'sae_ragged_openai_workspace_mib', 'auto') == 'auto':
        args.sae_ragged_openai_workspace_mib = DEFAULT_OPENAI_WORKSPACE_MIB
    return args


def option_specs():
    # flag, destination, config key, type, default, choices
    old = [
        ('--sae-ragged-engine', 'sae_ragged_engine', 'ragged_decoder_engine', str, 'triton', ['openai','triton','torch_reference']),
        ('--sae-ragged-main-compute', 'sae_ragged_main_compute', 'ragged_main_compute', str, 'sparse', ['sparse','auto','local_dense','compact_dense']),
        ('--sae-ragged-aux-compute', 'sae_ragged_aux_compute', 'ragged_aux_compute', str, 'sparse', ['sparse','auto','local_dense','compact_dense']),
        ('--sae-ragged-wgrad-split', 'sae_ragged_wgrad_split', 'ragged_wgrad_split', int, 1, [1,2,4,8]),
        ('--sae-ragged-index-backend', 'sae_ragged_index_backend', 'ragged_index_backend', str, 'sort', ['sort','histogram']),
        ('--sae-ragged-openai-page-k', 'sae_ragged_openai_page_k', 'ragged_openai_page_k', int, 512, [32,64,128,256,512]),
        ('--sae-ragged-openai-workspace-mib', 'sae_ragged_openai_workspace_mib', 'ragged_openai_workspace_mib', workspace_argument, 'auto', None),
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
        ('--sae-compute-workspace-mib','sae_compute_workspace_mib','v5_workspace_mib',workspace_argument,'auto',None),
    ]
    return result


def add_v5_arguments(parser):
    named = parser._option_string_actions
    option_help = {
        '--sae-compute-workspace-mib':
            'Local decoder row-tile budget: auto (default) chooses 128 MiB, or 512 for Main compact/mixed stages. '
            'Also gates direct select-all ragged Aux. Explicit integer: 1..4096 MiB; not a total memory cap.',
        '--sae-ragged-openai-workspace-mib':
            'OpenAI sparse forward/dvalues page-row budget: auto (default) chooses 256 MiB. '
            'Explicit integer: 1..4096 MiB; excludes dweight sort/COO and whole-step memory.',
        '--sae-dispatch-k-metric':
            'Metric for compute=auto: mean uses actual local entries / rows; max adds a GPU scalar readback.',
        '--sae-compact-max-ratio':
            'For compute=auto, maximum compact-column / local-shard width ratio in [0,1]; default 0.5.',
        '--sae-compact-min-density':
            'For compute=auto, minimum entries / (rows * compact columns) in [0,1]; default 0.25.',
        '--sae-aux-k':
            'AuxK budget: omitted uses d_in//2, 0 disables AuxK, positive values cap selected dead features.',
    }
    for branch in ('main', 'aux'):
        option_help[f'--sae-{branch}-dense-threshold'] = (
            'For compute=auto, prefer sparse at or below this actual LOCAL K metric; '
            'above it, choose compact or dense from column width/density. Default 512.'
        )
        for stage in ('forward', 'dvalues', 'dweight'):
            option_help[f'--sae-{branch}-{stage}-threshold'] = (
                f'Override the {branch} auto threshold for {stage}; omitted inherits its branch threshold.'
            )
    for flag, dest, key, typ, default, choices in option_specs():
        if flag in named:
            # Prior v4 optional runner integration is allowed. Do not duplicate
            # or replace defaults on an existing argparse action.
            continue
        parser.add_argument(flag, dest=dest, type=typ, default=default, choices=choices,
                            help=option_help.get(flag) or (
                                f'SAE config {key}; inherit preserves the V4 route.' if default=='inherit'
                                else f'SAE config {key}. V5 thresholds use actual LOCAL entries, not global K/TP.'))
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
    return resolve_workspace_defaults(args)


def v5_config_kwargs(args, exclude=()):
    args = resolve_workspace_defaults(copy.copy(args))
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


LEGACY_REP = (
    "--sae-topk-backend",
    "--use-sparse-activations",
    "--no-use-sparse-activations",
    "--sae-ragged-main-compute",
    "--sae-ragged-aux-compute",
    "--sae-auxk-decoder",
)
LEGACY_OVERLAP = (
    "--multi-sae-distributed-architecture",
    "--multi-sae-tp-wavefront-schedule",
    "--multi-sae-tp-wavefront-max-live-hooks",
)


def supplied(argv, flag):
    return any(x == flag or x.startswith(flag + "=") for x in argv)


def add_execution_arguments(parser):
    add_v5_arguments(parser)
    set_experiment_policy_defaults(parser)
    # Old launch scripts remain readable, but these are no longer public knobs.
    parser.add_argument(
        LEGACY_OVERLAP[0],
        default="unified_multi_hook",
        choices=("legacy_per_hook_wrapper", "unified_multi_hook"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        LEGACY_OVERLAP[1],
        default="lazy",
        choices=("eager", "lazy", "bounded"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(LEGACY_OVERLAP[2], type=int, default=2, help=argparse.SUPPRESS)
    public = parser.add_argument_group("SAE storage, computation and TP overlap")
    for flag in (
        *LEGACY_REP,
        *LEGACY_OVERLAP,
        "--sae-ragged-engine",
        "--sae-sparse-decoder",
    ):
        action = parser._option_string_actions.get(flag)
        if action is not None:
            action.help = argparse.SUPPRESS
    for branch in ("main", "aux"):
        public.add_argument(
            f"--{branch}-storage",
            dest=f"sae_{branch}_representation",
            type=representation,
            choices=REPRESENTATIONS,
            default="none",
            help=f"TopK activation representation; none defaults to {DEFAULT_REPRESENTATION} (does not disable the branch).",
        )
        parser.add_argument(
            f"--sae-{branch}-representation",
            dest=f"sae_{branch}_representation",
            type=representation,
            choices=REPRESENTATIONS,
            default="none",
            help=argparse.SUPPRESS,
        )
        for stage in ("compute", "forward", "dvalues", "dweight", "backward"):
            action = parser._option_string_actions[f"--sae-{branch}-{stage}"]
            action.type = computation
            action.choices = COMPUTATIONS
            action.default = None
            help_text = (
                "Independent decoder computation; none defaults to "
                + ("sparse" if branch == "main" else "compact")
                + "."
                if stage == "compute"
                else "Stage override; none inherits branch computation (backward sets dvalues/dweight)."
            )
            action.help = argparse.SUPPRESS
            public.add_argument(
                f"--{branch}-{stage}",
                dest=action.dest,
                type=computation,
                choices=COMPUTATIONS,
                default=None,
                help=help_text,
            )
    parser.add_argument(
        "--sae-decoder-engine",
        dest="sae_ragged_engine",
        choices=("openai", "triton", "torch_reference"),
        default="openai",
        help="Kernel engine for sparse computation, independent of activation representation.",
    )
    public.add_argument(
        "--tp-overlap",
        dest="sae_tp_overlap",
        choices=("lazy", "bounded", "eager", "off"),
        default=None,
        help="TP overlap with early sharded AuxK selection (also for one hook); default bounded uses rolling encoder lookahead. Lazy/eager submit all main forwards first. Independent of optimizer overlap.",
    )
    public.add_argument(
        "--tp-overlap-max-live-hooks",
        dest="sae_tp_overlap_max_live_hooks",
        type=int,
        metavar="N",
        default=2,
        help="Live main-graph limit for rolling bounded TP overlap (default 2); full retained outputs are additional storage. Ignored by lazy/eager/off.",
    )
    parser.add_argument(
        "--sae-tp-overlap",
        dest="sae_tp_overlap",
        choices=("lazy", "bounded", "eager", "off"),
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--sae-tp-overlap-max-live-hooks",
        dest="sae_tp_overlap_max_live_hooks",
        type=int,
        default=2,
        help=argparse.SUPPRESS,
    )


def resolve_execution_arguments(args, argv):
    legacy = any(supplied(argv, f) for f in LEGACY_REP)
    explicit_rep = any(
        supplied(argv, f)
        for b in ("main", "aux")
        for f in (f"--sae-{b}-representation", f"--{b}-storage")
    )
    if legacy and explicit_rep:
        raise ValueError(
            "Do not mix legacy TopK/representation controls with --main-storage/--aux-storage"
        )
    if legacy:
        for branch in ("main", "aux"):
            for stage in ("compute", "forward", "dvalues", "dweight", "backward"):
                key = f"sae_{branch}_{stage}"
                v = getattr(args, key)
                if v is None and stage == "compute":
                    continue
                setattr(
                    args,
                    key,
                    {
                        "none": "inherit",
                        "dense": "local_dense",
                        "compact": "compact_dense",
                    }.get(v, v or "inherit"),
                )
        resolve_experiment_policy_defaults(args, argv)
        args._execution_fields = None
    else:
        fields = {}
        for branch in ("main", "aux"):
            fields[branch + "_representation"] = getattr(
                args, f"sae_{branch}_representation"
            )
            fields[branch + "_compute"] = (
                getattr(args, f"sae_{branch}_compute") or "none"
            )
            backward = getattr(args, f"sae_{branch}_backward") or "none"
            for stage in ("forward", "dvalues", "dweight"):
                value = getattr(args, f"sae_{branch}_{stage}") or "none"
                if stage != "forward" and value == "none":
                    value = backward
                fields[branch + "_" + stage] = value
        args._execution_fields = fields
        # Legacy fields remain only for reading old checkpoints and entrypoints.
        args.sae_topk_backend = "sharded_dense"
    old_overlap = any(supplied(argv, f) for f in LEGACY_OVERLAP)
    if old_overlap and (
        supplied(argv, "--sae-tp-overlap")
        or supplied(argv, "--sae-tp-overlap-max-live-hooks")
        or supplied(argv, "--tp-overlap")
        or supplied(argv, "--tp-overlap-max-live-hooks")
    ):
        raise ValueError("Do not mix legacy wavefront controls with --tp-overlap")
    if not old_overlap:
        args.sae_tp_overlap = args.sae_tp_overlap or "bounded"
        args.multi_sae_distributed_architecture = (
            "legacy_per_hook_wrapper"
            if args.sae_tp_overlap == "off"
            else "unified_multi_hook"
        )
        args.multi_sae_tp_wavefront_schedule = (
            "lazy" if args.sae_tp_overlap == "off" else args.sae_tp_overlap
        )
        args.multi_sae_tp_wavefront_max_live_hooks = args.sae_tp_overlap_max_live_hooks
    if args.sae_tp_overlap_max_live_hooks < 1:
        raise ValueError("TP overlap max-live-hooks must be positive")
    resolve_workspace_defaults(args)
    execution_config_kwargs(args)  # Validate before loading a model or starting workers.
    return args


def execution_config_kwargs(args, exclude=()):
    if args._execution_fields is None:
        values = v5_config_kwargs(args)
    else:
        old = copy.copy(args)
        for branch in ("main", "aux"):
            for stage in ("compute", "forward", "dvalues", "dweight", "backward"):
                setattr(old, f"sae_{branch}_{stage}", "inherit")
        values = v5_config_kwargs(old)
        values.update(args._execution_fields)
    validate_config(SimpleNamespace(topk_backend=args.sae_topk_backend, **values))
    return {key: value for key, value in values.items() if key not in exclude}
