"""Reporting-only migration. No alteration of dead-age or optimizer state."""
import warnings


def reset_legacy_frequency_history(trainer):
    warnings.warn('V5: resetting legacy feature-frequency reporting accumulators; '
                  'Old DP-local/global reporting cannot safely be restored as V5 global reporting. '
                  'Weights, gradients, Adam, progress and dead-feature ages are unchanged.',
                  RuntimeWarning, stacklevel=2)
    for hook in trainer.hook_names:
        trainer.act_freq_scores_by_hook[hook].zero_()
        trainer.n_frac_active_samples_by_hook[hook] = 0


def migrate_frequency_history(trainer, version, *, saved_dp_size):
    """Reset only old DP>1 reporting accumulators; preserve DP1 history.

    No collectives, no parameter/optimizer/progress/dead-age mutation. Reset is
    intentionally conservative when only a rank0 aggregate checkpoint exists.
    """
    if int(version) < 2 and int(saved_dp_size) > 1:
        reset_legacy_frequency_history(trainer)
