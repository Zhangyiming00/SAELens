#!/usr/bin/env python3
import argparse

from sae_lens import PretokenizeRunner, PretokenizeRunnerConfig


def parse_special_token(value: str):
    lowered = value.lower()
    if lowered == "none":
        return None
    if lowered in {"bos", "eos", "sep"}:
        return lowered
    return int(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pretokenize a text dataset with SAELens and save it locally."
    )
    parser.add_argument("--tokenizer-name", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--column-name", default="text")
    parser.add_argument("--context-size", type=int, default=2048)
    parser.add_argument("--num-proc", type=int, default=4)
    parser.add_argument("--pretokenize-batch-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--begin-batch-token",
        default="none",
        type=parse_special_token,
        help="One of: none, bos, eos, sep, or an integer token id.",
    )
    parser.add_argument(
        "--begin-sequence-token",
        default="none",
        type=parse_special_token,
        help="One of: none, bos, eos, sep, or an integer token id.",
    )
    parser.add_argument(
        "--sequence-separator-token",
        default="none",
        type=parse_special_token,
        help="One of: none, bos, eos, sep, or an integer token id.",
    )
    parser.add_argument(
        "--disable-concat-sequences",
        action="store_true",
        help="Disable concatenation and truncate each sample to context size.",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Keep output order instead of shuffling after tokenization.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = PretokenizeRunnerConfig(
        tokenizer_name=args.tokenizer_name,
        dataset_path=args.dataset_path,
        split=args.split,
        column_name=args.column_name,
        context_size=args.context_size,
        num_proc=args.num_proc,
        pretokenize_batch_size=args.pretokenize_batch_size,
        shuffle=not args.no_shuffle,
        seed=args.seed,
        begin_batch_token=args.begin_batch_token,
        begin_sequence_token=args.begin_sequence_token,
        sequence_separator_token=args.sequence_separator_token,
        disable_concat_sequences=args.disable_concat_sequences,
        save_path=args.save_path,
    )
    PretokenizeRunner(cfg).run()


if __name__ == "__main__":
    main()
