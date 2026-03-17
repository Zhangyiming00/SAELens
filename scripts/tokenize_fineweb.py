"""Tokenize fineweb-edu/sample/10BT with Llama-3.1-8B tokenizer, ctx=2048."""
from sae_lens.config import PretokenizeRunnerConfig
from sae_lens.pretokenize_runner import PretokenizeRunner

cfg = PretokenizeRunnerConfig(
    tokenizer_name='/data/models/Llama-3.1-8B',
    dataset_path='/data/fineweb-edu/sample/10BT',
    column_name='text',
    context_size=2048,
    split='train',
    shuffle=True,
    seed=42,
    begin_batch_token='bos',
    begin_sequence_token=None,
    sequence_separator_token=None,
    save_path='~/datasets/fineweb-edu-10BT_tokenized_llama31_ctx2048',
    num_proc=16,
    pretokenize_batch_size=4096,
    streaming=False,
)

runner = PretokenizeRunner(cfg)
ds = runner.run()
print(f'Done! Total sequences: {len(ds)}')
print(f'Total tokens: {len(ds) * 2048 / 1e9:.2f}B')
