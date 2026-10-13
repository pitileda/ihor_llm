# Small Ukrainian story model for an RTX A2000 (6 GB).
# Run from the nanoGPT folder (train.py loads configurator.py and data/ relative to cwd):
#   cd nanoGPT && python train.py ../config/train_uk_stories.py
# Any value can be overridden on the command line, e.g. --max_iters=500 --compile=False

out_dir = 'out-uk-stories'
dataset = 'uk_stories'  # nanoGPT/data/uk_stories -> ../../data/uk_stories (prepare.py output)

# --- model: ~14M params with an 8k vocab ------------------------------------
# params ~= vocab*n_embd + 12*n_layer*n_embd^2 = 8192*384 + 12*6*384^2 ~= 3.1M + 10.6M
# vocab_size is read from data/uk_stories/meta.pkl
n_layer = 6
n_head = 6           # head size 64
n_embd = 384
block_size = 512     # one story is ~250-350 tokens, so the context sees 1-2 whole stories
bias = False
dropout = 0.1        # we train several epochs over the same synthetic stories

# --- batch: 32 x 512 x 4 = 65,536 tokens per iteration ----------------------
batch_size = 32
gradient_accumulation_steps = 4

# --- schedule ----------------------------------------------------------------
# Set max_iters from your data: epochs * train_tokens / 65,536.
# e.g. 100M train tokens x 3 epochs = 300M tokens -> ~4,600 iters (~1-2 h on the A2000).
max_iters = 5000
lr_decay_iters = 5000    # = max_iters
learning_rate = 1e-3     # small models tolerate a higher LR
min_lr = 1e-4            # learning_rate / 10
warmup_iters = 200
beta2 = 0.99
weight_decay = 0.1
grad_clip = 1.0

# --- evaluation / checkpoints -----------------------------------------------
eval_interval = 250
eval_iters = 100
log_interval = 10
always_save_checkpoint = False  # only save when val loss improves

compile = True  # torch.compile: slower start, ~1.5-2x faster training
