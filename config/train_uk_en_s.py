# Model S: bilingual Ukrainian/English base model on real data (data/uk_en, ~1B tokens).
# Run from the nanoGPT folder:
#   cd nanoGPT && python train.py ../config/train_uk_en_s.py
# Any value can be overridden on the command line, e.g. --max_iters=200 --compile=False

out_dir = 'out-uk-en-s'
dataset = 'uk_en'  # nanoGPT/data/uk_en -> ../../data/uk_en (prepare.py output)

# --- model: ~42M params with the 32k bilingual vocab -------------------------
# params ~= vocab*n_embd + 12*n_layer*n_embd^2 = 32768*512 + 12*8*512^2 ~= 16.8M + 25.2M
n_layer = 8
n_head = 8           # head size 64
n_embd = 512
block_size = 1024    # Wikipedia articles and PDFs are long: give the model room
bias = False
dropout = 0.0        # ~1 epoch over 1B tokens: no repetition to overfit to

# --- batch: 8 x 1024 x 16 = 131,072 tokens per iteration ---------------------
# micro-batch 8 keeps the 32k-wide logits (8 x 1024 x 32768) within 6 GB
batch_size = 8
gradient_accumulation_steps = 16

# --- schedule: 1 epoch = 1e9 / 131,072 ~= 7,600 iterations -------------------
max_iters = 7600
lr_decay_iters = 7600    # = max_iters
learning_rate = 8e-4
min_lr = 8e-5            # learning_rate / 10
warmup_iters = 400
beta2 = 0.95
weight_decay = 0.1
grad_clip = 1.0

# --- evaluation / checkpoints -----------------------------------------------
eval_interval = 500
eval_iters = 100
log_interval = 10
always_save_checkpoint = False  # only save when val loss improves

compile = True
