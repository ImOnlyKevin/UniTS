# How UniTS Anomaly Detection Works

This guide explains what happens in this UniTS fork when you feed in a telemetry
dataset and get back anomaly scores and anomaly windows.

It is written from the ground up. You do not need to already know what an
autoencoder, transformer, attention block, or PyTorch training loop is.

## Quick Map

In this repo, anomaly detection works like this:

```text
raw telemetry
  -> standardized train/test arrays
  -> fixed-length windows
  -> UniTS reconstructs each window
  -> reconstruction error becomes an anomaly score
  -> high-score points are flagged
  -> adjacent flagged points become anomaly windows
```

The key idea is simple:

```text
If the model can reconstruct a time period well, that period looks normal.
If the model reconstructs it poorly, that period may be anomalous.
```

## Helpful Videos Before Diving In

These are not required, but they are good companions while reading this file.

- Autoencoders: [Simple Explanation of AutoEncoders](https://www.youtube.com/watch?v=H1AllrJ-_30)
- Autoencoders and anomaly detection: [Autoencoders explained - AssemblyAI](https://www.youtube.com/watch?v=qiUEgSCyY5o)
- Transformers, friendly overview: [StatQuest - Transformer Neural Networks, Clearly Explained](https://www.youtube.com/watch?v=zxQyTK8quyY)
- Attention, visual deep dive: [3Blue1Brown - Attention in transformers, step-by-step](https://www.youtube.com/watch?v=eMlx5fFNoYc)
- Attention with code: [Andrej Karpathy - Let's build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- PyTorch training loop: [PyTorch - Training with PyTorch](https://www.youtube.com/watch?v=jF43_wj_DCQ)

## What Is An Autoencoder?

An autoencoder is a neural network trained to copy its input back out.

That sounds useless at first. If the input is:

```text
[1.0, 2.0, 3.0]
```

and the desired output is also:

```text
[1.0, 2.0, 3.0]
```

why train a model at all?

The trick is that the model is usually forced to pass the input through a
compressed or structured internal representation. It cannot simply memorize each
training example as a table lookup. It has to learn patterns that make the input
reconstructable.

For example, in satellite telemetry, normal data is not random. Channels tend to
move together:

- temperatures drift slowly
- voltages stay within operating bands
- mode changes create recognizable transitions
- related sensors often rise or fall together

If the model mostly trains on normal data, it learns those normal relationships.
When it later sees something unusual, the reconstruction often gets worse.

That reconstruction error becomes the anomaly signal.

## Is UniTS Literally An Autoencoder?

Not exactly in the classic textbook sense, but for anomaly detection it behaves
like an autoencoder.

Classic autoencoder:

```text
input -> encoder -> latent representation -> decoder -> reconstructed input
```

UniTS anomaly path:

```text
input window
  -> patch embedding / prompt tokens
  -> UniTS backbone
  -> forecast/reconstruction head
  -> reconstructed input window
```

So when this guide says "autoencoder-style", it means:

```text
The model is trained to reconstruct the same telemetry window it was given.
```

The relevant training code is in `exp/exp_sup.py`:

```python
outputs = model(batch_x, None, None, None, task_id=task_id, task_name=task_name)
loss = criterion(outputs, batch_x)
```

That is the whole anomaly training objective: make the reconstructed output
match the input.

Code anchor: `exp/exp_sup.py`, `train_anomaly_detection`.

## Dataset Files

The anomaly loader expects `.npy` files in each dataset directory:

```text
{dataset_name}_train.npy
{dataset_name}_test.npy
{dataset_name}_test_label.npy
```

In the satellite datasets, these are usually shaped like:

```text
train:      [T_train, channels]
test:       [T_test, channels]
test_label: [T_test]
```

For example:

```text
STPSat7-EPS_train.npy      -> [many timesteps, 242 channels]
STPSat7-EPS_test.npy       -> [many timesteps, 242 channels]
STPSat7-EPS_test_label.npy -> [many timesteps]
```

The loader lives in:

```text
data_provider/data_loader.py
```

Look for the generalized anomaly detection dataset loader near
`Dataset_Custom_AD`.

## Step 1: Standardization

Before UniTS sees telemetry, the loader standardizes it.

It fits a `StandardScaler` on the training data:

```python
self.scaler.fit(raw_train)
```

Then it transforms both train and test:

```python
self.train = self.scaler.transform(raw_train)
self.test  = self.scaler.transform(raw_test)
```

This means the model sees normalized values, not raw engineering units.

Why this matters:

- A temperature channel and a voltage channel may have very different numeric
  scales.
- Without standardization, high-magnitude channels could dominate the loss.
- Standardization makes reconstruction error more comparable across channels.

Important detail: the scaler is fit only on training data. That is the right
thing to do, because test data should not influence the normalization baseline.

Code anchor: `data_provider/data_loader.py`, anomaly loader `__init__`.

## Step 2: Windowing

UniTS does not process the entire mission timeline at once. It processes fixed
windows.

For anomaly detection, the configured `seq_len` becomes the window length. In
the STPSat7 config, many tasks use:

```yaml
seq_len: 96
```

So each sample is:

```text
96 timesteps x number_of_channels
```

For STPSat7-EPS:

```text
[96, 242]
```

When batched:

```text
[batch_size, 96, 242]
```

Your anomaly SLURM script runs with:

```bash
--batch_size 64
```

So a typical batch may be:

```text
[64, 96, 242]
```

In your fork, test windows are non-overlapping:

```python
b = index * self.win_size
return self.test[b:b+self.win_size]
```

So test window 0 covers timesteps `0..95`, window 1 covers `96..191`, and so on.

Code anchor: `data_provider/data_loader.py`, anomaly loader `__getitem__`.

## Step 3: The Training Loop

The high-level training loop lives in `exp/exp_sup.py`.

During each epoch:

1. The data loader provides a batch.
2. The code checks which task it belongs to.
3. For anomaly detection, it calls `train_anomaly_detection`.
4. The model reconstructs the input batch.
5. Mean squared error compares reconstruction vs input.
6. Gradients update trainable parameters.

The dispatch looks like:

```python
elif task_name == 'anomaly_detection':
    loss = self.train_anomaly_detection(...)
```

The loss function defaults to MSE for anomaly detection:

```python
elif each_config[1]['task_name'] == 'anomaly_detection':
    loss_name = 'MSE'
```

MSE means mean squared error:

```text
error = (actual_value - reconstructed_value)^2
```

Squaring has two effects:

- negative and positive errors both become positive
- larger mistakes are punished more heavily

Code anchors:

- `exp/exp_sup.py`, `_select_criterion`
- `exp/exp_sup.py`, `train_one_epoch`
- `exp/exp_sup.py`, `train_anomaly_detection`

## Step 4: Prompt Tuning vs Full Fine-Tuning

Your anomaly script currently runs:

```bash
--pretrained_weight "$CKPT"
--train_epochs 0
--prompt_tune_epoch 10
--prompt_num 10
--patch_len 16
--stride 16
--e_layers 3
--d_model 32
```

This is important.

It means the run loads pretrained UniTS weights, then performs prompt tuning.
During prompt tuning, most of the model is frozen. The dataset/task prompt
parameters are trainable.

In plain English:

```text
The pretrained UniTS model already knows broad time-series structure.
Prompt tuning teaches it how to steer that knowledge for this telemetry dataset.
```

The function that freezes or unfreezes parameters is:

```python
choose_training_parts(prompt_tune=True)
```

When prompt tuning is enabled, non-prompt parameters get:

```python
param.requires_grad = False
```

Code anchor: `exp/exp_sup.py`, `choose_training_parts`.

## Step 5: Tokenization Inside UniTS

The model code lives in:

```text
models/UniTS.py
```

The anomaly forward path starts here:

```python
def anomaly_detection(self, x, x_mark, task_id):
```

At this point, `x` is shaped:

```text
[batch, time, channels]
```

Example:

```text
[64, 96, 242]
```

The first major call is:

```python
x, means, stdev, n_vars, padding = self.tokenize(x)
```

### Per-Window Normalization

Inside `tokenize`, UniTS normalizes each input window again:

```python
means = x.mean(1, keepdim=True).detach()
x = x - means
stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
x /= stdev
```

This is separate from the dataset-level `StandardScaler`.

The dataset scaler normalizes each channel using training-set statistics.

The model's internal normalization normalizes each individual window before it is
converted into tokens.

This helps the model focus on shape and relationships within the current window,
rather than only absolute offsets.

The original `means` and `stdev` are saved so the output can be de-normalized
later.

### Patch Embedding

After normalization, UniTS flips the tensor:

```python
x = x.permute(0, 2, 1)
```

Shape changes from:

```text
[batch, time, channels]
```

to:

```text
[batch, channels, time]
```

Then it splits the time axis into patches.

Your run uses:

```bash
--patch_len 16
--stride 16
```

For a 96-timestep window:

```text
96 / 16 = 6 patches
```

Each channel becomes 6 patch tokens.

Before patching:

```text
[64, 242, 96]
```

After patching and embedding:

```text
[64, 242, 6, d_model]
```

With `d_model=32`:

```text
[64, 242, 6, 32]
```

Each token represents a 16-timestep chunk of one telemetry channel.

Code anchors:

- `models/UniTS.py`, `tokenize`
- `models/UniTS.py`, `PatchEmbedding`

## Step 6: Prompt Tokens

UniTS adds learned prompt tokens before the actual time-series tokens.

For anomaly detection:

```python
x = x + self.position_embedding(x)
x = torch.cat((this_prompt, x), dim=2)
```

With `prompt_num=10`, the token count becomes:

```text
10 prompt tokens + 6 telemetry patch tokens = 16 tokens
```

Shape:

```text
[64, 242, 16, 32]
```

The prompt tokens are learned parameters. You can think of them as a small
task-specific instruction prefix, except they are numeric vectors rather than
English words.

During prompt tuning, these prompts are the main thing being adapted.

Code anchor: `models/UniTS.py`, `prepare_prompt`.

## Step 7: The UniTS Backbone

The backbone is the stack of `BasicBlock`s:

```python
for block in self.blocks:
    x = block(x, prefix_seq_len=prefix_len + seq_len, attn_mask=attn_mask)
```

Each `BasicBlock` does three major operations:

```python
x = self.seq_att_block(x, attn_mask)
x = self.var_att_block(x)
x = self.dynamic_mlp(x, prefix_seq_len=prefix_seq_len)
```

### Sequence Attention

Sequence attention lets tokens from the same channel look at each other across
time.

In telemetry terms:

```text
For this battery-voltage channel, how does this 16-timestep patch relate to the
patches before and after it?
```

This helps the model learn temporal shape:

- slow drift
- sudden jumps
- oscillations
- recovery after mode changes

Code anchor: `models/UniTS.py`, `SeqAttBlock`.

### Variable Attention

Variable attention lets channels look at each other.

In telemetry terms:

```text
How does this channel behave relative to the other channels in the same window?
```

This matters because many anomalies are relational.

A single channel value might look plausible by itself, but suspicious when other
channels do not agree with it.

For example:

- current rises but voltage does not respond normally
- a thermal channel changes without related heaters or modes changing
- one sensor diverges from a group of redundant sensors

Code anchor: `models/UniTS.py`, `VarAttBlock`.

### Dynamic MLP

The dynamic MLP is a feed-forward network that can adapt to different sequence
lengths and prompt lengths. It mixes information after attention has already
shared context across time and variables.

The main thing to know:

```text
Attention moves information around.
The MLP transforms the information after it has been mixed.
```

Code anchor: `models/UniTS.py`, `DynamicLinearMlp`.

## Step 8: Reconstruction Head

After the backbone, UniTS calls:

```python
x = self.forecast_head(x, seq_len + padding, seq_token_len)
```

Even though the function is called `forecast_head`, anomaly detection uses it as
a reconstruction head.

It takes the final patch tokens and maps each token back to a 16-timestep patch:

```python
self.proj_out = nn.Linear(d_model, patch_len)
```

Then it folds patches back into a full time sequence:

```python
torch.nn.functional.fold(...)
```

For the STPSat7-style example:

```text
[64, 242, 6, 32]
  -> [64, 242, 6, 16]
  -> [64, 96, 242]
```

That output has the same shape as the original input.

Code anchor: `models/UniTS.py`, `ForecastHead`.

## Step 9: De-Normalization

Remember the per-window mean and standard deviation saved during tokenization?

UniTS applies them back to the reconstructed output:

```python
x = x * stdev
x = x + means
```

This puts the reconstruction back into the same normalized scale as `batch_x`,
so the loss can compare:

```text
reconstructed window vs original window
```

Code anchor: `models/UniTS.py`, `anomaly_detection`.

## Step 10: Reconstruction Loss During Training

The training loss compares every reconstructed value to its corresponding input
value.

Conceptually:

```text
for every window:
  for every timestep:
    for every channel:
      error = input - reconstruction
      squared_error = error * error
```

The model is optimized to reduce this error.

When training succeeds, the model becomes good at reconstructing patterns that
look like the training distribution.

This is why training data quality matters so much. If the training set contains
many anomalies, the model may learn to reconstruct those anomalies too.

## Step 11: Scoring During Test

Testing uses a different loss object:

```python
anomaly_criterion = nn.MSELoss(reduce=False)
```

The key part is `reduce=False`.

During training, MSE is reduced to one scalar loss so the optimizer can update
weights.

During testing, UniTS keeps the individual errors so it can create one anomaly
score per timestep.

The score is:

```python
score = torch.mean(anomaly_criterion(batch_x, outputs), dim=-1)
```

Input shape:

```text
[batch, time, channels]
```

MSE shape before averaging:

```text
[batch, time, channels]
```

After averaging over channels:

```text
[batch, time]
```

So each timestep gets one score:

```text
score[t] = average reconstruction error across all channels at timestep t
```

Code anchor: `exp/exp_sup.py`, `test_anomaly_detection`.

## Step 12: Thresholding

Once train and test scores are collected, this fork combines them:

```python
combined_energy = np.concatenate([train_energy, test_energy], axis=0)
```

Then it chooses a percentile threshold:

```python
threshold = np.percentile(combined_energy, 100 - self.args.anomaly_ratio)
```

If:

```bash
--anomaly_ratio 1.0
```

then:

```text
threshold = 99th percentile reconstruction error
```

A test point is flagged when:

```python
pred = (test_energy > threshold).astype(int)
```

So `anomaly_ratio` does not mean "the model discovered the true anomaly rate."
It means:

```text
Flag approximately the top anomaly_ratio percent of reconstruction-error scores.
```

Lower `anomaly_ratio`:

```text
higher threshold, fewer anomalies, stricter detector
```

Higher `anomaly_ratio`:

```text
lower threshold, more anomalies, more sensitive detector
```

Code anchor: `exp/exp_sup.py`, `test_anomaly_detection`.

## Step 13: Detection Adjustment

Before metrics and CSV export, the code runs:

```python
gt, pred = adjustment(gt, pred)
```

The helper is in `utils/tools.py`.

It does this:

```text
For every ground-truth anomaly segment:
  if the prediction hits any point inside that segment:
    mark the entire ground-truth segment as predicted
```

This is common in anomaly detection benchmarks because detecting any part of an
event is often considered a successful event-level detection.

However, it is important to understand the consequence:

```text
The exported predicted labels may be expanded using ground-truth labels.
```

The raw `anomaly_score` column is still the actual reconstruction error. But the
`is_anomaly_predicted` column after adjustment may cover a full ground-truth
segment even if only one point crossed the threshold.

This matters when interpreting exported windows.

Code anchor: `utils/tools.py`, `adjustment`.

## Step 14: Exporting Point Scores

Your ESA patch writes a point-level CSV when timestamp files exist:

```text
checkpoints/.../anomaly_results/{data_task_name}_points.csv
```

Columns:

```text
timestamp
anomaly_score
is_anomaly_predicted
is_anomaly_ground_truth
```

The most important column for understanding the model is:

```text
anomaly_score
```

That is the reconstruction error.

High score means:

```text
UniTS had trouble reconstructing this timestep.
```

Code anchor: `exp/exp_sup.py`, timestamp export block in
`test_anomaly_detection`.

## Step 15: Exporting Anomaly Windows

The windows CSV is not produced directly by the neural network.

It is created from adjacent predicted anomaly points:

```python
_mask = _df["is_anomaly_predicted"] == 1
_df["_blk"] = (_mask != _mask.shift()).cumsum()
_wins = _df[_mask].groupby("_blk").agg(...)
```

Each contiguous block gets:

```text
start       first timestamp in the block
end         last timestamp in the block
peak_score  highest reconstruction error in the block
n_points    number of timesteps in the block
```

So the model produces scores. The threshold produces point labels. The export
code groups point labels into windows.

Code anchor: `exp/exp_sup.py`, anomaly CSV export block.

## Full Shape Walkthrough

Here is a concrete STPSat7-EPS-style example.

Assume:

```text
batch_size = 64
seq_len    = 96
channels   = 242
patch_len  = 16
stride     = 16
prompt_num = 10
d_model    = 32
```

Input batch:

```text
[64, 96, 242]
```

After model normalization and permutation:

```text
[64, 242, 96]
```

After patching time into 16-step chunks:

```text
[64, 242, 6, 32]
```

After adding 10 prompt tokens:

```text
[64, 242, 16, 32]
```

After UniTS backbone:

```text
[64, 242, 16, 32]
```

After reconstruction head, keeping only the 6 data tokens and folding patches:

```text
[64, 96, 242]
```

After timestep scoring:

```text
[64, 96]
```

After flattening all windows:

```text
[total_test_scored_timesteps]
```

Then thresholding turns that into:

```text
0 = normal
1 = predicted anomaly
```

Then grouping turns adjacent `1`s into anomaly windows.

## How To Read A Predicted Window

Suppose you see:

```text
start:      2024-01-10 13:00:00
end:        2024-01-10 13:42:00
peak_score: 0.87
n_points:   43
```

That means:

```text
For 43 adjacent timesteps, the point-level prediction was 1.
The worst reconstruction error inside that block was 0.87.
```

It does not automatically tell you which channel caused the anomaly.

The main score is averaged across channels. To investigate cause, you usually
need per-channel attribution:

```text
Which channels had the largest reconstruction errors or largest deviations
during this window?
```

This repo has a helper script:

```text
scripts/attribute_anomalies.py
```

That script compares channels during predicted windows against normal baselines
and ranks channels by deviation.

## Why Reconstruction Error Finds Anomalies

Imagine the model has learned that normal windows look like this:

```text
channel A rises slowly
channel B rises slightly after A
channel C stays flat unless mode changes
```

Now test data shows:

```text
channel A rises sharply
channel B does not move
channel C spikes
```

The model tries to reconstruct the window using patterns it knows. If this
combination is unusual, its reconstruction may be wrong:

```text
actual:        sharp rise, no matching response, spike
reconstruction: smoother rise, expected response, no spike
```

That mismatch creates high MSE.

High MSE becomes high anomaly score.

## Important Limitations

### The model detects unusual reconstruction behavior, not root cause

UniTS does not directly say:

```text
"The EPS battery heater caused this event."
```

It says:

```text
"This timestep was hard to reconstruct from learned normal patterns."
```

Root-cause analysis requires follow-up inspection.

### Threshold choice matters a lot

The `anomaly_ratio` argument controls how many high-score points are flagged.

If the ratio is too low, you may miss real events.

If the ratio is too high, you may flag too much normal variation.

The script `scripts/ratio_sweep.py` exists to compare thresholds and window
counts across different anomaly ratios.

### Training data quality matters

If anomalies are present in the training data, the model may learn to reconstruct
them as normal.

If the training data does not cover enough operating modes, normal-but-rare modes
may get high anomaly scores.

### Windowing affects resolution

The model reconstructs 96-timestep windows, then scores each timestep.

In your fork, test windows are non-overlapping. That is faster, but it means each
test point appears in only one test window. Overlapping windows can sometimes
smooth scores, but they cost more compute.

### Detection adjustment affects exported predictions

For labeled datasets, `adjustment(gt, pred)` can expand predictions across an
entire ground-truth anomaly segment.

When you want to inspect the model's raw behavior, look first at:

```text
anomaly_score
```

Then compare that with:

```text
is_anomaly_predicted
```

## Files Worth Reading Next

Start here:

```text
data_provider/data_loader.py
```

Read:

```text
anomaly detection dataset loader
```

Then:

```text
exp/exp_sup.py
```

Read:

```text
_get_data
_select_criterion
choose_training_parts
train_one_epoch
train_anomaly_detection
test_anomaly_detection
```

Then:

```text
models/UniTS.py
```

Read:

```text
tokenize
PatchEmbedding
prepare_prompt
BasicBlock
SeqAttBlock
VarAttBlock
ForecastHead
anomaly_detection
```

Finally:

```text
utils/tools.py
```

Read:

```text
adjustment
```

## The Whole Pipeline In One Sentence

UniTS anomaly detection trains a pretrained, prompt-tuned transformer-style time
series model to reconstruct normal telemetry windows, treats high reconstruction
error as an anomaly score, thresholds the highest scores according to
`anomaly_ratio`, and groups adjacent flagged points into anomaly windows.

