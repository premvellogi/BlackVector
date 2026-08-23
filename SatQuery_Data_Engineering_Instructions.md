You are my ML/data-engineering lead for the SatQuery project.

I need you to set up a LOCAL, REPRODUCIBLE BigEarthNet.txt training subset for LoRA fine-tuning of a compact InternVL-family VLM.

IMPORTANT CONTEXT
=================

Project:
SatQuery — a multimodal remote-sensing vision-language system.

Hardware:
- 6 GB VRAM laptop GPU
- Solo developer
- Approximately 15 days available
- I cannot afford a huge, complicated data-engineering pipeline.
- I want a working research/hackathon prototype first, then scale only if justified.

Primary training model:
- InternVL3-1B is the FIRST CHOICE — it is the exact backbone used by the paper's own
  published baseline (RS-InternVL) on this exact dataset, so it comes with a validated
  precedent rather than a guess.
- InternVL2-2B is the fallback ONLY if InternVL3-1B has a tooling/compatibility blocker
  (e.g. missing PEFT/LoRA support in your fine-tuning framework of choice). If you fall
  back, say explicitly why InternVL3-1B didn't work before switching.
- LoRA/PEFT fine-tuning in both cases.

Dataset:
- BigEarthNet.txt (arXiv:2603.29630, Herzog et al., 2026)
- It is NOT the original classification-only BigEarthNet.
- BigEarthNet.txt provides language annotations/tasks including:
  1. image captioning
  2. binary VQA
  3. multiple-choice VQA
  4. referring-expression detection (including bounding-box prediction)
- The underlying imagery consists of co-registered Sentinel-1 + Sentinel-2 patches,
  drawn from BigEarthNet v2.0.

KNOWN VERIFIED FACTS — DO NOT RE-DISCOVER THESE, JUST CONFIRM THEM STILL HOLD
==============================================================================

These were verified directly against the HF dataset card and bigearth.net as of this
writing. Treat them as your starting point, but spot-check them against current source
pages in Phase 1 in case anything has changed since — don't skip verification, just
don't start from zero.

- Annotations live in `BigEarthNet.txt.parquet` at
  huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt — ~467MB, 9.55M rows.
  This repo contains ONLY text annotations, no imagery.
- Verified parquet columns: `ID`, `s1_name` (Sentinel-1 patch identifier), `patch_id`
  (Sentinel-2 patch identifier — note this is NOT called `s2_name`, don't invent that
  field), `input` (instruction/question), `output` (reference answer), `type` (binary /
  mcq / captioning / bounding box), `category` (fine-grained task, e.g. adjacency, area,
  count, presence, climate zone, country, season, relative position), `split` (train /
  validation / test / bench), `latitude`, `longitude`, `country`, `season`,
  `climate_zone`. Use these exact names — do not invent `s2_id` or similar.
- **The `split` field has FOUR values, not three: `train`, `validation`, `test`, and
  `bench`.** `bench` is a distinct, separate, manually-verified split (1,082 image pairs,
  15,029 annotations) — NOT a subset of `test`. The paper is explicit that captions in
  the regular train/validation/test splits are LLM-augmented and may contain hallucinated
  content; `bench` is the one split that passed manual verification on all four quality
  dimensions and is balanced across answer options and LULC classes. Use `bench` as your
  primary evaluation set, not a slice of `test`. Do not conflate the two.
- Imagery is NOT on Hugging Face. It's hosted separately at bigearth.net / Zenodo (record
  10891137) as two monolithic archives: `BigEarthNet-S2.tar.zst` (~59 GiB) and
  `BigEarthNet-S1.tar.zst` (~51 GiB). There is no official per-patch selective download —
  confirm this is still true in Phase 1, but budget for it being true.
- The official dataset structure/naming-convention reference is
  `Description_BigEarthNet_v2.pdf`, linked from bigearth.net's "Additional Links" —
  read this, don't reconstruct the directory layout from tutorials or guesses.
- `metadata.parquet` and `metadata_for_patches_with_snow_cloud_or_shadow.parquet` are
  also hosted on the Zenodo record — these carry per-patch BigEarthNet v2.0 metadata
  (separate from BigEarthNet.txt's own parquet) and may help with the S1↔S2↔LULC mapping.

MY MAIN OBJECTIVE
=================

Create a manageable local subset of the ORIGINAL BigEarthNet v2.0 imagery that corresponds
to the BigEarthNet.txt annotations.

I do NOT want to permanently download/store the full ~110 GB S1+S2 archive.

The final local working dataset should ideally be only a few GB, depending on the selected
number of samples.

CRITICAL DATASET FIDELITY REQUIREMENT
=====================================

DO NOT replace BigEarthNet v2.0 imagery with arbitrary Sentinel-1/Sentinel-2 imagery from
Planetary Computer, Sentinel Hub, or another source for the main BigEarthNet.txt
training/evaluation dataset.

The reason is dataset fidelity, and it is a correctness issue, not a style preference:
BigEarthNet.txt's annotations — including bounding boxes and area/count answers — were
generated against the exact pixel grid, windowing, and processing (sen2cor v2.11) of the
official BigEarthNet v2.0 patches. A re-crop from an independently-sourced Sentinel-1/2
archive, even of the "same" underlying scene, risks a different processing baseline and
different pixel alignment — which silently invalidates bounding boxes and quantity
answers against the actual image. This would not error out; it would just quietly produce
wrong ground truth. Do not take this shortcut for the primary dataset under any framing.

Planetary Computer may be used later for:
- external/generalization experiments
- temporal SatQuery experiments
- additional unseen imagery

But it must NOT silently replace the official BigEarthNet v2.0 imagery in the
BigEarthNet.txt training dataset.

If you discover that the official BigEarthNet imagery distribution is monolithic and
requires downloading a large archive, that is acceptable as a TEMPORARY extraction step.

The desired workflow is:

BigEarthNet.txt metadata/annotations
        ↓
select desired patch IDs (image-pair level, stratified, bench split kept separate)
        ↓
identify corresponding S1 + S2 patches
        ↓
download official BigEarthNet v2.0 archive(s) if necessary — ONE AT A TIME (see Phase 3)
        ↓
extract ONLY selected patches
        ↓
verify them
        ↓
delete the archive just used before downloading the next one
        ↓
keep only the selected local subset

Do NOT tell me to permanently store 110 GB unless absolutely unavoidable.

BEFORE DOING ANYTHING
=====================

First inspect and verify the CURRENT official dataset structure and tooling.

Use authoritative/current sources, in this priority order:
1. `Description_BigEarthNet_v2.pdf` (bigearth.net "Additional Links") — the canonical
   structure/naming-convention reference.
2. bigearth.net itself (About / Downloads / FAQ sections).
3. The official BigEarthNet.txt Hugging Face dataset card
   (BIFOLD-BigEarthNetv2-0/BigEarthNet.txt) — read its "Parquet File Structure" and
   "How to use" sections in full; it links a reference PyTorch Dataset/DataModule
   (`ben_txt_datamodule.py`) and a preprocessing tool (`rico-hdl`) — check both before
   designing your own loader from scratch.
4. The BigEarthNet.txt paper itself (arXiv:2603.29630) for the annotation-generation
   methodology and the `bench` split definition.
5. TorchGeo only as a secondary implementation reference if useful — verify whether its
   BigEarthNet dataset class actually streams patches on demand or downloads the full
   archive locally on first use before relying on it for anything.

Do not assume that:
- Hugging Face contains the imagery (confirmed it does NOT — annotations only)
- Hugging Face streaming magically provides the original imagery
- Planetary Computer patches are identical to BigEarthNet patches
- the dataset is organized exactly as described in older tutorials
- a pre-built LMDB archive is available for direct download (as of this writing, LMDB is
  something you BUILD from the raw archive via `rico-hdl`, not something you download
  instead of it — confirm this is still the case, don't assume a shortcut exists)

Verify these things from the actual current dataset/tooling before designing the
downloader.

IMPORTANT:
If sources disagree, show me the disagreement and determine which source should be
treated as authoritative. Do not silently invent a mapping.

PHASE 1 — UNDERSTAND THE DATASET
================================

Before downloading imagery, inspect the BigEarthNet.txt dataset and determine:

1. What files/metadata are available?
2. What is the exact patch identifier?
3. How are S1 and S2 patches identified? (Starting point: `s1_name` and `patch_id` — confirm.)
4. How are annotations linked to image patches?
5. What train/validation/test/**bench** split information exists, and how is `bench`
   distinguished from `test`?
6. What fields identify:
   - task (`type`)
   - question/instruction (`input`)
   - answer (`output`)
   - caption
   - referring expression
   - bounding box / mask if applicable
7. How many annotations exist per image pair?
8. How are the 15 tasks / 4 categories represented in `type` and `category`?
9. What metadata is available for land-cover labels, location, season, country, climate
   zone?
10. What exact relationship maps BigEarthNet.txt examples to BigEarthNet v2.0 imagery?

Do NOT start downloading 110 GB before this mapping is understood.

Create a small report:

dataset_structure_report.md

containing:
- verified dataset structure
- relevant fields (confirm or correct the field list given above)
- patch-ID mapping
- S1/S2 relationship
- annotation/task relationship
- split information, with `bench` explicitly called out as distinct from `test`
- source references, with a note on which claims above you CONFIRMED vs. found DIFFERENT

PHASE 2 — DESIGN THE SUBSET
===========================

Do NOT simply do:

ds.take(10000)

because the first 10,000 streaming examples may not be representative.

I want a representative subset.

Target progression:

Stage A:
~500 S1/S2 image pairs

Purpose:
- pipeline validation
- debugging

Stage B:
~5,000–10,000 S1/S2 image pairs

Purpose:
- primary LoRA experiment

Stage C:
~20,000–30,000 pairs ONLY if Stage B works and additional data is justified.

Do not jump directly to 30k.

The subset must be selected at the IMAGE-PAIR level, not independently at the
text-annotation level.

One S1/S2 pair can have multiple annotations.

For example:

image_pair_123
 ├── caption
 ├── binary VQA
 ├── MCQ
 └── referring expression

Therefore preserve the relationship between:
image pair ↔ all relevant annotations.

**The `bench` split (1,082 pairs) is your evaluation set — pull it in full, as-is, kept
entirely separate from whatever you sample for training. Do not stratify-sample it, do
not mix it into `test`, do not train on it.**

SAMPLING REQUIREMENTS
======================

Use a stratified/representative selection strategy, drawn from the `train` split only.

Consider:
- task distribution (`type`/`category`)
- multi-label land-cover distribution
- geographic diversity (`country`, `latitude`/`longitude`)
- season diversity (`season`)
- climate zone diversity (`climate_zone`)
- S1/S2 availability
- rare labels/classes

BigEarthNet is multi-label, so DO NOT assume:
one image = one class.

Try to prevent the subset from being dominated by common land-cover categories.

However, do not distort the official validation/test distribution — if you also build a
validation subset, sample it from `validation` the same way, and keep it separate from
your training sample and from `bench`.

VERY IMPORTANT:
Do not sample validation/test examples into training.
Do not sample `bench` examples into training or validation — `bench` is evaluation-only.

Prefer the official dataset splits if available.

Create:

subset_manifest.jsonl

with one entry per selected image pair, containing enough information to reproduce the
subset. Use the ACTUAL verified field names from Phase 1 — starting point (confirm before
finalizing):

{
  "id": ...,
  "s1_name": "...",
  "patch_id": "...",
  "split": "train",
  "types": [...],
  "categories": [...],
  "latitude": ...,
  "longitude": ...,
  "country": "...",
  "season": "...",
  "climate_zone": "...",
  "source": "BigEarthNet_v2"
}

Do not invent field names not present in the actual parquet schema.

PHASE 3 — DETERMINE THE IMAGERY DOWNLOAD STRATEGY
=================================================

Now determine exactly how the official BigEarthNet v2.0 imagery is distributed.

If it is distributed as large archives (starting assumption: S2 ~59 GiB, S1 ~51 GiB,
confirm current sizes), then design a temporary extraction workflow.

**Download and process the two archives SEQUENTIALLY, not simultaneously — this caps
peak temporary disk usage at roughly one archive's size instead of both combined:**

1. Download `BigEarthNet-S2.tar.zst` only.
2. Extract only the S2 patches matching the subset manifest.
3. Verify the extracted S2 patches (see Phase 4).
4. Delete `BigEarthNet-S2.tar.zst` only after verification succeeds.
5. Only then download `BigEarthNet-S1.tar.zst`.
6. Extract only the matching S1 patches, verify, then delete the S1 archive.

Requirements:

1. Check available disk space BEFORE downloading each archive — I have roughly 176 GB
   free total; confirm there's enough headroom for one archive plus its extracted subset
   plus working space before starting.
2. Calculate peak temporary storage required for EACH step (one archive at a time, per
   the sequential order above), not the combined S1+S2 total.
3. Warn me if my disk cannot safely handle a given step.
4. Make downloads resumable if possible.
5. Verify archive integrity/checksums if available.
6. Extract ONLY selected patch directories/files — use `tar`'s ability to extract named
   members from a compressed stream rather than fully decompressing the archive to disk
   first.
7. Do not duplicate the entire archive into another directory.
8. Delete each archive ONLY after:
   - extraction succeeds
   - selected patches from that archive are verified
   - manifest matches extracted files
9. Never delete a source archive automatically without first confirming verification
   succeeded.

If even one archive is too large for my available temporary disk, STOP and explain the
exact storage requirement instead of inventing a workaround. Then propose the smallest
safe alternative (e.g. smaller Stage A/B sample first).

PHASE 4 — VERIFY S1/S2 PATCHES
==============================

For every selected pair, verify:

- S1 exists
- S2 exists
- IDs correspond correctly
- dimensions are sensible
- CRS is present where expected
- transform/georeferencing is present
- expected bands/channels are available
- no corrupted files
- NoData handling is understood
- S1 and S2 correspond to the same BigEarthNet patch/location

Use Rasterio for verification where appropriate.

Create a verification script:

verify_bigearthnet_subset.py

It should produce a report like:

total selected pairs: 5000
valid S1/S2 pairs: ...
missing S1: ...
missing S2: ...
mismatched pairs: ...
corrupt files: ...
unexpected dimensions: ...
missing metadata: ...

Do not silently skip errors.

PHASE 5 — KEEP THE DATA IN A SIMPLE FORMAT
==========================================

For the first version, DO NOT introduce LMDB/HDF5/WebDataset unless it is actually
necessary.

I want a simple, debuggable structure.

Prefer something conceptually like:

bigearthnet_subset/
│
├── metadata/
│   ├── train.jsonl
│   ├── validation.jsonl
│   ├── bench.jsonl
│   └── subset_manifest.jsonl
│
├── imagery/
│   ├── PATCH_001/
│   │   ├── S1/
│   │   └── S2/
│   ├── PATCH_002/
│   │   ├── S1/
│   │   └── S2/
│   └── ...
│
└── reports/
    ├── dataset_structure_report.md
    └── verification_report.json

Adapt this to the actual BigEarthNet archive structure once confirmed in Phase 1.

Do NOT copy files unnecessarily.

PHASE 6 — RASTERIO PREPROCESSING
================================

Create a minimal Rasterio-based loader.

It should be able to:

1. Open S1
2. Open S2
3. Inspect CRS
4. Inspect resolution
5. Inspect bands
6. Read only required windows/patches
7. Handle NoData
8. Convert data into NumPy/PyTorch tensors
9. Return metadata needed for later geospatial operations

Do NOT load gigantic scenes into memory unnecessarily.

The final training pipeline should eventually look like:

BigEarthNet patch
      ↓
Rasterio
      ↓
S1 + S2 arrays
      ↓
preprocessing
      ↓
PyTorch tensors
      ↓
InternVL3-1B (or InternVL2-2B fallback)
      ↓
LoRA

PHASE 7 — SPECTRAL BAND HANDLING
================================

Do NOT automatically discard Sentinel-2 bands just because InternVL is RGB-oriented.

First determine exactly what BigEarthNet v2.0 provides and how the paper's own
RS-InternVL baseline adapts S1/S2 (modality-specific projection layers into the LLM
embedding space, frozen ViT backbones per modality, per the paper — confirm this against
the paper text directly rather than assuming).

For the FIRST sanity test, it is acceptable to use a simpler representation (e.g. RGB
only) if necessary.

But keep the full S1/S2 information available for the actual RS-adaptation experiment.

I want two clearly separated concepts:

A. Minimal pipeline sanity check (RGB-only acceptable)
B. Actual multisensor RS experiment (full S1/S2, matching the paper's approach)

Do not accidentally turn the final experiment into an RGB-only experiment and call it
S1/S2 multimodal learning.

Also document:
- Sentinel-2 band count and native resolutions
- resampling strategy
- Sentinel-1 VV/VH handling
- normalization/scaling
- NoData handling

Do not invent preprocessing values. Verify them from the dataset documentation
(`Description_BigEarthNet_v2.pdf`) or an established implementation (e.g. ConfigILM).

PHASE 8 — BIGEARTHNET.TXT TASK STRUCTURE
=========================================

Preserve the language annotations exactly as they appear in the `input`/`output` fields.

I eventually want to train/evaluate:

1. Binary VQA
2. Captioning
3. Referring-expression detection
4. MCQ VQA

Prioritize:

P0:
- Binary VQA
- Captioning

P1:
- Referring expression

P2:
- MCQ

The dataset loader should be capable of returning examples in a common instruction format
such as:

{
  "image": ...,
  "type": "binary",
  "category": "...",
  "instruction": "...",
  "answer": "..."
}

but use the actual annotation structure and field names from BigEarthNet.txt (`type`,
`category`, `input`, `output`) rather than inventing new ones.

Do NOT rewrite or hallucinate annotations.

PHASE 9 — LOADING STRATEGY FOR TRAINING
=======================================

I want a PyTorch/Hugging Face-compatible Dataset/DataLoader.

Requirements:

- lazy loading
- no loading of entire dataset into RAM
- Rasterio reads samples when needed
- configurable number of workers
- configurable batch size
- gradient accumulation support
- mixed precision where compatible
- ability to train from the 500-sample subset first
- ability to switch to 5k/10k without changing the model code

Keep preprocessing deterministic where possible.

PHASE 10 — 6 GB VRAM CONSTRAINT
===============================

Design everything around 6 GB VRAM.

Do NOT assume:
- 24 GB VRAM
- 48 GB VRAM
- multi-GPU
- huge batch sizes

The first goal is:

ONE successful LoRA training run.

Use conservative:
- image resolution
- batch size
- gradient accumulation
- sequence length
- number of trainable parameters
- activation memory

If full S1/S2 + InternVL3-1B cannot fit, DO NOT silently redesign the project.

Instead report:
1. exact memory bottleneck
2. what component causes it
3. smallest feasible model/configuration (try InternVL3-1B first, InternVL2-2B only as
   a documented fallback with reasons)
4. recommended fallback

Potential fallback:
- smaller input resolution
- fewer trainable layers
- projector-only LoRA
- vision/projector adaptation instead of full VLM LoRA

But make the decision based on measured memory, not guesses.

PHASE 11 — DATA LEAKAGE PROTECTION
==================================

This is critical.

Ensure that:
- training patches are not in validation
- validation patches are not in `bench`
- `bench` patches never appear in training or validation
- duplicate patch IDs are not split across datasets
- annotations belonging to the same image pair remain associated
- no derived copy of a training patch appears in validation/bench

Create a script:

check_dataset_leakage.py

that checks this automatically.

PHASE 12 — REPRODUCIBILITY
===========================

Everything should be reproducible.

Create:

README_DATASET.md

including:

- source datasets and versions (BigEarthNet.txt / arXiv:2603.29630, BigEarthNet v2.0)
- download procedure (sequential S2-then-S1, per Phase 3)
- subset selection procedure
- sampling seed
- selected number of pairs
- train/validation/bench counts
- S1/S2 processing
- preprocessing
- local directory structure
- disk requirements (peak, per-step, per Phase 3)
- commands to reproduce
- how to delete temporary archives safely

Create a config file, for example:

configs/bigearthnet_subset.yaml

containing:
- seed
- target number of pairs
- split
- tasks
- sampling strategy
- storage paths
- preprocessing settings

Do not hard-code paths throughout Python scripts.

PHASE 13 — DO NOT OVERENGINEER
==============================

My priority is:

WORKING PIPELINE > PERFECT INFRASTRUCTURE

Do NOT immediately build:
- custom database
- complex distributed loader
- LMDB
- WebDataset
- custom cloud pipeline
- massive preprocessing cache
- multi-GPU training

unless the simple pipeline demonstrably cannot work.

Start with:
JSONL + official patch files + Rasterio + PyTorch Dataset.

Optimize later.

PHASE 14 — WHAT I WANT YOU TO PRODUCE
=====================================

Before executing expensive downloads, give me:

1. VERIFIED DATASET ARCHITECTURE
2. EXACT SOURCE OF ANNOTATIONS
3. EXACT SOURCE OF IMAGERY
4. EXACT MAPPING BETWEEN ANNOTATIONS AND S1/S2 PATCHES
5. EXPECTED DOWNLOAD SIZE (per archive, since they're downloaded sequentially)
6. EXPECTED TEMPORARY DISK REQUIREMENT (peak, per Phase 3's sequential approach)
7. EXPECTED FINAL SUBSET SIZE
8. RECOMMENDED INITIAL SAMPLE SIZE
9. SAMPLING STRATEGY
10. TRAIN/VALIDATION/BENCH STRATEGY (with `bench` explicitly distinguished from `test`)
11. COMPLETE DIRECTORY STRUCTURE
12. COMMANDS TO EXECUTE
13. RISKS / FAILURE MODES
14. WHAT YOU VERIFIED VS WHAT YOU INFERRED — for every claim in the "KNOWN VERIFIED
    FACTS" section above, explicitly confirm it still holds or flag what's changed.

Do not start a 50–110 GB download before showing me this report.

AFTER I APPROVE THE PLAN
=========================

Implement the following scripts/modules:

1. inspect_bigearthnet.py
2. build_subset_manifest.py
3. download_bigearthnet_subset.py
4. verify_bigearthnet_subset.py
5. check_dataset_leakage.py
6. bigearthnet_dataset.py
7. raster_loader.py
8. configs/bigearthnet_subset.yaml
9. README_DATASET.md

The scripts should be modular and easy to debug.

IMPORTANT FINAL RULES
=====================

1. Do not use Planetary Computer imagery as a substitute for BigEarthNet v2.0 training
   imagery.
2. Do not download the full ~110 GB permanently — download and process the S2 and S1
   archives SEQUENTIALLY (Phase 3), deleting each before starting the next.
3. Do not assume Hugging Face streaming provides the imagery — it does not; that repo is
   annotations only.
4. Do not use `ds.take(N)` as the final sampling strategy.
5. Sample at image-pair level.
6. Preserve all relevant annotations belonging to selected pairs.
7. Preserve official train/validation/test/bench splits — and never conflate `bench`
   with `test`; `bench` is the manually-verified evaluation set.
8. Do not invent annotation fields or patch mappings — use the verified field names from
   the "KNOWN VERIFIED FACTS" section, confirming them in Phase 1.
9. Verify all assumptions against current authoritative documentation/tooling.
10. Do not begin a massive download until storage requirements and the extraction
    strategy are verified, including the sequential (not simultaneous) archive order.
11. Keep the first implementation simple.
12. Optimize only after the 500-sample end-to-end pipeline works.
13. If something cannot be verified, explicitly say "UNVERIFIED" rather than guessing.
14. If you find a better official/authoritative way to obtain only selected BigEarthNet
    patches without downloading the entire archive, investigate it first.
15. If no partial-download mechanism exists, use temporary sequential archive download +
    selective extraction + verification + deletion, per Phase 3.

START NOW.

First perform ONLY the dataset/source/tooling investigation and give me the verified
plan. Do not download the large imagery archives yet.
