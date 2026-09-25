# Continuous Spatiotemporal Event-edge Propagation V1

## Scope

Base is the existing Partial-L2-F80. Constant and Conditioned are trained from
scratch for seeds 20260715, 20260716, 20260717, for 24 complete epochs.
The fixed sample plan has 34,048 training and 7,296 validation transitions.
No graph, target, window, optimizer, task loss or shared/private branch changes.
The original Event loss includes its existing division by log(10).

Only same-period consecutive Event next/gap edges are gated. Parallel edges
share a gate, without relation embeddings. All other edges have gate one.
The gate is applied after unchanged attention normalization, before aggregation.
Features are log1p(clipped seconds)/log(7201), signed normalized displacement,
normalized metric distance, and a position-valid flag. Coordinates use the
destination actor team's view and end-position-first fallback. Constant zeros
all five inputs. Three 5-16-1 controllers add 339 parameters; final layers start
at zero, so gates start at one. Main/private controllers start equal but are
independent. Module construction preserves RNG and baseline forward ordering.

## Selection

Training uses the original five-task objective, fixed denominator 3, learning
rate 9e-4, AdamW weight decay 1e-4, batch 256, gradient clip 5, no early stopping.
Guarded checkpoints minimize raw core validation loss among epochs passing
both raw Team/Player guards: 1pp versus Five-F80 and 0.5pp versus Partial-L2-F80.
An ineligible seed never falls back to an unguarded checkpoint.

Each eligible new checkpoint builds its own frozen train/validation context
cache and refits its original 130-parameter Position head. Player uses the
locked TC-SoftPred prior, lambda 1.5. Base reuses existing predictions and the
fixed-lambda prior, not the cross-fitted variable-lambda predictions.

The three preregistered comparisons are Constant-Base, Conditioned-Constant,
and Conditioned-Base. Improvements require two seeds in the right direction,
a match-cluster 95% paired CI excluding zero, and Event F1 +0.005, Time -0.01s
or Position -0.25m. All configurations and seeds share 10,000 match draws;
seeds are not independently resampled. Conditioned must beat both controls
on the same task. Postprocessed Base guard limits are Event Accuracy -1pp,
F1 strictly better than -0.02, Time strictly less than +0.05s, Position strictly
less than +0.50m, and Team/Player -0.5pp.

Among eligible models, select the minimum postprocessed core loss, breaking
differences below 1e-4 in favor of Base, Constant, then Conditioned. All core
components use their original definitions and all task-valid samples. Epoch
selection retains the original batch-weighted metric; final comparison uses
the same global masked loss calculation for every configuration.

No test data or predictions may be read before selection/spatiotemporal_lock.json.
Base selection skips new test evaluation. Otherwise only the locked model and
necessary controls are evaluated. Test does not change the selection.

## Execution

Run scripts/run_spatiotemporal_edge_matrix.py with --stage
verify, smoke, train, refit, select, test, report, or pipeline.
Train/refit also accept --mode, --seed and --device. Interrupted training
automatically resumes its last.pt with model/optimizer/RNG/sampler/history.
Source changes invalidate a partial run rather than silently changing it.
Smoke outputs never satisfy production eligibility or training completeness.

Background execution uses the independent football-hgt-v4-spatiotemporal-edge
systemd user service. Per-task logs/PIDs/status live under background/.
Each GPU runs at most two jobs, reduced to one if CUDA smoke exceeds 10 GiB.

Reports separate raw and deployed metrics, per-class confusion, gate samples
by time/distance and branch, and CPU/H2D/forward/composed inference costs.
The gate diagnostic sample is explicitly labeled; it is not a population count.
Low-cost targets: forward <=1.20x Base and peak memory <=1.10x Base.

This experiment does not resume State Gating. Regardless of outcome, the next
research stage is cross-match historical coverage and causal data interfaces,
not a more complex gate search. Failure concerns this scalar adjacent-event
mechanism only, not all forms of spatiotemporal propagation.
