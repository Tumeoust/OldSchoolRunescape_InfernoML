
# Old School Runescape Inferno ML

A reinforcement-learning agent that completes the [Inferno](https://oldschool.runescape.wiki/w/Inferno) —
Old School RuneScape's hardest solo PvM challenge — in a tick-accurate Python simulator.

Architectural decisions, reward shaping, environmental datapoints and other PPO related solutions were ideated by me and built with Claude. 
All of the contents in this repository are written by Claude apart from this paragraph.

I found that Claude was unable to come up with good strategies on reward terms, observations nor was it good at interpreting TensorBoard logs.
This was my first real ML project, so there were a lot of mistakes made but in general I think I got a good feel for how AI learns, how rewards shape behavior and how to find a balance.

There were a bunch of mistakes in reward side where the agent would find loop holes, such as hiding instead of fighting. Drafting rewards that would punish the agent after 20 ticks of idle time meant
 the agent would come up with strategies where it could be in safety for 19 ticks and peek out for 1 tick, just in time to reset the counter to avoid the penalty and keep avoiding death penalties.
 
 Combatting exploit was one of the toughest challenges. Also the agent initially had defensive bonuses during training. This meant that not every prayer missed would necessarily result in taking damage, which didn't push the agent to find the optimal strategy.
 Reducing defence to 0 during training meant that every NPC attack would almost certainly deal damage on every missed prayer. This finally forced the agent to find an optimal solution for the problem.



Stack: custom PPO implementation (recurrent policy: 256-unit LSTM with residual raw-observation skip,
512×512 actor/critic heads) is trained against a headless simulator that reproduces Inferno wave
mechanics: NPC pathfinding, line of sight, attack cycles, prayer, pillar geometry, NPC mechanics,
and gear-dependent combat formulas.

> **Educational / research project.** Everything here runs against an offline, self-contained
> Python simulator. This is **not** a bot: it has no connection to the live game, performs no
> screen reading or input automation, and is not intended for use against Old School RuneScape.
> The interesting part is the reinforcement-learning problem, not playing the real game.
> No model is shipped with the project.

## Demo

The agent clearing waves 62–64:

https://github.com/user-attachments/assets/c69519ea-2961-4fc0-85cb-4cf0692bd3a1

Deterministic V52 policy: the agent displays that it was learned crucial tactics, such as pillar usage, off-ticking NPC's and other high-level game play. ([download](media/RL_Waves_62-64.mp4), 3 MB)

## Agent capabilities

It learned high-level tactics such as leveraging the pillar for safety, long-horizon planning and efficient kill orders when applicable and safe to do so (such as killing the Mager first).
Every single wave of the game is purely randomized, every NPC has 12 possible spawning locations so every single wave the agent sees is "new". It has to generalize on knowledge, rather than memorize a working solution.
LSTM allowed the agent to stay still when necessary, wait for NPC's to move to optimal positions and avoid danger more effectively than a pure MLP did.

## Results

Best checkpoint (`V52`, 740.6M training steps), evaluated over waves 1–66, 50 seeds per loadout,
real gear stats, deterministic policy — **99.2% weighted clear rate**:

| Loadout       | Clear% | Death% | Mean Wave |
|---------------|--------|--------|-----------|
| BUDGET_RCB    | 100.0% | 0.0%   | 66.0      |
| MID_ACB       | 98.0%  | 2.0%   | 65.7      |
| CRYSTAL_BP    | 100.0% | 0.0%   | 66.0      |
| CRYSTAL_NO_BP | 100.0% | 0.0%   | 66.0      |
| MAX_TBOW      | 98.0%  | 2.0%   | 65.9      |

Scope: the RL agent plays waves 1–66, waves 67-69 are arguably easier than learning to complete waves 1-66 without any food. 67-69 could be introduced, but I don't have an interest to expand the simulator.

## How it works

### The problem

The Inferno is a 69-wave survival gauntlet played on a 600 ms tick clock. Every tick the player
may move, attack one target, switch weapons, and change overhead prayer — and a single mistimed
prayer against the wrong attack style can end a multi-hour run. What makes it hard for RL:

- **One-tick decisions with delayed consequences.** Positioning mistakes (giving two attack styles
  simultaneous line of sight) are often lethal several ticks later.
- **Long horizons.** A full run is tens of thousands of ticks; the terminal success signal is
  extremely sparse.
- **Partial observability.** Blob attack styles are hidden until scanned, NPC attack timers
  matter across ticks, and mid-wave state (which pillar nibblers target, what died where) has
  long-range consequences — hence the recurrent policy.

### Tick-accurate simulator (`simulator/`)

Training runs against a headless Python reimplementation of the fight, validated against the
real game: NPC "dumb" pathfinding with diagonal-clipping
rules, fixed-point Bresenham line of sight, per-style attack cycles and prayer protection,
attack-drag movement, blob scan/split mechanics, meleer digging, nibbler/pillar behavior, and
OSRS accuracy/max-hit formulas for real equipment stats. Discrepancies were found and fixed by
replaying RuneLite traces of real fights through the simulator (see
`tools/inferno_rl/docs/TRACE_VALIDATION.md`).

The hot paths (movement resolution, forecasting, LOS) have Cython backends
(`setup_cython.py`) for a 3–5x speedup; training runs dozens of vectorized environments.

### Observation space — 602 dims (`training/observation_v4.py`)

The observation is a flat 602-dim vector. The design lesson of the project: **observation
quality beats observation size** — pre-computed tactical features outperform raw entity dumps
because they remove inference burden from the policy. Layout:

| Block                 | Size | Contents |
|-----------------------|------|----------|
| Global                | 51   | Player state, weapon/prayer one-hots, pillar HP, wave, kill/dead-pool counts, blob-scan counts, nibbler-per-pillar counts, resurrection hazard |
| Neighborhood forecast | 108  | 9 tiles × 12 features: for each adjacent tile (and stay), a one-tick-ahead prediction — LOS count/delta, imminent attacks by style, whether auto-prayer would leave you unprotected, blob-scan triggers, multi-step LOS lookahead |
| Threat horizon        | 9    | Incoming attack counts by style, forecast 3 ticks ahead |
| Temporal              | 7    | Rolling 5-tick damage sums, ticks since last attack/engagement, previous-action flags |
| Exact target slots    | 420  | 14 slots × 30 features, sorted by threat priority: geometry, HP, signed attack delay, frozen/stun timers, LOS both ways, dig pressure, blob scan state, pillar-relative angles, entity-type one-hot |
| Loadout               | 7    | Weapon speed/range/bonuses, max HP for the current gear set |

The 14 target slots share their ordering with the action space (below), so "attack slot 3"
always means the entity described in slot 3 of the observation.

Full reference: `tools/inferno_rl/docs/OBSERVATION_SPACE.md`.

### Action space and masking (`training/actions.py`)

The policy outputs a **factored MultiDiscrete action** with four autoregressive heads:

| Head   | Size | Meaning |
|--------|------|---------|
| mode   | 5    | stay / move / attack / switch / no-op |
| move   | 32   | 8 directions × 4 distances |
| attack | 14   | one per exact-target slot |
| switch | 4    | BoFa / blowpipe / ice barrage / blood barrage |

Earlier versions used one flat softmax over 43+ actions; factoring was the single biggest jump
in the project (V44) because the flat head entangled unrelated logits — noise in movement
probabilities perturbed attack decisions. Later heads are conditioned on earlier ones
(autoregressive), and **per-head action masks** remove invalid choices before sampling:
blocked movement directions, empty/unattackable target slots, weapons not in the current
loadout, and sub-heads that don't match the chosen mode (dependency masks). Prayer is not an
action: the environment auto-prays against the forecast threat, and the observation tells the
policy how that auto-prayer will resolve — the agent's job is to position so that one prayer
is enough.

### Network (`ppo/policy.py`)

`flat_lstm_residual`: the normalized 602-dim observation feeds a 256-unit LSTM, and the LSTM
output is **concatenated with the raw observation** (residual skip) before separate 512×512
actor and critic MLPs. Heavy recurrent front-ends (an earlier entity-pool LSTM encoder)
consistently underperformed — gating all information through a recurrent bottleneck slowed
learning. The small residual LSTM adds multi-tick memory (blob scans, attack cycles) without
hiding the current tick from the heads. Actor and critic see the same observation; a
privileged-critic experiment (V44–V46) was retired with observation v4.

The PPO stack is custom (`ppo/`): recurrent rollouts with sequence length 32 and burn-in 16,
observation and reward normalization, target-KL early stopping, and masked/autoregressive
action heads. The recipe was adapted from the open-source
[osrs-pvp-reinforcement-learning](https://github.com/Naton1/osrs-pvp-reinforcement-learning)
project and tuned from there.

### Reward shaping (`training/rewards.py`)

Two findings dominated the reward design:

1. **Dense shaping is structurally necessary.** With only sparse terminal rewards, entropy
   collapses — there is no per-tick gradient between wave completions. Removing dense terms
   was tried twice and failed both times.
2. **But shaping must decay or it gets farmed.** Fixed-magnitude positioning bonuses were
   exploited (idling on "good" tiles instead of fighting), so shaping terms are
   schedule-decayed over training, and dense rewards require recent real engagement.

The final config is deliberately small (~11 active terms) after V51 stripped 28 terms down:
damage taken (−0.05/HP) and dealt (+0.003/HP), wave completion (+3) plus a wave-scaled
progress bonus, end-of-wave HP bonus, Inferno completion (+15), death (−20), wave timeout
(−15), an escalating stall penalty, a small LOS-separation bonus for splitting enemy
sightlines, and **one-time pillar-death penalties** (−2.5 NW/S, −5 NE) — the final change
that took the model from ~86% to 99.2%: with no pillar signal it had learned to let its own
cover die, and per-HP pillar penalties were too noisy, but discrete death events worked.

### Curriculum (`adaptive_curriculum.py`, `rollout_sampler.py`)

Wave sampling went through four generations:

1. **Frontier climb** — advance the wave range as the model masters it. Sample-efficient early,
   but over-concentrates on the frontier and forgets earlier waves.
2. **Drill / hard-example mining** — retry failed waves. Plateaus: remaining failures needed
   observation/reward fixes, not more exposure.
3. **Adaptive controller** — champion-tracked regime switching between harden / backfill /
   opener modes with rollback on regression.
4. **Failure-weighted sweep** (final) — all waves in range available from the start, sampled
   proportionally to measured per-wave failure rate, with **persistent episodes** across PPO
   rollouts (LSTM state and episode survive rollout boundaries instead of resetting every
   128 ticks).

Generalization came from the environment side: training randomizes over 5 gear loadouts
(budget crossbow → max tbow, with a loadout observation block), and — critically — forces
**1 Defence with uniform defensive bonuses** during training. At 1 Defence nearly every NPC
hit lands, so tanking is never viable and the only surviving policy is perfect
positioning/prayer; real defensive stats at eval time are then a free safety margin.

### Tooling that fed the loop

- **Behavior-cloning warmstart** bootstrapped early versions (`pretrain/`); later versions
  restarted fresh once RL surpassed the warmstart.
- **Death analysis** (`death_analysis.py`) buckets failures per wave/loadout/cause and drove
  most observation and reward changes.
- **Replay + trace tools** (`cli/`, `analyze_trace.py`) replay recorded episodes tick-by-tick
  and diff simulator behavior against real-game traces.
- **Pygame visualizer** (`visualizer/`) for watching checkpoints and stepping through waves.
- Every run's config, reward changes, TensorBoard metrics, and eval results are logged in
  `tools/inferno_rl/docs/V*_TB_TRACKING.md` — the full experimental record.

## How it got to 99.2%

Condensed from the V9–V53 tracking docs (clear-rate numbers below use different benchmarks
across eras and are not directly comparable):

| Versions | What changed | Outcome |
|----------|--------------|---------|
| V9–V21   | SB3 MaskablePPO baseline, 186-dim obs, BC warmstart | 30% clear W49–66; long-lived baseline |
| V22–V26  | Bigger nets, LSTM-256, richer obs — all at once | Worse than baseline; full revert. Lesson: change one variable at a time, small models learned faster here |
| V27–V29  | Reward-magnitude fight (grad norm ~6.0): aux head removed, rewards ÷5, pvp-ml recipe adopted | Stable training framework, [512,512] nets |
| V30–V36  | Careful recurrence + structured obs v2 + adaptive curriculum | 36% W55–66, then plateau — the encoder-heavy LSTM arch capped out |
| V37–V40  | Forecast-first observations (threat horizon, per-tile safety forecast) + `flat_lstm_residual` arch | Final architecture in place |
| V41–V43  | Failure-weighted sweep curriculum, persistent episodes, reward-farming fixes | Cleaner training signal at all waves |
| **V44**  | **Factored action space + privileged critic + schedule-decayed shaping** | **89% single-loadout W49–66 — the breakthrough** |
| V45–V46  | 5 randomized loadouts, then 1-Defence forcing | Generalization across gear instead of stat-tanking |
| V47–V49  | Observation v4 (fresh restart): exact-target slots, identity anchors, pillar geometry, LOS lookahead (504→602 dims) | ~62% five-loadout eval |
| V50–V51  | Task-dominant rewards (death/completion), then stripping 28 shaping terms to ~11 | Cleaner critic targets |
| **V52**  | **One-time pillar-death penalties, W31–66 sweep** | **99.2% weighted clear over 5 loadouts** |
| V53      | Research notes only (PLR, self-predictive aux losses) — remaining bottleneck is long-gap credit assignment | — |

Recurring lessons: dense shaping is necessary but must decay; observation quality beats size;
factored actions beat flat softmax; smaller models learned faster than bigger ones; and
"less is more" applied twice (V26 full revert, V51 reward strip). Details:
`tools/inferno_rl/docs/TRAINING_FINDINGS.md`.

## Documentation

| Topic                        | Location                                         |
|------------------------------|--------------------------------------------------|
| System architecture          | `tools/inferno_rl/ARCHITECTURE.md`               |
| Training findings            | `tools/inferno_rl/docs/TRAINING_FINDINGS.md`     |
| Observation space            | `tools/inferno_rl/docs/OBSERVATION_SPACE.md`     |
| Simulator-vs-reality checks  | `tools/inferno_rl/docs/TRACE_VALIDATION.md`      |
| Per-version training history | `tools/inferno_rl/docs/V*_TB_TRACKING.md`        |
| Inferno tactics knowledge    | `tools/inferno_rl/docs/inferno_knowledge/`       |

The `V*_TB_TRACKING.md` docs are the full experimental record — every training run's
configuration, reward-shaping changes, TensorBoard metrics, and eval results from V20 through V53.
