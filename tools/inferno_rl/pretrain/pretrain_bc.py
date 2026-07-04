"""
Behaviour-cloning pre-training from collected data.

Loads observations/actions/action_masks from a .npz file produced by
collect_from_model.py. Creates a PPO instance with
the target architecture, runs supervised training, then saves a checkpoint
compatible with `train_gpu.py --load`.

If the .npz contains a 'logits' array (from collect_from_model.py), uses
KL-divergence distillation to match the teacher's full action distribution.
Otherwise falls back to cross-entropy on hard action labels.

Usage:
    python -m tools.inferno_rl.pretrain.pretrain_bc \
        --data models/bc_data/v21_rollouts.npz \
        --actor-sizes 512,512 --critic-sizes 512,512 \
        --epochs 10 \
        --lr 1e-3 \
        --batch-size 512 \
        --save models/bc_warmstart_512.pt \
        --normalize-obs
"""

import argparse
import os

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from ..ppo.ppo import PPO, PolicyParams
from ..ppo.mlp_helper import MlpConfig, default_mlp_config
from ..training.observation import get_observation_size, get_public_observation_size
from ..training.actions import (
    ACTION_HEAD_SIZES,
    POLICY_ACTION_DEPENDENCIES,
    encode_policy_action,
    legacy_action_mask_to_policy_mask,
    uses_factored_policy_actions,
)


def _build_policy_params(
    actor_sizes: list[int] | None = None,
    critic_sizes: list[int] | None = None,
    lstm_hidden_size: int | None = 128,
    lstm_seq_len: int = 16,
) -> PolicyParams:
    """Policy architecture matching train_gpu.py exactly."""
    return PolicyParams(
        max_sequence_length=1,
        actor_input_size=get_public_observation_size(),
        critic_input_size=get_observation_size(),
        action_head_sizes=ACTION_HEAD_SIZES,
        actor_config=default_mlp_config(actor_sizes or [512, 512]),
        critic_config=default_mlp_config(critic_sizes or [512, 512]),
        feature_extractor_config=MlpConfig(),
        action_dependencies=POLICY_ACTION_DEPENDENCIES,
        autoregressive_actions=True,
        lstm_hidden_size=lstm_hidden_size,
        lstm_seq_len=lstm_seq_len,
        policy_arch="flat_lstm_residual",
    )


def pretrain(
    data_path: str,
    save_path: str,
    n_epochs: int = 10,
    lr: float = 1e-3,
    batch_size: int = 512,
    device: str = "cpu",
    normalize_obs: bool = False,
    log_dir: str = "logs/bc_pretrain",
    actor_sizes: list[int] | None = None,
    critic_sizes: list[int] | None = None,
    lstm_hidden_size: int | None = 128,
    lstm_seq_len: int = 16,
) -> None:
    # Load dataset
    print(f"Loading BC data from {data_path}")
    data = np.load(data_path)
    observations = data["observations"]   # (N, obs_dim)
    actions = np.asarray(data["actions"], dtype=np.int32)
    action_masks = np.asarray(data["action_masks"], dtype=bool)
    teacher_logits = data["logits"] if "logits" in data else None

    N = len(observations)
    print(f"  Dataset size: {N} steps")
    print(f"  observations: {observations.shape}  dtype={observations.dtype}")
    print(f"  actions:      {actions.shape}  dtype={actions.dtype}")
    print(f"  action_masks: {action_masks.shape}  dtype={action_masks.dtype}")
    if teacher_logits is not None:
        print(f"  logits:       {teacher_logits.shape}  dtype={teacher_logits.dtype} (distillation mode)")
    else:
        print(f"  logits:       not present (cross-entropy mode)")

    policy_params = _build_policy_params(
        actor_sizes=actor_sizes, critic_sizes=critic_sizes,
        lstm_hidden_size=lstm_hidden_size, lstm_seq_len=lstm_seq_len,
    )
    if uses_factored_policy_actions(policy_params.action_head_sizes):
        if action_masks.shape[-1] == 43:
            raise ValueError(
                "The BC dataset uses the obsolete 43-action layout. "
                "Recollect it with the exact-target 52-action interface."
            )
        if teacher_logits is not None and teacher_logits.shape[-1] == 43:
            raise ValueError(
                "The BC dataset includes obsolete 43-action teacher logits. "
                "Recollect it with the exact-target interface."
            )
        if actions.ndim == 1:
            actions = np.stack(
                [encode_policy_action(int(action)) for action in actions],
                axis=0,
            )
        if action_masks.shape[-1] == 52:
            action_masks = legacy_action_mask_to_policy_mask(action_masks)
        elif action_masks.shape[-1] != sum(policy_params.action_head_sizes):
            raise ValueError(
                f"Unsupported action mask width {action_masks.shape[-1]}. "
                f"Expected 52 legacy entries or {sum(policy_params.action_head_sizes)} factored entries."
            )
        if teacher_logits is not None and teacher_logits.shape[-1] != sum(policy_params.action_head_sizes):
            print("  teacher logits do not match the target factored action space; using hard labels only")
            teacher_logits = None

    # Create PPO with matching architecture
    ppo = PPO.new_instance(policy_params, device=device, normalize_observations=normalize_obs)
    print(f"\nPolicy:\n{ppo}")

    # TensorBoard writer
    os.makedirs(log_dir, exist_ok=True)
    from datetime import datetime
    run_name = f"bc_pretrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    summary_writer = SummaryWriter(log_dir=os.path.join(log_dir, run_name))

    # Run BC pre-training
    print(f"\nPre-training: {n_epochs} epochs, batch_size={batch_size}, lr={lr}")
    ppo.pretrain_bc(
        observations=observations,
        actions=actions,
        action_masks=action_masks,
        n_epochs=n_epochs,
        batch_size=batch_size,
        learning_rate=lr,
        summary_writer=summary_writer,
        teacher_logits=teacher_logits,
    )

    # Save checkpoint in train_gpu.py-compatible format
    ppo.save(save_path)
    print(f"\nSaved BC warm-start checkpoint to {save_path}")
    summary_writer.close()


def main() -> None:
    import torch as th

    parser = argparse.ArgumentParser(description="BC pre-training from heuristic data")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to .npz file from collect_from_model.py")
    parser.add_argument("--save", type=str, default="models/bc_warmstart.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", type=str,
                        default="cuda" if th.cuda.is_available() else "cpu")
    parser.add_argument("--normalize-obs", action="store_true")
    parser.add_argument("--log-dir", type=str, default="logs/bc_pretrain")
    parser.add_argument("--actor-sizes", type=str, default=None,
                        help="Actor hidden layer sizes, comma-separated (default: 512,512)")
    parser.add_argument("--critic-sizes", type=str, default=None,
                        help="Critic hidden layer sizes, comma-separated (default: 512,512)")
    parser.add_argument("--lstm-hidden-size", type=int, default=128,
                        help="LSTM hidden size (default: 128)")
    parser.add_argument("--lstm-seq-len", type=int, default=16,
                        help="LSTM sequence length (default: 16)")
    args = parser.parse_args()

    actor_sizes = [int(x) for x in args.actor_sizes.split(",")] if args.actor_sizes else None
    critic_sizes = [int(x) for x in args.critic_sizes.split(",")] if args.critic_sizes else None

    pretrain(
        data_path=args.data,
        save_path=args.save,
        n_epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=args.device,
        normalize_obs=args.normalize_obs,
        log_dir=args.log_dir,
        actor_sizes=actor_sizes,
        critic_sizes=critic_sizes,
        lstm_hidden_size=args.lstm_hidden_size,
        lstm_seq_len=args.lstm_seq_len,
    )


if __name__ == "__main__":
    main()
