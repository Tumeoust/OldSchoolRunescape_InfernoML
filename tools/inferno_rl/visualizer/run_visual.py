"""
Run visual simulation with trained model.

Usage:
    python -m tools.inferno_rl.visualizer.run_visual --model path/to/model.zip

Supports two playback modes:
- Auto mode (default): Ticks advance automatically
- Manual mode (press J): Use Left/Right arrows to step through ticks
"""

import argparse
import random
import time
from typing import Optional
import numpy as np

from ..inference_state import StatefulPolicyRunner
from ..simulator.simulator import InfernoSimulator
from ..training.observation import TemporalState, build_observation, update_temporal_state
from ..training.actions import decode_policy_action, get_action_mask, get_mask_for_action_space
from ..training.rewards import InfernoReward
from ..simulator.equipment import Loadout, LoadoutId, LOADOUTS, DEFAULT_LOADOUT
from ..simulator.eval_loadouts import configure_sim_for_eval
from .visualizer import InfernoVisualizer


class _CustomPPOWrapper:
    """Wraps the custom PPO so run_visual can call model.predict() uniformly."""

    def __init__(self, ppo):
        self._ppo = ppo
        self._runner = StatefulPolicyRunner(ppo)
        self.observation_version = ppo.policy_params.observation_version

    def predict(self, obs: np.ndarray, action_masks: np.ndarray = None, deterministic: bool = True):
        prediction = self._runner.predict(obs, action_masks, deterministic=deterministic)
        assert prediction.action is not None
        return decode_policy_action(prediction.action), None

    def reset(self) -> None:
        self._runner.reset()

    @property
    def action_head_sizes(self):
        return self._ppo.policy_params.action_head_sizes


def load_model(model_path: str):
    """Load trained model — supports both custom .pt and SB3 .zip formats."""
    if model_path.endswith(".pt"):
        from ..ppo.ppo import PPO
        ppo = PPO.load(model_path, trainable=False)
        print(f"Loaded custom PPO checkpoint (trained_steps={ppo.meta.trained_steps:,})")
        return _CustomPPOWrapper(ppo)
    try:
        from sb3_contrib import MaskablePPO
        path = model_path.removesuffix(".zip") if model_path.endswith(".zip") else model_path
        return MaskablePPO.load(path)
    except ImportError:
        raise ImportError("sb3_contrib required: pip install sb3-contrib")


def run_visual_simulation(
    model_path: Optional[str] = None,
    start_wave: int = 35,
    max_wave: int = 66,
    fps: int = 4,
    tile_size: int = 20,
    random_actions: bool = False,
    episodes: int = 1,
    seed: Optional[int] = None,
    loadout: Optional[Loadout] = None,
    real_stats: bool = False,
):
    """
    Run visual simulation with trained model or random actions.

    Args:
        model_path: Path to trained model (None for random)
        start_wave: Starting wave
        max_wave: Maximum wave
        fps: Target FPS
        tile_size: Pixel size per tile
        random_actions: Use random actions instead of model
        episodes: Number of episodes to run
        seed: RNG seed for spawn positions (None = different spawns each run)
        loadout: Loadout preset to use (None = default)
    """
    # Create simulator and visualizer
    simulator = InfernoSimulator(start_wave, max_wave)
    simulator.initial_barrage_enabled = True
    if loadout is not None:
        simulator.set_loadout(loadout)
    if real_stats:
        lid = loadout.id if loadout is not None else DEFAULT_LOADOUT.id
        configure_sim_for_eval(simulator, lid)
    visualizer = InfernoVisualizer(tile_size=tile_size, fps=fps)
    reward_calculator = InfernoReward()
    
    # Load model if provided
    model = None
    if model_path and not random_actions:
        print(f"Loading model from {model_path}...")
        model = load_model(model_path)
    
    visualizer.initialize()
    
    try:
        for episode in range(episodes):
            print(f"\n=== Episode {episode + 1}/{episodes} ===")
            print("Controls: J=Toggle manual mode, Left/Right=Step (in manual mode)")
            # Seed spawn RNG so NPC positions vary (or are reproducible with --seed)
            spawn_seed = seed if seed is not None else int(time.time() * 1e6) + episode
            random.seed(spawn_seed)

            # Reset
            simulator.reset()
            visualizer.clear_history()
            total_reward = 0.0
            step = 0
            episode_done = False
            if model is not None and hasattr(model, "reset"):
                model.reset()
            temporal = TemporalState()
            
            # Record initial state
            visualizer.record_frame(simulator.state, 0, 0.0)
            
            while not episode_done:
                # Handle manual mode
                if visualizer.is_manual_mode():
                    # In manual mode, wait for user input
                    while visualizer.is_manual_mode():
                        visualizer._handle_events()
                        
                        if visualizer.has_step_backward_request():
                            # Step backward through history
                            frame = visualizer.step_backward_in_history()
                            if frame:
                                visualizer.restore_state_from_snapshot(simulator.state, frame.state_snapshot)
                                visualizer.last_action = frame.action
                                visualizer.last_reward = frame.reward
                                visualizer.last_reward_breakdown = frame.reward_breakdown
                                # Re-render without advancing total_reward
                                visualizer.render(simulator.state, frame.action, frame.reward)
                            continue
                        
                        if visualizer.has_step_forward_request():
                            if visualizer.can_step_forward_in_history():
                                # Step forward through existing history
                                frame = visualizer.step_forward_in_history()
                                if frame:
                                    visualizer.restore_state_from_snapshot(simulator.state, frame.state_snapshot)
                                    visualizer.last_action = frame.action
                                    visualizer.last_reward = frame.reward
                                    visualizer.last_reward_breakdown = frame.reward_breakdown
                                    visualizer.render(simulator.state, frame.action, frame.reward)
                            else:
                                # Compute new tick
                                break
                            continue
                        
                        # Re-render current state while waiting
                        visualizer.render(simulator.state, visualizer.last_action, visualizer.last_reward)
                        visualizer.clock.tick(30)
                    
                    # If we exited manual mode, continue with normal flow
                    if not visualizer.is_manual_mode():
                        continue
                else:
                    # Check for pause in auto mode
                    if visualizer.is_paused():
                        visualizer.wait_for_unpause()
                
                # Build observation
                obs = build_observation(
                    simulator.state,
                    simulator.get_ticks_in_wave(),
                    temporal=temporal,
                    dead_mobs=simulator.dead_mobs,
                )
                
                # Get action mask
                model_action_heads = getattr(model, "action_head_sizes", [43]) if model is not None else [43]
                mask = get_mask_for_action_space(simulator.state, model_action_heads)
                
                # Choose action
                if model is not None:
                    action, _ = model.predict(obs, action_masks=mask, deterministic=True)
                    action = int(action)
                elif random_actions:
                    # Random valid action
                    valid_actions = np.where(get_action_mask(simulator.state))[0]
                    action = int(np.random.choice(valid_actions))
                else:
                    # Default: stay
                    action = 0
                
                # Execute step
                result = simulator.step(action)
                update_temporal_state(temporal, result.executed_action, result)
                breakdown = reward_calculator.calculate_with_breakdown(result)
                reward = breakdown.total
                total_reward += reward
                step += 1
                
                # Record frame for history (with breakdown)
                visualizer.record_frame(simulator.state, action, reward, breakdown.get_nonzero_components())
                
                # Render
                visualizer.render(simulator.state, action, reward)
                
                # Check terminal
                if result.is_terminal():
                    if result.player_died:
                        print(f"Player died at wave {simulator.state.current_wave}")
                    elif result.wave_timeout:
                        print(f"Wave timeout at wave {simulator.state.current_wave}")
                    elif result.inferno_complete:
                        print(f"Inferno completed!")
                    
                    print(f"Total reward: {total_reward:.1f}")
                    print(f"Steps: {step}")
                    
                    # In manual mode, allow reviewing history after episode ends
                    if visualizer.is_manual_mode():
                        print("Episode ended. Use Left/Right to review history, ESC to quit.")
                        while visualizer.is_manual_mode():
                            visualizer._handle_events()
                            
                            if visualizer.has_step_backward_request():
                                frame = visualizer.step_backward_in_history()
                                if frame:
                                    visualizer.restore_state_from_snapshot(simulator.state, frame.state_snapshot)
                                    visualizer.last_reward_breakdown = frame.reward_breakdown
                                    visualizer.render(simulator.state, frame.action, frame.reward)
                            elif visualizer.has_step_forward_request():
                                frame = visualizer.step_forward_in_history()
                                if frame:
                                    visualizer.restore_state_from_snapshot(simulator.state, frame.state_snapshot)
                                    visualizer.last_reward_breakdown = frame.reward_breakdown
                                    visualizer.render(simulator.state, frame.action, frame.reward)
                            else:
                                visualizer.render(simulator.state, visualizer.last_action, visualizer.last_reward)
                            
                            visualizer.clock.tick(30)
                    else:
                        # Wait a moment before next episode
                        time.sleep(2)
                    
                    episode_done = True
    
    except (SystemExit, KeyboardInterrupt):
        pass
    finally:
        visualizer.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run Inferno Visual Simulation")
    
    parser.add_argument("--model", "-m", type=str, default=None,
                       help="Path to trained model (.zip)")
    parser.add_argument("--start-wave", type=int, default=35,
                       help="Starting wave")
    parser.add_argument("--max-wave", type=int, default=66,
                       help="Maximum wave")
    parser.add_argument("--fps", type=int, default=4,
                       help="Target FPS")
    parser.add_argument("--tile-size", type=int, default=20,
                       help="Pixel size per tile")
    parser.add_argument("--random", action="store_true",
                       help="Use random actions")
    parser.add_argument("--episodes", type=int, default=1,
                       help="Number of episodes")
    parser.add_argument("--seed", type=int, default=None,
                       help="RNG seed for spawn positions (default: random each run)")
    parser.add_argument("--loadout", type=str, default=None,
                       choices=[lid.name for lid in LoadoutId],
                       help="Loadout preset (default: CRYSTAL_BP)")
    parser.add_argument("--real-stats", action="store_true",
                       help="Use real defence levels and equipment defence bonuses (same as eval --real-stats)")

    args = parser.parse_args()

    loadout = LOADOUTS[LoadoutId[args.loadout]] if args.loadout else None

    run_visual_simulation(
        model_path=args.model,
        start_wave=args.start_wave,
        max_wave=args.max_wave,
        fps=args.fps,
        tile_size=args.tile_size,
        random_actions=args.random,
        episodes=args.episodes,
        seed=args.seed,
        loadout=loadout,
        real_stats=args.real_stats,
    )


if __name__ == "__main__":
    main()
