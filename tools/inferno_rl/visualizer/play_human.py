"""
Human Play Mode with Reward Scoring.

Play waves manually with real-time reward breakdown, then optionally compare
against the RL model on the same seed.

Controls:
- Left click tile: Queue movement (pathfinding, 2 tiles/tick running)
- Left click entity: Set attack target
- Right click / C: Clear attack target, movement queue, and queued action
- 1-9 / 0 / - / = / [ / ]: Queue `ATTACK_TARGET_1..14`
- F1-F4: Switch BoFa / Blowpipe / Ice Barrage / Blood Barrage
- Right arrow: Advance one tick
- Left arrow: Step back through history (review only)
- R: Reset episode
- Escape: Quit
"""

import argparse
import json
import os
import random
import time
from typing import Optional, List, Tuple, Dict

import numpy as np
import pygame

from ..simulator.simulator import InfernoSimulator
from ..simulator.entity import PlacedEntity, EntityTypes
from ..simulator.geometry import GRID_WIDTH, GRID_HEIGHT, PILLARS, SimulatorGeometry, InfernoLineOfSight
from ..simulator.pathfinding import OSRSPathfinding
from ..simulator.equipment import GearPreset
from ..training.actions import InfernoAction, get_action_mask, get_mask_for_action_space, NUM_ACTIONS
from ..training.rewards import InfernoReward, RewardBreakdown, normalize_reward_term_name
from ..training.observation import TemporalState, build_observation, update_temporal_state
from ..cli.state_decoder import decode_entities_from_state, decode_pillars, decode_nibblers
from .visualizer import InfernoVisualizer, ENTITY_COLORS, COLOR_BG, COLOR_PLAYER, COLOR_PLAYER_TARGET


# UI colors
COLOR_PANEL_BG = (25, 25, 35)
COLOR_HEADER = (100, 200, 255)
COLOR_TEXT = (220, 220, 220)
COLOR_DIM = (130, 130, 130)
COLOR_POSITIVE = (100, 255, 100)
COLOR_NEGATIVE = (255, 100, 100)
COLOR_HIGHLIGHT = (255, 255, 100)
COLOR_MOVE_PATH = (80, 180, 255)
COLOR_QUEUED_DEST = (80, 180, 255, 128)

_ATTACK_SLOT_KEYMAP = {
    pygame.K_1: 0,
    pygame.K_2: 1,
    pygame.K_3: 2,
    pygame.K_4: 3,
    pygame.K_5: 4,
    pygame.K_6: 5,
    pygame.K_7: 6,
    pygame.K_8: 7,
    pygame.K_9: 8,
    pygame.K_0: 9,
    pygame.K_MINUS: 10,
    pygame.K_EQUALS: 11,
    pygame.K_LEFTBRACKET: 12,
    pygame.K_RIGHTBRACKET: 13,
}


class HumanPlaySession:
    """Interactive human play session with per-tick reward scoring."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        start_wave: int = 55,
        max_wave: int = 66,
        seed: int = 42,
        tile_size: int = 22,
        no_compare: bool = False,
        log_dir: Optional[str] = None,
    ):
        self.model_path = model_path
        self.start_wave = start_wave
        self.max_wave = max_wave
        self.seed = seed
        self.tile_size = tile_size
        self.no_compare = no_compare
        self.log_dir = log_dir

        # Layout
        self.info_panel_width = 320
        self.grid_width = GRID_WIDTH * tile_size
        self.grid_height = GRID_HEIGHT * tile_size
        self.window_width = self.grid_width + self.info_panel_width
        self.window_height = self.grid_height

        # Simulator & reward
        self.simulator = InfernoSimulator(start_wave, max_wave)
        self.simulator.initial_barrage_enabled = False
        self.reward_calculator = InfernoReward()

        # Visualizer — used for drawing primitives only (not render())
        self.visualizer = InfernoVisualizer(tile_size=tile_size)

        # Pygame state
        self.screen: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None
        self.font: Optional[pygame.font.Font] = None
        self.small_font: Optional[pygame.font.Font] = None

        # Movement queue
        self.movement_path: List[Tuple[int, int]] = []
        self.movement_target: Optional[Tuple[int, int]] = None

        # Queued attack action for next tick
        self.queued_action: int = InfernoAction.NO_ACTION_IDX

        # Tick state
        self.total_reward: float = 0.0
        self.tick_count: int = 0
        self.last_breakdown: Optional[RewardBreakdown] = None
        self.last_action: int = InfernoAction.NO_ACTION_IDX
        self.last_reward: float = 0.0
        self.cumulative_categories: Dict[str, float] = {}

        # History for backward stepping
        self.history: List[dict] = []
        self.history_index: int = -1

        # Episode result storage (for comparison)
        self.episode_result: Optional[dict] = None

        # File logging state
        self._wave_tick_logs: List[dict] = []
        self._current_log_wave: int = 0

    # ------------------------------------------------------------------
    # Pygame init / close
    # ------------------------------------------------------------------

    def _init_pygame(self):
        pygame.init()
        pygame.display.set_caption("Inferno — Human Play (Reward Scoring)")
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 15)
        self.small_font = pygame.font.SysFont("monospace", 12)
        # Share screen/fonts with visualizer so its draw methods work
        self.visualizer.screen = self.screen
        self.visualizer.clock = self.clock
        self.visualizer.font = self.font
        self.visualizer.small_font = self.small_font

    def _close_pygame(self):
        if pygame.get_init():
            pygame.quit()

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def run(self):
        """Run human play session, then optionally compare with model."""
        self._init_pygame()
        try:
            human_result = self._play_human_episode()
            if human_result and not self.no_compare and self.model_path:
                model_result = self._run_model_episode()
                self._print_comparison(human_result, model_result)
            elif human_result:
                print("\n=== HUMAN RESULT ===")
                self._print_single_result(human_result, "Human")
        except (SystemExit, KeyboardInterrupt):
            pass
        finally:
            self._close_pygame()

    # ------------------------------------------------------------------
    # Human episode
    # ------------------------------------------------------------------

    def _play_human_episode(self) -> Optional[dict]:
        """Run the human-controlled episode. Returns result dict or None."""
        random.seed(self.seed)
        self.simulator.reset()
        self.total_reward = 0.0
        self.tick_count = 0
        self.cumulative_categories.clear()
        self.history.clear()
        self.history_index = -1
        self.queued_action = InfernoAction.NO_ACTION_IDX
        self.movement_path = []
        self.movement_target = None
        self.last_breakdown = None
        self.last_action = InfernoAction.NO_ACTION_IDX
        self.last_reward = 0.0

        # Init file logging
        self._wave_tick_logs = []
        self._current_log_wave = self.start_wave
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)

        # Record initial frame
        self._record_history_frame(InfernoAction.NO_ACTION_IDX, 0.0, None)

        # Render initial state
        self._render_frame()

        done = False
        while not done:
            action = self._wait_for_tick_advance()
            if action is None:
                return None  # quit

            if action == "RESET":
                return self._play_human_episode()

            if action == "BACK":
                self._step_backward()
                self._render_frame()
                continue

            # Advance tick
            result = self._advance_tick(action)
            done = result.is_terminal()

            if done:
                term_type = "DEATH" if result.player_died else "TIMEOUT" if result.wave_timeout else "COMPLETE"
                self._flush_wave_log(terminal_type=term_type)
                self._write_episode_summary(term_type)
                print(f"\nEpisode ended: {term_type} at wave {self.simulator.state.current_wave}, "
                      f"tick {self.tick_count}, total reward {self.total_reward:+.1f}")

                # Allow history review after death
                self._post_episode_review()

        return {
            "max_wave": self.simulator.state.current_wave,
            "ticks": self.tick_count,
            "total_reward": self.total_reward,
            "categories": dict(self.cumulative_categories),
        }

    def _wait_for_tick_advance(self) -> Optional:
        """Block until user presses Right arrow (or another action key). Returns action int, 'BACK', 'RESET', or None."""
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if event.type == pygame.KEYDOWN:
                    result = self._handle_keydown(event)
                    if result is not None:
                        return result
                if event.type == pygame.MOUSEBUTTONDOWN:
                    self._handle_mouse(event)
                    self._render_frame()

            self._render_frame()
            self.clock.tick(30)

    def _handle_keydown(self, event) -> Optional:
        """Handle key press. Returns action/command or None if not a tick-advancing key."""
        key = event.key

        if key == pygame.K_ESCAPE:
            return None

        if key == pygame.K_r:
            return "RESET"

        if key == pygame.K_LEFT:
            return "BACK"

        # Tick advance: Right arrow submits the queued action
        if key == pygame.K_RIGHT:
            action = self.queued_action
            self.queued_action = InfernoAction.NO_ACTION_IDX
            return action

        # Exact target slot keys queue for next tick
        if key in _ATTACK_SLOT_KEYMAP:
            self.queued_action = InfernoAction.action_for_target_index(_ATTACK_SLOT_KEYMAP[key])

        # Clear target / movement / queued action
        elif key == pygame.K_c:
            self.simulator.state.attack_target = None
            self.movement_path = []
            self.movement_target = None
            self.queued_action = InfernoAction.NO_ACTION_IDX

        # Gear switches — queue for next tick
        elif key == pygame.K_F1:
            self.queued_action = InfernoAction.SWITCH_BOFA
        elif key == pygame.K_F2:
            self.queued_action = InfernoAction.SWITCH_BLOWPIPE
        elif key == pygame.K_F3:
            self.queued_action = InfernoAction.SWITCH_ICE_BARRAGE
        elif key == pygame.K_F4:
            self.queued_action = InfernoAction.SWITCH_BLOOD_BARRAGE

        return None  # Not a tick-advance key

    def _handle_mouse(self, event):
        """Left click tile = queue move, left click entity = set target, right click = clear."""
        mx, my = event.pos

        # Right click: clear attack target and movement queue
        if event.button == 3:
            self.simulator.state.attack_target = None
            self.movement_path = []
            self.movement_target = None
            self.queued_action = InfernoAction.NO_ACTION_IDX
            return

        gx, gy = self._screen_to_grid(mx, my)
        if gx is None:
            return

        # Check if an entity is at this position
        entity = self._get_entity_at(gx, gy)
        if entity:
            # Set as attack target directly on state
            self.simulator.state.attack_target = entity
            print(f"Target: {entity.entity_type.name} at ({entity.x}, {entity.y})")
        else:
            # Queue movement
            self._queue_movement(gx, gy)

    # ------------------------------------------------------------------
    # Grid / entity helpers
    # ------------------------------------------------------------------

    def _screen_to_grid(self, mx: int, my: int) -> Tuple[Optional[int], Optional[int]]:
        if mx >= self.grid_width:
            return None, None
        gx = mx // self.tile_size
        gy = (self.grid_height - my) // self.tile_size
        if 0 <= gx < GRID_WIDTH and 0 <= gy < GRID_HEIGHT:
            return gx, gy
        return None, None

    def _get_entity_at(self, gx: int, gy: int) -> Optional[PlacedEntity]:
        # Return the smallest entity at this tile so nibblers (1x1) can be
        # clicked even when overlapped by larger NPCs (mager 4x4, etc.)
        best = None
        for entity in self.simulator.state.entities:
            if entity.is_dead():
                continue
            size = entity.entity_type.size_in_tiles
            if entity.x <= gx < entity.x + size and entity.y <= gy < entity.y + size:
                if best is None or size < best.entity_type.size_in_tiles:
                    best = entity
        return best

    def _queue_movement(self, target_x: int, target_y: int):
        state = self.simulator.state
        if not SimulatorGeometry.is_valid_tile(target_x, target_y, state.pillar_alive):
            return

        def checker(x: int, y: int, size: int) -> bool:
            return SimulatorGeometry.is_valid_tile(x, y, state.pillar_alive)

        path = OSRSPathfinding.find_player_path(
            state.player_x, state.player_y,
            target_x, target_y,
            checker,
        )
        if path:
            self.movement_path = path
            self.movement_target = (target_x, target_y)

    def _process_queued_movement(self):
        if not self.movement_path:
            return
        # Validate next step is adjacent
        next_pos = self.movement_path[0]
        dx = abs(next_pos[0] - self.simulator.state.player_x)
        dy = abs(next_pos[1] - self.simulator.state.player_y)
        if dx > 1 or dy > 1:
            self.movement_path = []
            self.movement_target = None
            return
        # Move up to 2 tiles per tick (running)
        tiles_moved = 0
        while self.movement_path and tiles_moved < 2:
            next_pos = self.movement_path.pop(0)
            self.simulator.state.player_x = next_pos[0]
            self.simulator.state.player_y = next_pos[1]
            tiles_moved += 1
        if not self.movement_path:
            self.movement_target = None

    # ------------------------------------------------------------------
    # Tick advancement
    # ------------------------------------------------------------------

    def _advance_tick(self, action: int):
        """Process movement, step sim, calc rewards, render."""
        is_attack = InfernoAction.is_attack(action)
        if not is_attack:
            self._process_queued_movement()
        elif self.movement_path:
            # Attack cancels movement (OSRS behavior)
            self.movement_path = []
            self.movement_target = None

        result = self.simulator.step(action)
        breakdown = self.reward_calculator.calculate_with_breakdown(result)
        reward = breakdown.total

        self.total_reward += reward
        self.tick_count += 1
        self.last_breakdown = breakdown
        self.last_action = action
        self.last_reward = reward

        # Accumulate per-category totals
        for name, value in breakdown.get_nonzero_components():
            norm_name = normalize_reward_term_name(name)
            self.cumulative_categories[norm_name] = self.cumulative_categories.get(norm_name, 0.0) + value

        # File logging
        self._log_tick(action, result, breakdown)
        if result.wave_completed:
            self._flush_wave_log()
            self._current_log_wave = self.simulator.state.current_wave

        self._record_history_frame(action, reward, breakdown)
        self._render_frame()
        return result

    # ------------------------------------------------------------------
    # File logging
    # ------------------------------------------------------------------

    def _log_tick(self, action: int, result, breakdown: RewardBreakdown):
        if not self.log_dir:
            return
        state = self.simulator.state
        entry = {
            "tick": self.tick_count,
            "wave": state.current_wave,
            "tick_in_wave": self.simulator.get_ticks_in_wave(),
            "action": self.visualizer._get_action_name(action),
            "player_pos": [state.player_x, state.player_y],
            "player_hp": result.health_at_step_start,
            "weapon": state.current_preset.value,
            "enemies_remaining": result.enemies_remaining,
            "npcs_with_los": result.npcs_with_los_now,
            "entities": decode_entities_from_state(state),
            "nibblers": decode_nibblers(state),
            "pillars": decode_pillars(state),
            "tick_reward": round(breakdown.total, 4),
            "cumulative_reward": round(self.total_reward, 4),
            "components": {name: round(value, 4) for name, value in breakdown.get_nonzero_components()},
        }
        self._wave_tick_logs.append(entry)

    def _flush_wave_log(self, terminal_type: Optional[str] = None):
        if not self.log_dir or not self._wave_tick_logs:
            return

        wave = self._current_log_wave
        filepath = os.path.join(self.log_dir, f"wave_{wave:02d}.json")

        wave_reward = sum(t["tick_reward"] for t in self._wave_tick_logs)
        category_totals: Dict[str, float] = {}
        for tick_entry in self._wave_tick_logs:
            for name, value in tick_entry["components"].items():
                norm = normalize_reward_term_name(name)
                category_totals[norm] = category_totals.get(norm, 0.0) + value

        wave_data = {
            "wave": wave,
            "ticks": len(self._wave_tick_logs),
            "total_reward": round(wave_reward, 4),
            "terminal": terminal_type,
            "category_totals": {k: round(v, 4) for k, v in category_totals.items()},
            "ticks_log": self._wave_tick_logs,
        }

        with open(filepath, "w") as f:
            json.dump(wave_data, f, indent=2)

        print(f"  Logged wave {wave}: {len(self._wave_tick_logs)} ticks, reward {wave_reward:+.1f} -> {filepath}")
        self._wave_tick_logs = []

    def _write_episode_summary(self, term_type: str):
        if not self.log_dir:
            return

        summary = {
            "seed": self.seed,
            "start_wave": self.start_wave,
            "max_wave_reached": self.simulator.state.current_wave,
            "total_ticks": self.tick_count,
            "total_reward": round(self.total_reward, 4),
            "terminal": term_type,
            "category_totals": {k: round(v, 4) for k, v in self.cumulative_categories.items()},
        }

        filepath = os.path.join(self.log_dir, "episode_summary.json")
        with open(filepath, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Episode summary -> {filepath}")

    # ------------------------------------------------------------------
    # History (backward stepping)
    # ------------------------------------------------------------------

    def _record_history_frame(self, action: int, reward: float, breakdown: Optional[RewardBreakdown]):
        snapshot = self.visualizer._snapshot_state(self.simulator.state)
        frame = {
            "snapshot": snapshot,
            "action": action,
            "reward": reward,
            "breakdown": breakdown,
            "total_reward": self.total_reward,
            "tick_count": self.tick_count,
            "cumulative_categories": dict(self.cumulative_categories),
            "movement_path": list(self.movement_path),
            "movement_target": self.movement_target,
        }
        # Truncate future if we stepped back then advanced
        if self.history_index < len(self.history) - 1:
            self.history = self.history[:self.history_index + 1]
        self.history.append(frame)
        self.history_index = len(self.history) - 1

    def _step_backward(self):
        if self.history_index <= 0:
            return
        self.history_index -= 1
        frame = self.history[self.history_index]
        self.visualizer.restore_state_from_snapshot(self.simulator.state, frame["snapshot"])
        self.last_action = frame["action"]
        self.last_reward = frame["reward"]
        self.last_breakdown = frame["breakdown"]
        self.total_reward = frame["total_reward"]
        self.tick_count = frame["tick_count"]
        self.cumulative_categories = dict(frame["cumulative_categories"])
        self.movement_path = list(frame["movement_path"])
        self.movement_target = frame["movement_target"]

    def _post_episode_review(self):
        """After episode ends, allow Left/Right to review history."""
        reviewing = True
        while reviewing:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return
                    if event.key == pygame.K_LEFT:
                        self._step_backward()
                    if event.key == pygame.K_RIGHT and self.history_index < len(self.history) - 1:
                        self.history_index += 1
                        frame = self.history[self.history_index]
                        self.visualizer.restore_state_from_snapshot(self.simulator.state, frame["snapshot"])
                        self.last_action = frame["action"]
                        self.last_reward = frame["reward"]
                        self.last_breakdown = frame["breakdown"]
                        self.total_reward = frame["total_reward"]
                        self.tick_count = frame["tick_count"]
                        self.cumulative_categories = dict(frame["cumulative_categories"])
                    if event.key == pygame.K_r:
                        return  # will restart in caller
            self._render_frame()
            self.clock.tick(30)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_frame(self):
        """Draw grid (via visualizer primitives) + custom reward panel."""
        self.screen.fill(COLOR_BG)

        # Reuse visualizer drawing methods
        self.visualizer._draw_grid()
        self.visualizer._draw_pillars(self.simulator.state)
        self.visualizer._draw_entities(self.simulator.state)
        self.visualizer._draw_player(self.simulator.state)

        # Draw movement path overlay
        self._draw_movement_path()

        # Draw reward panel
        self._draw_reward_panel()

        pygame.display.flip()

    def _draw_movement_path(self):
        """Draw queued movement path on the grid."""
        if not self.movement_path:
            return
        ts = self.tile_size
        prev = (self.simulator.state.player_x, self.simulator.state.player_y)
        for pos in self.movement_path:
            sx = pos[0] * ts + ts // 2
            sy = self.grid_height - (pos[1] + 1) * ts + ts // 2
            pygame.draw.circle(self.screen, COLOR_MOVE_PATH, (sx, sy), 3)
            prev = pos
        # Highlight destination
        if self.movement_target:
            dx = self.movement_target[0] * ts
            dy = self.grid_height - (self.movement_target[1] + 1) * ts
            pygame.draw.rect(self.screen, COLOR_MOVE_PATH, (dx, dy, ts, ts), 2)

    def _draw_reward_panel(self):
        """Draw the right-side reward/controls panel."""
        panel_x = self.grid_width
        # Panel background
        pygame.draw.rect(self.screen, COLOR_PANEL_BG,
                         (panel_x, 0, self.info_panel_width, self.window_height))

        x = panel_x + 10
        y = 8
        line_h = 17
        small_h = 14

        def text(msg, color=COLOR_TEXT, small=False):
            nonlocal y
            f = self.small_font if small else self.font
            self.screen.blit(f.render(msg, True, color), (x, y))
            y += small_h if small else line_h

        def section(title):
            nonlocal y
            y += 4
            text(f"=== {title} ===", COLOR_HEADER)

        # --- Wave / Player ---
        section("STATE")
        state = self.simulator.state
        text(f"Wave {state.current_wave}  Tick {self.tick_count}")
        hp_color = COLOR_POSITIVE if state.player_health > 50 else COLOR_HIGHLIGHT if state.player_health > 25 else COLOR_NEGATIVE
        text(f"HP {state.player_health}/99  Pos ({state.player_x},{state.player_y})", hp_color)
        text(f"Weapon: {state.current_preset.value}", COLOR_DIM, small=True)

        # Queued action display
        if self.queued_action != InfernoAction.NO_ACTION_IDX:
            action_name = self.visualizer._get_action_name(self.queued_action)
            text(f"Queued: {action_name}", COLOR_HIGHLIGHT, small=True)

        # --- This tick ---
        section("THIS TICK")
        action_name = self.visualizer._get_action_name(self.last_action)
        text(f"Action: {action_name}")
        r_color = COLOR_POSITIVE if self.last_reward > 0 else COLOR_NEGATIVE if self.last_reward < 0 else COLOR_DIM
        text(f"Tick Reward: {self.last_reward:+.2f}", r_color)
        total_color = COLOR_POSITIVE if self.total_reward > 0 else COLOR_NEGATIVE if self.total_reward < 0 else COLOR_DIM
        text(f"Running Total: {self.total_reward:+.1f}", total_color)

        # --- Breakdown ---
        section("BREAKDOWN")
        if self.last_breakdown:
            components = self.last_breakdown.get_nonzero_components()
            sorted_comp = sorted(components, key=lambda c: abs(c[1]), reverse=True)
            for name, value in sorted_comp[:10]:
                c = COLOR_POSITIVE if value > 0 else COLOR_NEGATIVE
                display = name[:24] if len(name) > 24 else name
                text(f"{value:+.1f} {display}", c, small=True)
        else:
            text("(no tick yet)", COLOR_DIM, small=True)

        # --- Cumulative categories ---
        section("CUMULATIVE")
        if self.cumulative_categories:
            sorted_cats = sorted(self.cumulative_categories.items(), key=lambda c: abs(c[1]), reverse=True)
            for name, value in sorted_cats[:10]:
                c = COLOR_POSITIVE if value > 0 else COLOR_NEGATIVE
                display = name[:20] if len(name) > 20 else name
                text(f"{value:+.1f} {display}", c, small=True)
        else:
            text("(none)", COLOR_DIM, small=True)

        # --- History position ---
        if self.history:
            hist_text = f"History: {self.history_index + 1}/{len(self.history)}"
            text(hist_text, COLOR_DIM, small=True)

        # --- Controls ---
        y = self.window_height - 120
        section("CONTROLS")
        text("LClick tile:Move  LClick NPC:Target", COLOR_DIM, small=True)
        text("RClick/C:Clear target+move+queue", COLOR_DIM, small=True)
        text("1-9/0/-/=/[/]:Attack slots 1-14", COLOR_DIM, small=True)
        text("F1-F4:Gear  Right:Tick  Left:Back", COLOR_DIM, small=True)
        text("R:Reset  ESC:Quit", COLOR_DIM, small=True)

    # ------------------------------------------------------------------
    # Model comparison
    # ------------------------------------------------------------------

    def _run_model_episode(self) -> Optional[dict]:
        """Silently run the model on the same seed."""
        from .run_visual import load_model

        print(f"\nRunning model comparison on seed {self.seed}...")
        model = load_model(self.model_path)

        sim = InfernoSimulator(self.start_wave, self.max_wave)
        sim.initial_barrage_enabled = False
        reward_calc = InfernoReward()

        random.seed(self.seed)
        sim.reset()
        if hasattr(model, "reset"):
            model.reset()
        temporal = TemporalState()

        total_reward = 0.0
        tick_count = 0
        categories: Dict[str, float] = {}
        done = False

        while not done:
            obs = build_observation(
                sim.state,
                sim.get_ticks_in_wave(),
                temporal=temporal,
                dead_mobs=sim.dead_mobs,
            )
            mask = get_mask_for_action_space(
                sim.state,
                getattr(model, "action_head_sizes", [43]),
            )
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            action = int(action)

            result = sim.step(action)
            update_temporal_state(temporal, result.executed_action, result)
            breakdown = reward_calc.calculate_with_breakdown(result)
            total_reward += breakdown.total
            tick_count += 1

            for name, value in breakdown.get_nonzero_components():
                norm = normalize_reward_term_name(name)
                categories[norm] = categories.get(norm, 0.0) + value

            done = result.is_terminal()

        term = "DEATH" if result.player_died else "TIMEOUT" if result.wave_timeout else "COMPLETE"
        print(f"Model finished: {term} at wave {sim.state.current_wave}, "
              f"tick {tick_count}, reward {total_reward:+.1f}")

        return {
            "max_wave": sim.state.current_wave,
            "ticks": tick_count,
            "total_reward": total_reward,
            "categories": categories,
        }

    def _print_comparison(self, human: dict, model: dict):
        """Print side-by-side comparison table."""
        print(f"\n{'=' * 55}")
        print(f"  COMPARISON: Seed {self.seed}, Waves {self.start_wave}-{self.max_wave}")
        print(f"{'=' * 55}")
        print(f"{'':20s} {'Human':>12s} {'Model':>12s}")
        print(f"{'-' * 55}")
        print(f"{'Max wave reached':20s} {human['max_wave']:>12d} {model['max_wave']:>12d}")
        print(f"{'Ticks survived':20s} {human['ticks']:>12d} {model['ticks']:>12d}")
        print(f"{'Total reward':20s} {human['total_reward']:>+12.1f} {model['total_reward']:>+12.1f}")
        print()

        # Merge category keys
        all_cats = sorted(
            set(human["categories"].keys()) | set(model["categories"].keys()),
            key=lambda k: abs(human["categories"].get(k, 0.0)) + abs(model["categories"].get(k, 0.0)),
            reverse=True,
        )
        print("Top categories:")
        for cat in all_cats[:12]:
            h_val = human["categories"].get(cat, 0.0)
            m_val = model["categories"].get(cat, 0.0)
            display = cat[:20] if len(cat) > 20 else cat
            print(f"  {display:20s} {h_val:>+12.1f} {m_val:>+12.1f}")
        print()

    def _print_single_result(self, result: dict, label: str):
        """Print single result summary."""
        print(f"  Max wave: {result['max_wave']}")
        print(f"  Ticks: {result['ticks']}")
        print(f"  Total reward: {result['total_reward']:+.1f}")
        if result["categories"]:
            sorted_cats = sorted(result["categories"].items(), key=lambda c: abs(c[1]), reverse=True)
            print("  Categories:")
            for name, value in sorted_cats[:10]:
                print(f"    {value:+8.1f}  {name}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Human Play Mode — play Inferno waves with reward scoring",
    )
    parser.add_argument("--model", "-m", type=str, default=None,
                        help="Path to trained model (.pt) for comparison")
    parser.add_argument("--start-wave", type=int, default=55,
                        help="Starting wave (default: 55)")
    parser.add_argument("--max-wave", type=int, default=66,
                        help="Maximum wave (default: 66)")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for spawns (default: 42)")
    parser.add_argument("--tile-size", type=int, default=22,
                        help="Tile size in pixels (default: 22)")
    parser.add_argument("--no-compare", action="store_true",
                        help="Skip model comparison run")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="Directory for per-tick/per-wave reward logs (default: no logging)")

    args = parser.parse_args()

    session = HumanPlaySession(
        model_path=args.model,
        start_wave=args.start_wave,
        max_wave=args.max_wave,
        seed=args.seed,
        tile_size=args.tile_size,
        no_compare=args.no_compare,
        log_dir=args.log_dir,
    )
    session.run()


if __name__ == "__main__":
    main()
