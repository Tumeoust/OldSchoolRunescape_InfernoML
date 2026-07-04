"""
Pygame-based visual simulator for Inferno RL.

Provides real-time visualization of:
- Arena with pillars
- Player and NPCs
- Attack targets and LOS
- Model decisions

Supports two playback modes:
- Auto mode (default): Ticks advance automatically
- Manual mode: Press Left/Right arrows to step through ticks
"""

from typing import Optional, Tuple, Dict, List, Any
from dataclasses import dataclass
import numpy as np

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

from ..simulator.simulator import InfernoSimulator
from ..simulator.entity import EntityTypes, PlacedEntity
from ..simulator.geometry import GRID_WIDTH, GRID_HEIGHT, PILLARS, SimulatorGeometry
from ..simulator.state import SimulatorState
from ..simulator.equipment import GearPreset
from ..testing.actions import InfernoAction


# Colors
COLOR_BG = (40, 40, 40)
COLOR_GRID = (60, 60, 60)
COLOR_PILLAR = (139, 119, 101)
COLOR_PILLAR_DEAD = (80, 80, 80)
COLOR_PLAYER = (0, 200, 0)
COLOR_PLAYER_TARGET = (255, 255, 0)
COLOR_MODE_AUTO = (100, 255, 100)
COLOR_MODE_MANUAL = (255, 200, 100)

# Entity colors
ENTITY_COLORS = {
    EntityTypes.NIBBLER: (200, 150, 100),
    EntityTypes.BAT: (150, 100, 200),
    EntityTypes.BLOB: (100, 200, 100),
    EntityTypes.BLOB_MAGE: (100, 100, 200),
    EntityTypes.BLOB_RANGE: (100, 200, 100),
    EntityTypes.BLOB_MELEE: (200, 100, 100),
    EntityTypes.RANGER: (0, 200, 200),
    EntityTypes.MAGER: (200, 0, 200),
    EntityTypes.MELEE: (200, 100, 50),
    EntityTypes.JAD: (255, 0, 0),
    EntityTypes.HEALER: (255, 200, 200),
    EntityTypes.ZUK: (255, 100, 0),
}


@dataclass
class HistoryFrame:
    """Stores a snapshot of simulation state for replay."""
    state_snapshot: dict
    action: int
    reward: float
    tick: int
    reward_breakdown: Optional[List[Tuple[str, float]]] = None  # List of (name, value) tuples


class InfernoVisualizer:
    """
    Pygame-based visualizer for the Inferno simulator.
    
    Supports two playback modes:
    - Auto mode (default): Ticks advance automatically at target FPS
    - Manual mode: User controls tick advancement with arrow keys
    """
    
    def __init__(
        self,
        tile_size: int = 20,
        info_panel_width: int = 575,
        fps: int = 4
    ):
        """
        Create visualizer.
        
        Args:
            tile_size: Pixel size per tile
            info_panel_width: Width of info panel
            fps: Target frames per second (ticks per second)
        """
        if not PYGAME_AVAILABLE:
            raise ImportError("pygame is required for visualization: pip install pygame")
        
        self.tile_size = tile_size
        self.info_panel_width = info_panel_width
        self.fps = fps
        
        # Calculate window size
        self.grid_width = GRID_WIDTH * tile_size
        self.grid_height = GRID_HEIGHT * tile_size
        self.window_width = self.grid_width + info_panel_width
        self.window_height = self.grid_height
        
        # Pygame state
        self.screen: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None
        self.font: Optional[pygame.font.Font] = None
        self.small_font: Optional[pygame.font.Font] = None
        
        # Simulation state
        self.last_action: int = 0
        self.last_reward: float = 0.0
        self.total_reward: float = 0.0
        self.step_count: int = 0
        self.paused: bool = False
        
        # Manual mode state
        self.manual_mode: bool = False
        self.step_forward_request: bool = False
        self.step_backward_request: bool = False
        self.history: List[HistoryFrame] = []
        self.history_index: int = -1
        self.max_history_size: int = 10000

        # Review mode navigation
        self.jump_start_request: bool = False
        self.jump_end_request: bool = False
        self.next_seed_request: bool = False
        self.review_info: Optional[Dict[str, Any]] = None

        # Reward breakdown for display
        self.last_reward_breakdown: Optional[List[Tuple[str, float]]] = None

    def initialize(self):
        """Initialize pygame."""
        pygame.init()
        pygame.display.set_caption("Inferno Simulator")
        
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 16)
        self.small_font = pygame.font.SysFont("monospace", 12)

    def close(self):
        """Close pygame."""
        if pygame.get_init():
            pygame.quit()

    def toggle_manual_mode(self):
        """Toggle between auto and manual playback modes."""
        self.manual_mode = not self.manual_mode
        mode_name = "MANUAL" if self.manual_mode else "AUTO"
        print(f"Playback mode: {mode_name}")
        
    def is_manual_mode(self) -> bool:
        """Check if in manual mode."""
        return self.manual_mode

    def has_step_forward_request(self) -> bool:
        """Check if forward step was requested (and consume the request)."""
        if self.step_forward_request:
            self.step_forward_request = False
            return True
        return False
    
    def has_step_backward_request(self) -> bool:
        """Check if backward step was requested (and consume the request)."""
        if self.step_backward_request:
            self.step_backward_request = False
            return True
        return False

    def has_jump_start_request(self) -> bool:
        """Check if jump-to-start was requested (HOME key, consume)."""
        if self.jump_start_request:
            self.jump_start_request = False
            return True
        return False

    def has_jump_end_request(self) -> bool:
        """Check if jump-to-end was requested (END key, consume)."""
        if self.jump_end_request:
            self.jump_end_request = False
            return True
        return False

    def has_next_seed_request(self) -> bool:
        """Check if next-seed was requested (N/ENTER key, consume)."""
        if self.next_seed_request:
            self.next_seed_request = False
            return True
        return False

    def record_frame(self, state: SimulatorState, action: int, reward: float, 
                     reward_breakdown: Optional[List[Tuple[str, float]]] = None):
        """
        Record current state for history (enables backward stepping).
        
        Args:
            state: Current simulator state to snapshot
            action: Action taken this tick
            reward: Reward received this tick
            reward_breakdown: Optional list of (name, value) tuples for reward components
        """
        snapshot = self._snapshot_state(state)
        frame = HistoryFrame(
            state_snapshot=snapshot,
            action=action,
            reward=reward,
            tick=state.current_tick,
            reward_breakdown=reward_breakdown
        )
        
        # Store breakdown for display
        self.last_reward_breakdown = reward_breakdown
        
        # If we've gone backward and now record new frame, truncate future history
        if self.history_index < len(self.history) - 1:
            self.history = self.history[:self.history_index + 1]
        
        self.history.append(frame)
        self.history_index = len(self.history) - 1
        
        # Limit history size
        if len(self.history) > self.max_history_size:
            remove_count = len(self.history) - self.max_history_size
            self.history = self.history[remove_count:]
            self.history_index -= remove_count

    def get_history_frame(self, index: int) -> Optional[HistoryFrame]:
        """Get a history frame by index."""
        if 0 <= index < len(self.history):
            return self.history[index]
        return None

    def can_step_backward(self) -> bool:
        """Check if we can step backward in history."""
        return self.history_index > 0

    def can_step_forward_in_history(self) -> bool:
        """Check if we can step forward through existing history."""
        return self.history_index < len(self.history) - 1
    
    def step_backward_in_history(self) -> Optional[HistoryFrame]:
        """Step backward through history and return the frame."""
        if self.can_step_backward():
            self.history_index -= 1
            return self.history[self.history_index]
        return None
    
    def step_forward_in_history(self) -> Optional[HistoryFrame]:
        """Step forward through existing history and return the frame."""
        if self.can_step_forward_in_history():
            self.history_index += 1
            return self.history[self.history_index]
        return None
    
    def is_at_history_end(self) -> bool:
        """Check if we're at the end of recorded history."""
        return self.history_index >= len(self.history) - 1 or len(self.history) == 0

    def clear_history(self):
        """Clear all recorded history (call on episode reset)."""
        self.history.clear()
        self.history_index = -1

    def _snapshot_state(self, state: SimulatorState) -> dict:
        """Create a serializable snapshot of state."""
        # Deep copy essential state elements for replay
        entities_data = []
        for e in state.entities:
            entities_data.append({
                'entity_type': e.entity_type,
                'x': e.x,
                'y': e.y,
                'current_health': e.current_health,
                'attack_delay': e.attack_delay,
                'stunned': e.stunned,
                'frozen': e.frozen,
                'placed_tick': e.placed_tick,
                'scanned_prayer': e.scanned_prayer,
                'target_pillar_index': e.target_pillar_index,
                'had_los': e.had_los,
                'dig_sequence_time': e.dig_sequence_time,
                'dig_location': e.dig_location,
                'has_resurrected': e.has_resurrected,
            })
        
        return {
            'player_x': state.player_x,
            'player_y': state.player_y,
            'player_health': state.player_health,
            'current_preset': state.current_preset,
            'use_blood_barrage': state.use_blood_barrage,
            'player_last_attack_tick': state.player_last_attack_tick,
            'player_attack_speed': state.player_attack_speed,
            'player_attack_range': state.player_attack_range,
            'active_prayer': state.active_prayer,
            'queued_prayer': state.queued_prayer,
            'queued_prayer_tick': state.queued_prayer_tick,
            'pillar_hp': list(state.pillar_hp),
            'pillar_alive': list(state.pillar_alive),
            'pending_pillar_collapses': list(state.pending_pillar_collapses),
            'current_wave': state.current_wave,
            'current_tick': state.current_tick,
            'wave_complete_timer': state.wave_complete_timer,
            'attack_target_index': state.entities.index(state.attack_target) if state.attack_target and state.attack_target in state.entities else -1,
            'entities': entities_data,
        }
    
    def restore_state_from_snapshot(self, state: SimulatorState, snapshot: dict):
        """Restore simulator state from a snapshot."""
        from ..simulator.entity import PlacedEntity
        
        state.player_x = snapshot['player_x']
        state.player_y = snapshot['player_y']
        state.player_health = snapshot['player_health']
        state.current_preset = snapshot['current_preset']
        state.use_blood_barrage = snapshot['use_blood_barrage']
        state.player_last_attack_tick = snapshot['player_last_attack_tick']
        state.player_attack_speed = snapshot['player_attack_speed']
        state.player_attack_range = snapshot['player_attack_range']
        state.active_prayer = snapshot['active_prayer']
        state.queued_prayer = snapshot['queued_prayer']
        state.queued_prayer_tick = snapshot['queued_prayer_tick']
        state.pillar_hp = list(snapshot['pillar_hp'])
        state.pillar_alive = list(snapshot['pillar_alive'])
        state.pending_pillar_collapses = list(snapshot['pending_pillar_collapses'])
        state.current_wave = snapshot['current_wave']
        state.current_tick = snapshot['current_tick']
        state.wave_complete_timer = snapshot['wave_complete_timer']
        
        # Rebuild entities
        state.entities.clear()
        for e_data in snapshot['entities']:
            entity = PlacedEntity(
                entity_type=e_data['entity_type'],
                x=e_data['x'],
                y=e_data['y'],
                placed_tick=e_data['placed_tick']
            )
            entity.current_health = e_data['current_health']
            entity.attack_delay = e_data['attack_delay']
            entity.stunned = e_data['stunned']
            entity.frozen = e_data['frozen']
            entity.scanned_prayer = e_data['scanned_prayer']
            entity.target_pillar_index = e_data['target_pillar_index']
            entity.had_los = e_data['had_los']
            entity.dig_sequence_time = e_data['dig_sequence_time']
            entity.dig_location = e_data['dig_location']
            entity.has_resurrected = e_data['has_resurrected']
            state.entities.append(entity)
        
        # Restore attack target reference
        target_idx = snapshot['attack_target_index']
        if target_idx >= 0 and target_idx < len(state.entities):
            state.attack_target = state.entities[target_idx]
        else:
            state.attack_target = None

    def render(self, state: SimulatorState, action: int = 0, reward: float = 0.0):
        """
        Render the current state.
        
        Args:
            state: Simulator state to render
            action: Last action taken
            reward: Last reward received
        """
        if self.screen is None:
            self.initialize()
        
        self.last_action = action
        self.last_reward = reward
        self.total_reward += reward
        self.step_count += 1
        
        # Clear screen
        self.screen.fill(COLOR_BG)
        
        # Draw grid
        self._draw_grid()
        
        # Draw pillars
        self._draw_pillars(state)
        
        # Draw entities
        self._draw_entities(state)
        
        # Draw player
        self._draw_player(state)
        
        # Draw info panel
        self._draw_info_panel(state)
        
        # Update display
        pygame.display.flip()
        
        # Handle events
        self._handle_events()
        
        # Cap framerate
        self.clock.tick(self.fps)

    def _draw_grid(self):
        """Draw the grid lines."""
        for x in range(GRID_WIDTH + 1):
            pygame.draw.line(
                self.screen, COLOR_GRID,
                (x * self.tile_size, 0),
                (x * self.tile_size, self.grid_height)
            )
        for y in range(GRID_HEIGHT + 1):
            pygame.draw.line(
                self.screen, COLOR_GRID,
                (0, self.grid_height - y * self.tile_size),
                (self.grid_width, self.grid_height - y * self.tile_size)
            )

    def _draw_pillars(self, state: SimulatorState):
        """Draw pillars."""
        for i, pillar in enumerate(PILLARS):
            px, py, pw, ph = pillar
            
            # Convert to screen coordinates (flip y)
            screen_x = px * self.tile_size
            screen_y = self.grid_height - (py + ph) * self.tile_size
            
            color = COLOR_PILLAR if state.pillar_alive[i] else COLOR_PILLAR_DEAD
            
            # Draw pillar rect
            pygame.draw.rect(
                self.screen, color,
                (screen_x, screen_y, pw * self.tile_size, ph * self.tile_size)
            )
            
            # Draw pillar HP
            if state.pillar_alive[i]:
                hp_text = f"{state.pillar_hp[i]}"
                text_surf = self.small_font.render(hp_text, True, (255, 255, 255))
                self.screen.blit(text_surf, (screen_x + 5, screen_y + 5))

    def _draw_entities(self, state: SimulatorState):
        """Draw all entities."""
        from ..simulator.geometry import InfernoLineOfSight
        
        # First draw LOS lines (so they appear behind entities)
        player_center = (
            state.player_x * self.tile_size + self.tile_size // 2,
            self.grid_height - (state.player_y + 1) * self.tile_size + self.tile_size // 2
        )
        
        for entity in state.entities:
            if entity.is_dead():
                continue
            
            # Check LOS
            has_los = InfernoLineOfSight.can_entity_attack_player(
                entity, state.player_x, state.player_y, state.pillar_alive
            )
            
            if has_los:
                size = entity.entity_type.size_in_tiles
                entity_center = (
                    entity.x * self.tile_size + (size * self.tile_size) // 2,
                    self.grid_height - (entity.y + size) * self.tile_size + (size * self.tile_size) // 2
                )
                
                # Draw faint LOS line
                los_color = (100, 50, 50, 128)  # Faint red
                pygame.draw.line(self.screen, los_color, entity_center, player_center, 1)
        
        # Then draw entities
        for entity in state.entities:
            if entity.is_dead():
                continue
            
            self._draw_entity(entity, state)

    def _draw_entity(self, entity: PlacedEntity, state: SimulatorState):
        """Draw a single entity with debug info."""
        # Get color
        color = ENTITY_COLORS.get(entity.entity_type, (150, 150, 150))
        
        # Highlight if it's the attack target
        is_target = state.attack_target == entity
        if is_target:
            color = tuple(min(255, c + 50) for c in color)
        
        # Dim if frozen/stunned
        if entity.frozen > 0 or entity.stunned > 0:
            color = tuple(c // 2 for c in color)
        
        # Convert to screen coordinates
        size = entity.entity_type.size_in_tiles
        screen_x = entity.x * self.tile_size
        screen_y = self.grid_height - (entity.y + size) * self.tile_size
        
        # Draw entity rect
        pygame.draw.rect(
            self.screen, color,
            (screen_x, screen_y, size * self.tile_size, size * self.tile_size)
        )
        
        # Draw border for target (yellow)
        if is_target:
            pygame.draw.rect(
                self.screen, COLOR_PLAYER_TARGET,
                (screen_x, screen_y, size * self.tile_size, size * self.tile_size),
                3
            )
        
        # Draw attack ready indicator (red border if ready to attack)
        if entity.attack_delay <= 0 and entity.stunned <= 0:
            pygame.draw.rect(
                self.screen, (255, 0, 0),
                (screen_x + 2, screen_y + 2, size * self.tile_size - 4, size * self.tile_size - 4),
                2
            )
        
        # Draw HP bar
        hp_ratio = entity.current_health / entity.entity_type.max_health
        bar_width = size * self.tile_size - 4
        bar_height = 4
        
        # Background
        pygame.draw.rect(
            self.screen, (50, 50, 50),
            (screen_x + 2, screen_y - 8, bar_width, bar_height)
        )
        
        # HP fill
        hp_color = (0, 255, 0) if hp_ratio > 0.5 else (255, 255, 0) if hp_ratio > 0.25 else (255, 0, 0)
        pygame.draw.rect(
            self.screen, hp_color,
            (screen_x + 2, screen_y - 8, int(bar_width * hp_ratio), bar_height)
        )
        
        # Draw entity type name (show part after "Jal-")
        name = entity.entity_type.name
        if name.startswith("Jal-"):
            abbrev = name[4:]  # Everything after "Jal-"
        else:
            abbrev = name
        text_surf = self.small_font.render(abbrev, True, (255, 255, 255))
        self.screen.blit(text_surf, (screen_x + 2, screen_y + 2))
        
        # Draw attack delay timer
        if entity.attack_delay > 0:
            atk_text = self.small_font.render(str(entity.attack_delay), True, (255, 200, 0))
            self.screen.blit(atk_text, (screen_x + size * self.tile_size - 12, screen_y + 2))
        
        # Draw blob scanned prayer indicator
        if entity.scanned_prayer:
            scan_color = (100, 150, 255) if entity.scanned_prayer == "MAGIC" else (100, 255, 100)
            scan_text = "M" if entity.scanned_prayer == "MAGIC" else "R"
            scan_surf = self.small_font.render(scan_text, True, scan_color)
            self.screen.blit(scan_surf, (screen_x + size * self.tile_size - 12, screen_y + size * self.tile_size - 14))
        
        # Draw frozen/stunned indicator
        if entity.frozen > 0:
            frz_surf = self.small_font.render("F", True, (100, 200, 255))
            self.screen.blit(frz_surf, (screen_x + 2, screen_y + size * self.tile_size - 14))
        elif entity.stunned > 0:
            stn_surf = self.small_font.render("S", True, (255, 255, 100))
            self.screen.blit(stn_surf, (screen_x + 2, screen_y + size * self.tile_size - 14))

    def _draw_player(self, state: SimulatorState):
        """Draw the player."""
        screen_x = state.player_x * self.tile_size
        screen_y = self.grid_height - (state.player_y + 1) * self.tile_size
        
        # Draw player circle
        center = (screen_x + self.tile_size // 2, screen_y + self.tile_size // 2)
        pygame.draw.circle(self.screen, COLOR_PLAYER, center, self.tile_size // 2 - 2)
        
        # Draw HP bar
        hp_ratio = state.player_health / 99
        bar_width = self.tile_size - 4
        bar_height = 4
        
        hp_color = (0, 255, 0) if hp_ratio > 0.5 else (255, 255, 0) if hp_ratio > 0.25 else (255, 0, 0)
        pygame.draw.rect(
            self.screen, hp_color,
            (screen_x + 2, screen_y - 6, int(bar_width * hp_ratio), bar_height)
        )
        
        # Draw attack target line if exists
        if state.attack_target and not state.attack_target.is_dead():
            target = state.attack_target
            target_size = target.entity_type.size_in_tiles
            target_center = (
                (target.x + target_size // 2) * self.tile_size + self.tile_size // 2,
                self.grid_height - (target.y + target_size // 2 + 1) * self.tile_size + self.tile_size // 2
            )
            pygame.draw.line(self.screen, COLOR_PLAYER_TARGET, center, target_center, 2)

    def _draw_info_panel(self, state: SimulatorState):
        """Draw the info panel with detailed debug information in two columns."""
        left_x = self.grid_width + 10
        right_x = self.grid_width + 295  # Second column
        line_height = 18
        small_line = 14
        
        # === LEFT COLUMN: Game State ===
        y = 10
        
        def draw_text_left(text: str, color=(255, 255, 255), small=False):
            nonlocal y
            font = self.small_font if small else self.font
            surf = font.render(text, True, color)
            self.screen.blit(surf, (left_x, y))
            y += small_line if small else line_height
        
        def draw_section_left(title: str):
            nonlocal y
            y += 5
            draw_text_left(f"=== {title} ===", (100, 200, 255))
        
        # Wave info
        draw_section_left("WAVE")
        draw_text_left(f"Wave: {state.current_wave}")
        draw_text_left(f"Tick: {state.current_tick}")
        if state.wave_complete_timer > 0:
            draw_text_left(f"Next wave in: {state.wave_complete_timer}", (255, 255, 0))
        
        # Player info
        draw_section_left("PLAYER")
        hp_color = (0, 255, 0) if state.player_health > 50 else (255, 255, 0) if state.player_health > 25 else (255, 0, 0)
        draw_text_left(f"HP: {state.player_health}/99", hp_color)
        draw_text_left(f"Pos: ({state.player_x}, {state.player_y})")
        draw_text_left(f"Weapon: {state.current_preset.value}")
        
        # Attack cooldown
        ticks_since_attack = state.current_tick - state.player_last_attack_tick
        can_attack = ticks_since_attack >= state.player_attack_speed
        cd_color = (0, 255, 0) if can_attack else (255, 100, 100)
        cd_text = "READY" if can_attack else f"{state.player_attack_speed - ticks_since_attack} ticks"
        draw_text_left(f"Atk CD: {cd_text}", cd_color)
        
        # Attack target
        if state.attack_target and not state.attack_target.is_dead():
            t = state.attack_target
            draw_text_left(f"Target: {t.entity_type.name}", (255, 200, 100))
        else:
            draw_text_left("Target: None", (150, 150, 150))
        
        # Prayer
        draw_section_left("PRAYER")
        prayer = state.active_prayer or "None"
        prayer_color = {
            "PROTECT_FROM_MAGIC": (100, 150, 255),
            "PROTECT_FROM_MISSILES": (100, 255, 100),
            "PROTECT_FROM_MELEE": (255, 150, 100)
        }.get(prayer, (150, 150, 150))
        draw_text_left(f"Active: {prayer}", prayer_color)
        if state.queued_prayer:
            draw_text_left(f"Queued: {state.queued_prayer}", (150, 150, 150), small=True)
        
        # Pillars
        draw_section_left("PILLARS")
        for i in range(3):
            name = ["NW", "NE", "S"][i]
            alive = state.pillar_alive[i]
            hp = state.pillar_hp[i]
            if alive:
                hp_ratio = hp / 255
                color = (0, 255, 0) if hp_ratio > 0.5 else (255, 255, 0) if hp_ratio > 0.25 else (255, 0, 0)
                draw_text_left(f"{name}: {hp}/255", color, small=True)
            else:
                draw_text_left(f"{name}: DESTROYED", (100, 100, 100), small=True)
        
        # Entity details
        draw_section_left("ENTITIES")
        alive_entities = [e for e in state.entities if not e.is_dead()]
        draw_text_left(f"Alive: {len(alive_entities)}")
        
        for entity in alive_entities[:5]:  # Show max 5 entities
            self._draw_entity_debug_compact(entity, state, left_x, y, draw_text_left)
        
        if len(alive_entities) > 5:
            draw_text_left(f"  +{len(alive_entities) - 5} more", (150, 150, 150), small=True)
        
        # === RIGHT COLUMN: Action & Rewards ===
        y = 10
        
        def draw_text_right(text: str, color=(255, 255, 255), small=False):
            nonlocal y
            font = self.small_font if small else self.font
            surf = font.render(text, True, color)
            self.screen.blit(surf, (right_x, y))
            y += small_line if small else line_height
        
        def draw_section_right(title: str):
            nonlocal y
            y += 5
            draw_text_right(f"=== {title} ===", (100, 200, 255))
        
        # Action & Reward
        draw_section_right("ACTION")
        action_name = self._get_action_name(self.last_action)
        draw_text_right(f"Action: {action_name}")
        reward_color = (0, 255, 0) if self.last_reward > 0 else (255, 0, 0) if self.last_reward < 0 else (200, 200, 200)
        draw_text_right(f"Reward: {self.last_reward:+.1f}", reward_color)
        draw_text_right(f"Total: {self.total_reward:+.1f}")
        draw_text_right(f"Steps: {self.step_count}")
        
        # Reward breakdown (show top components)
        draw_section_right("BREAKDOWN")
        if self.last_reward_breakdown:
            # Sort by absolute value, show top 8
            sorted_components = sorted(self.last_reward_breakdown, key=lambda x: abs(x[1]), reverse=True)
            for name, value in sorted_components[:8]:
                comp_color = (100, 255, 100) if value > 0 else (255, 100, 100)
                # Truncate long names
                display_name = name[:20] if len(name) > 20 else name
                draw_text_right(f"{value:+.0f} {display_name}", comp_color, small=True)
            if len(sorted_components) > 8:
                draw_text_right(f"  +{len(sorted_components) - 8} more", (120, 120, 120), small=True)
        else:
            draw_text_right("(none)", (120, 120, 120), small=True)
        
        # Mode indicator
        draw_section_right("MODE")
        if self.review_info:
            ri = self.review_info
            draw_text_right(f"REVIEW: Seed {ri['seed']} W{ri['death_wave']}", COLOR_MODE_MANUAL)
            draw_text_right(f"Death {ri['current']}/{ri['total']}", (180, 180, 180), small=True)
            history_info = f"Frame: {self.history_index + 1}/{len(self.history)}"
            draw_text_right(history_info, (180, 180, 180), small=True)
        elif self.manual_mode:
            draw_text_right("MANUAL", COLOR_MODE_MANUAL)
            history_info = f"Frame: {self.history_index + 1}/{len(self.history)}"
            draw_text_right(history_info, (180, 180, 180), small=True)
        else:
            draw_text_right("AUTO", COLOR_MODE_AUTO)

        # Controls at bottom (spans both columns)
        y = self.window_height - 50
        draw_text_right("Controls:", (100, 100, 100), small=True)
        if self.review_info:
            draw_text_right("L/R=Step HOME/END=Jump N=Next ESC=Quit", (100, 100, 100), small=True)
        elif self.manual_mode:
            draw_text_right("Left/Right=Step J=Auto ESC=Quit", (100, 100, 100), small=True)
        else:
            draw_text_right("SPACE=Pause J=Manual ESC=Quit", (100, 100, 100), small=True)
    
    def _draw_entity_debug_compact(self, entity, state, x, y_start, draw_fn):
        """Draw compact debug info for a single entity."""
        from ..simulator.geometry import InfernoLineOfSight
        
        name = entity.entity_type.name
        if name.startswith("Jal-"):
            name = name[4:]
        name = name[:8]  # Truncate
        
        hp = entity.current_health
        max_hp = entity.entity_type.max_health
        
        # Check LOS
        has_los = InfernoLineOfSight.can_entity_attack_player(
            entity, state.player_x, state.player_y, state.pillar_alive
        )
        los_str = "LOS" if has_los else ""
        los_color = (0, 255, 0) if has_los else (255, 100, 100)
        
        # Attack status
        atk_str = "RDY" if entity.attack_delay <= 0 else str(entity.attack_delay)
        
        draw_fn(f"  {name} {hp}/{max_hp} {los_str} ATK:{atk_str}", los_color, small=True)

    def _get_action_name(self, action: int) -> str:
        """Get human-readable action name."""
        if action == 0:
            return "STAY"
        if 1 <= action <= 32:
            return f"MOVE_{action}"
        if InfernoAction.is_attack(action):
            return f"ATK_T{InfernoAction.get_target_index(action) + 1}"
        if action == InfernoAction.NO_ACTION_IDX:
            return "NO_ACTION"
        if action == InfernoAction.SWITCH_BOFA:
            return "SW_BOFA"
        if action == InfernoAction.SWITCH_BLOWPIPE:
            return "SW_BPIPE"
        if action == InfernoAction.SWITCH_ICE_BARRAGE:
            return "SW_ICE_B"
        if action == InfernoAction.SWITCH_BLOOD_BARRAGE:
            return "SW_BLOOD_B"
        return f"ACTION_{action}"

    def _handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                raise SystemExit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.close()
                    raise SystemExit()
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_j:
                    self.toggle_manual_mode()
                elif event.key == pygame.K_RIGHT:
                    if self.manual_mode:
                        self.step_forward_request = True
                elif event.key == pygame.K_LEFT:
                    if self.manual_mode:
                        self.step_backward_request = True
                elif event.key == pygame.K_HOME:
                    self.jump_start_request = True
                elif event.key == pygame.K_END:
                    self.jump_end_request = True
                elif event.key in (pygame.K_n, pygame.K_RETURN):
                    self.next_seed_request = True

    def is_paused(self) -> bool:
        """Check if visualization is paused."""
        return self.paused

    def wait_for_unpause(self):
        """Wait until unpaused."""
        while self.paused:
            self._handle_events()
            self.clock.tick(30)
