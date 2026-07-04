"""
Interactive Debug Sandbox for Inferno Simulator.

Allows manual entity placement, tick stepping, and state inspection.

Controls:
- SPACE: Step one tick
- R: Reset simulation
- P: Print detailed state
- 1-9: Place entity at mouse position
- Delete/Backspace: Remove entity under mouse
- Click: Select entity / Set player position (with Shift)
- ESC: Quit

Entity placement keys:
- 1: Nibbler
- 2: Bat
- 3: Blob
- 4: Melee
- 5: Ranger
- 6: Mager
"""

import sys
import argparse
import pygame
import numpy as np
from typing import Optional, List, Tuple

# Add parent to path for imports
sys.path.insert(0, str(__file__).replace('\\', '/').rsplit('/', 3)[0])

from inferno_rl.simulator.simulator import InfernoSimulator
from inferno_rl.simulator.state import SimulatorState, spawn_wave_entities
from inferno_rl.simulator.equipment import GearPreset
from inferno_rl.simulator.entity import PlacedEntity, EntityTypes, InfernoEntityType
from inferno_rl.simulator.exact_targeting import (
    get_exact_target_slot_index,
    get_exact_target_slots,
    select_center_nibbler,
)
from inferno_rl.simulator.geometry import SimulatorGeometry, InfernoLineOfSight, GRID_WIDTH, GRID_HEIGHT, PILLARS
from inferno_rl.training.actions import InfernoAction, get_action_mask, NUM_ACTIONS
from inferno_rl.training.rewards import InfernoReward


# Colors
COLOR_BG = (30, 30, 30)
COLOR_GRID = (50, 50, 50)
COLOR_PLAYER = (0, 255, 0)
COLOR_PILLAR = (139, 90, 43)
COLOR_PILLAR_DEAD = (60, 40, 20)
COLOR_SELECTED = (255, 255, 0)
COLOR_TEXT = (255, 255, 255)
COLOR_HEADER = (100, 200, 255)

ENTITY_COLORS = {
    EntityTypes.NIBBLER: (255, 100, 100),
    EntityTypes.BAT: (200, 150, 100),
    EntityTypes.BLOB: (100, 255, 100),
    EntityTypes.MELEE: (255, 0, 255),
    EntityTypes.RANGER: (100, 100, 255),
    EntityTypes.MAGER: (255, 0, 100),
}

ENTITY_KEYS = {
    pygame.K_1: EntityTypes.NIBBLER,
    pygame.K_2: EntityTypes.BAT,
    pygame.K_3: EntityTypes.BLOB,
    pygame.K_4: EntityTypes.MELEE,
    pygame.K_5: EntityTypes.RANGER,
    pygame.K_6: EntityTypes.MAGER,
}


class DebugSandbox:
    """Interactive debug sandbox for the Inferno simulator."""
    
    def __init__(self, tile_size: int = 22):
        self.tile_size = tile_size
        self.info_panel_width = 450
        self.grid_width = GRID_WIDTH * tile_size
        self.grid_height = GRID_HEIGHT * tile_size
        self.window_width = self.grid_width + self.info_panel_width
        self.window_height = self.grid_height
        
        # Simulator
        self.simulator = InfernoSimulator(start_wave=35, max_wave=69)
        self.simulator.initial_barrage_enabled = False  # Disable auto-barrage
        self.reward_calculator = InfernoReward()
        
        # State
        self.selected_entity: Optional[PlacedEntity] = None
        self.last_action: int = 0
        self.last_reward: float = 0.0
        self.last_step_result = None
        self.action_history: List[Tuple[int, int, float]] = []  # (tick, action, reward)
        self.paused = True  # Start paused
        
        # Pygame
        pygame.init()
        pygame.display.set_caption("Inferno Debug Sandbox")
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 14)
        self.small_font = pygame.font.SysFont("monospace", 11)
        self.header_font = pygame.font.SysFont("monospace", 14, bold=True)
        
        # Initialize
        self._init_sandbox()
    
    def _init_sandbox(self):
        """Initialize sandbox state."""
        self.simulator.state.clear_entities()
        self.simulator.state.reset_player()
        self.simulator.state.reset_pillars()
        self.simulator.state.current_tick = 1
        self.simulator.state.current_wave = 35
        self.simulator.wave_start_tick = 1
        self.action_history.clear()
        self.selected_entity = None
        print("Sandbox initialized. Press SPACE to step, 1-6 to place entities.")
    
    def run(self):
        """Main loop."""
        running = True
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    running = self._handle_key(event)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self._handle_mouse(event)
            
            # Render
            self._render()
            self.clock.tick(30)
        
        pygame.quit()
    
    def _handle_key(self, event) -> bool:
        """Handle keyboard input. Returns False to quit."""
        key = event.key
        mods = pygame.key.get_mods()
        
        if key == pygame.K_ESCAPE:
            return False
        
        elif key == pygame.K_SPACE:
            self._step_tick()
        
        elif key == pygame.K_r:
            self._init_sandbox()
        
        elif key == pygame.K_p:
            self._print_detailed_state()
        
        elif key == pygame.K_w:
            # Spawn wave
            self._spawn_wave()
        
        elif key == pygame.K_a:
            # Attack the selected entity's exact target slot
            self._execute_selected_target_action()
        
        elif key == pygame.K_n:
            # Attack the first visible nibbler slot
            self._execute_first_nibbler_action()
        
        elif key == pygame.K_b:
            # Attack the center nibbler slot
            self._execute_center_nibbler_action()
        
        elif key in ENTITY_KEYS:
            # Place entity at mouse position
            mx, my = pygame.mouse.get_pos()
            gx, gy = self._screen_to_grid(mx, my)
            if gx is not None:
                self._place_entity(ENTITY_KEYS[key], gx, gy)
        
        elif key in (pygame.K_DELETE, pygame.K_BACKSPACE):
            # Remove entity under mouse
            mx, my = pygame.mouse.get_pos()
            gx, gy = self._screen_to_grid(mx, my)
            if gx is not None:
                self._remove_entity_at(gx, gy)
        
        elif key == pygame.K_k:
            # Kill selected entity
            if self.selected_entity and not self.selected_entity.is_dead():
                self.selected_entity.take_damage(self.selected_entity.current_health)
                print(f"Killed {self.selected_entity.entity_type.name}")
        
        elif key == pygame.K_d:
            # Damage selected entity
            if self.selected_entity and not self.selected_entity.is_dead():
                self.selected_entity.take_damage(10)
                print(f"Damaged {self.selected_entity.entity_type.name} (HP: {self.selected_entity.current_health})")
        
        elif key == pygame.K_t:
            # Set selected entity as attack target
            if self.selected_entity and not self.selected_entity.is_dead():
                self.simulator.state.attack_target = self.selected_entity
                print(f"Attack target set to {self.selected_entity.entity_type.name}")
            else:
                self.simulator.state.attack_target = None
                print("Attack target cleared")
        
        elif key == pygame.K_h:
            # Print help
            self._print_help()
        
        return True
    
    def _handle_mouse(self, event):
        """Handle mouse input."""
        mx, my = event.pos
        gx, gy = self._screen_to_grid(mx, my)
        
        if gx is None:
            return
        
        mods = pygame.key.get_mods()
        
        if mods & pygame.KMOD_SHIFT:
            # Shift+click: Move player
            if SimulatorGeometry.is_valid_tile(gx, gy, self.simulator.state.pillar_alive):
                self.simulator.state.player_x = gx
                self.simulator.state.player_y = gy
                print(f"Player moved to ({gx}, {gy})")
        else:
            # Click: Select entity
            self.selected_entity = self._get_entity_at(gx, gy)
            if self.selected_entity:
                print(f"Selected: {self.selected_entity.entity_type.name} at ({self.selected_entity.x}, {self.selected_entity.y})")
    
    def _screen_to_grid(self, mx: int, my: int) -> Tuple[Optional[int], Optional[int]]:
        """Convert screen coordinates to grid coordinates."""
        if mx >= self.grid_width:
            return None, None
        
        gx = mx // self.tile_size
        gy = (self.grid_height - my) // self.tile_size
        
        if 0 <= gx < GRID_WIDTH and 0 <= gy < GRID_HEIGHT:
            return gx, gy
        return None, None
    
    def _get_entity_at(self, gx: int, gy: int) -> Optional[PlacedEntity]:
        """Get entity at grid position."""
        for entity in self.simulator.state.entities:
            if entity.is_dead():
                continue
            size = entity.entity_type.size_in_tiles
            if entity.x <= gx < entity.x + size and entity.y <= gy < entity.y + size:
                return entity
        return None
    
    def _place_entity(self, entity_type: InfernoEntityType, x: int, y: int):
        """Place an entity at the given position."""
        entity = PlacedEntity(
            entity_type=entity_type,
            x=x,
            y=y,
            placed_tick=self.simulator.state.current_tick
        )
        
        # For nibblers, assign a pillar target
        if entity_type == EntityTypes.NIBBLER:
            alive = self.simulator.state.get_alive_pillar_indices()
            if alive:
                import random
                entity.target_pillar_index = random.choice(alive)
        
        self.simulator.state.add_entity(entity)
        print(f"Placed {entity_type.name} at ({x}, {y})")
    
    def _remove_entity_at(self, gx: int, gy: int):
        """Remove entity at grid position."""
        entity = self._get_entity_at(gx, gy)
        if entity:
            entity.take_damage(entity.current_health + 100)
            if self.selected_entity == entity:
                self.selected_entity = None
            print(f"Removed {entity.entity_type.name}")
    
    def _spawn_wave(self):
        """Spawn the current wave's entities."""
        wave = self.simulator.state.current_wave
        spawn_wave_entities(self.simulator.state, wave)
        print(f"Spawned wave {wave}")
    
    def _step_tick(self, action: int = 0):
        """Step one tick with the given action."""
        result = self.simulator.step(action)
        reward = self.reward_calculator.calculate(result)
        
        self.last_action = action
        self.last_reward = reward
        self.last_step_result = result
        self.action_history.append((self.simulator.state.current_tick - 1, action, reward))
        
        # Print step info
        print(f"\n=== Tick {self.simulator.state.current_tick - 1} → {self.simulator.state.current_tick} ===")
        print(f"Action: {self._get_action_name(action)}")
        print(f"Reward: {reward:+.1f}")
        
        if result.player_died:
            print("*** PLAYER DIED ***")
        if result.wave_timeout:
            print("*** WAVE TIMEOUT ***")
        if result.wave_completed:
            print(f"*** WAVE {result.wave_number} COMPLETE ***")
    
    def _execute_action(self, action: int):
        """Execute a specific action."""
        mask = get_action_mask(self.simulator.state)
        if not mask[action]:
            print(f"Action {self._get_action_name(action)} is not valid!")
            return
        self._step_tick(action)

    def _execute_targeted_attack(self, target: Optional[PlacedEntity], label: str):
        if target is None or target.is_dead():
            print(f"No {label} target available.")
            return
        slot_index = get_exact_target_slot_index(self.simulator.state, target)
        if slot_index is None:
            print(f"{label.capitalize()} target is not in the exact target slot list.")
            return
        self._execute_action(InfernoAction.action_for_target_index(slot_index))

    def _execute_selected_target_action(self):
        """Attack the currently selected entity via its exact target slot."""
        self._execute_targeted_attack(self.selected_entity, "selected")

    def _execute_first_nibbler_action(self):
        """Attack the first live nibbler in exact-target order."""
        target = next(
            (entity for entity in get_exact_target_slots(self.simulator.state)
             if entity.entity_type == EntityTypes.NIBBLER),
            None,
        )
        self._execute_targeted_attack(target, "nibbler")

    def _execute_center_nibbler_action(self):
        """Attack the center nibbler used for barrage-style debugging."""
        self._execute_targeted_attack(select_center_nibbler(self.simulator.state.entities), "center nibbler")
    
    def _print_detailed_state(self):
        """Print detailed simulation state."""
        state = self.simulator.state
        
        print("\n" + "=" * 60)
        print("DETAILED STATE")
        print("=" * 60)
        
        print(f"\n--- SIMULATION ---")
        print(f"Wave: {state.current_wave}")
        print(f"Tick: {state.current_tick}")
        print(f"Ticks in wave: {self.simulator.get_ticks_in_wave()}")
        
        print(f"\n--- PLAYER ---")
        print(f"Position: ({state.player_x}, {state.player_y})")
        print(f"Health: {state.player_health}/99")
        print(f"Weapon: {state.current_preset.value}")
        print(f"Attack cooldown: {state.get_player_attack_cooldown()}t remaining")
        print(f"Active prayer: {state.active_prayer or 'None'}")
        print(f"Queued prayer: {state.queued_prayer or 'None'}")
        print(f"Attack target: {state.attack_target.entity_type.name if state.attack_target else 'None'}")
        
        print(f"\n--- PILLARS ---")
        for i in range(3):
            name = ["NW", "NE", "S"][i]
            alive = state.pillar_alive[i]
            hp = state.pillar_hp[i]
            collapse = state.pending_pillar_collapses[i]
            status = f"HP: {hp}/255" if alive else "DESTROYED"
            if collapse:
                status += f" (collapse in {collapse} ticks)"
            print(f"  {name}: {status}")
        
        print(f"\n--- ENTITIES ({len([e for e in state.entities if not e.is_dead()])} alive) ---")
        for entity in state.entities:
            if entity.is_dead():
                continue
            
            has_los = InfernoLineOfSight.can_entity_attack_player(
                entity, state.player_x, state.player_y, state.pillar_alive
            )
            
            flags = []
            if entity.stunned > 0:
                flags.append(f"STUN:{entity.stunned}")
            if entity.frozen > 0:
                flags.append(f"FROZEN:{entity.frozen}")
            if entity.attack_delay > 0:
                flags.append(f"ATK_CD:{entity.attack_delay}")
            else:
                flags.append("ATK_RDY")
            if entity.scanned_prayer:
                flags.append(f"SCAN:{entity.scanned_prayer}")
            if entity.target_pillar_index >= 0:
                flags.append(f"PIL:{entity.target_pillar_index}")
            
            print(f"  {entity.entity_type.name}")
            print(f"    Pos: ({entity.x}, {entity.y}), HP: {entity.current_health}/{entity.entity_type.max_health}")
            print(f"    LOS: {'YES' if has_los else 'NO'}, {' '.join(flags)}")
        
        print(f"\n--- ACTION MASK ---")
        mask = get_action_mask(state)
        valid_attacks = [
            f"T{slot_index + 1}"
            for slot_index in range(14)
            if mask[InfernoAction.ATTACK_TARGET_1 + slot_index]
        ]
        print(f"  Valid attacks: {', '.join(valid_attacks) if valid_attacks else 'None'}")
        
        print("=" * 60)
    
    def _print_help(self):
        """Print help."""
        print("""
=== DEBUG SANDBOX CONTROLS ===

TICK CONTROL:
  SPACE     Step one tick (with STAY action)
  
ENTITY PLACEMENT:
  1         Place Nibbler at mouse
  2         Place Bat at mouse
  3         Place Blob at mouse
  4         Place Melee at mouse
  5         Place Ranger at mouse
  6         Place Mager at mouse
  Del/Back  Remove entity at mouse
  
ACTIONS:
  A         Attack selected entity slot
  N         Attack first live nibbler slot
  B         Attack center nibbler slot
  W         Spawn current wave
  
SELECTION:
  Click     Select entity
  Shift+Click  Move player to position
  T         Set selected as attack target
  K         Kill selected entity
  D         Damage selected entity (-10 HP)
  
INFO:
  P         Print detailed state
  H         Show this help
  R         Reset sandbox
  ESC       Quit
""")
    
    def _render(self):
        """Render the sandbox."""
        self.screen.fill(COLOR_BG)
        
        # Draw grid
        self._draw_grid()
        
        # Draw pillars
        self._draw_pillars()
        
        # Draw entities
        self._draw_entities()
        
        # Draw player
        self._draw_player()
        
        # Draw info panel
        self._draw_info_panel()
        
        pygame.display.flip()
    
    def _draw_grid(self):
        """Draw grid lines."""
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
    
    def _draw_pillars(self):
        """Draw pillars."""
        state = self.simulator.state
        for i, pillar in enumerate(PILLARS):
            px, py, pw, ph = pillar
            screen_x = px * self.tile_size
            screen_y = self.grid_height - (py + ph) * self.tile_size
            
            color = COLOR_PILLAR if state.pillar_alive[i] else COLOR_PILLAR_DEAD
            pygame.draw.rect(
                self.screen, color,
                (screen_x, screen_y, pw * self.tile_size, ph * self.tile_size)
            )
            
            # HP text
            if state.pillar_alive[i]:
                hp_text = str(state.pillar_hp[i])
                surf = self.small_font.render(hp_text, True, COLOR_TEXT)
                self.screen.blit(surf, (screen_x + 2, screen_y + 2))
    
    def _draw_entities(self):
        """Draw all entities."""
        state = self.simulator.state
        
        for entity in state.entities:
            if entity.is_dead():
                continue
            
            self._draw_entity(entity)
    
    def _draw_entity(self, entity: PlacedEntity):
        """Draw a single entity."""
        state = self.simulator.state
        color = ENTITY_COLORS.get(entity.entity_type, (150, 150, 150))
        
        # Highlight selected
        is_selected = entity == self.selected_entity
        
        # Dim if frozen/stunned
        if entity.frozen > 0 or entity.stunned > 0:
            color = tuple(c // 2 for c in color)
        
        size = entity.entity_type.size_in_tiles
        screen_x = entity.x * self.tile_size
        screen_y = self.grid_height - (entity.y + size) * self.tile_size
        
        # Draw rect
        pygame.draw.rect(
            self.screen, color,
            (screen_x, screen_y, size * self.tile_size, size * self.tile_size)
        )
        
        # Selection border
        if is_selected:
            pygame.draw.rect(
                self.screen, COLOR_SELECTED,
                (screen_x, screen_y, size * self.tile_size, size * self.tile_size),
                3
            )
        
        # Attack ready indicator
        if entity.attack_delay <= 0 and entity.stunned <= 0:
            pygame.draw.rect(
                self.screen, (255, 0, 0),
                (screen_x + 2, screen_y + 2, size * self.tile_size - 4, size * self.tile_size - 4),
                2
            )
        
        # HP bar
        hp_ratio = entity.current_health / entity.entity_type.max_health
        bar_width = size * self.tile_size - 4
        pygame.draw.rect(self.screen, (50, 50, 50), (screen_x + 2, screen_y - 6, bar_width, 4))
        hp_color = (0, 255, 0) if hp_ratio > 0.5 else (255, 255, 0) if hp_ratio > 0.25 else (255, 0, 0)
        pygame.draw.rect(self.screen, hp_color, (screen_x + 2, screen_y - 6, int(bar_width * hp_ratio), 4))
        
        # Name
        name = entity.entity_type.name
        if name.startswith("Jal-"):
            name = name[4:]
        surf = self.small_font.render(name, True, COLOR_TEXT)
        self.screen.blit(surf, (screen_x + 2, screen_y + 2))
        
        # Attack delay
        if entity.attack_delay > 0:
            atk_surf = self.small_font.render(str(entity.attack_delay), True, (255, 200, 0))
            self.screen.blit(atk_surf, (screen_x + size * self.tile_size - 12, screen_y + 2))
    
    def _draw_player(self):
        """Draw the player."""
        state = self.simulator.state
        screen_x = state.player_x * self.tile_size
        screen_y = self.grid_height - (state.player_y + 1) * self.tile_size
        
        center = (screen_x + self.tile_size // 2, screen_y + self.tile_size // 2)
        pygame.draw.circle(self.screen, COLOR_PLAYER, center, self.tile_size // 2 - 2)
        
        # Attack target line
        if state.attack_target and not state.attack_target.is_dead():
            t = state.attack_target
            ts = t.entity_type.size_in_tiles
            target_center = (
                t.x * self.tile_size + (ts * self.tile_size) // 2,
                self.grid_height - (t.y + ts) * self.tile_size + (ts * self.tile_size) // 2
            )
            pygame.draw.line(self.screen, COLOR_SELECTED, center, target_center, 2)
    
    def _draw_info_panel(self):
        """Draw the info panel."""
        x = self.grid_width + 10
        y = 10
        line = 16
        small_line = 13
        
        def text(t, color=COLOR_TEXT, header=False, small=False):
            nonlocal y
            font = self.header_font if header else (self.small_font if small else self.font)
            surf = font.render(t, True, color)
            self.screen.blit(surf, (x, y))
            y += line if not small else small_line
        
        def section(title):
            nonlocal y
            y += 5
            text(f"=== {title} ===", COLOR_HEADER, header=True)
        
        state = self.simulator.state
        
        # Simulation
        section("SIMULATION")
        text(f"Wave: {state.current_wave}  Tick: {state.current_tick}")
        text(f"Ticks in wave: {self.simulator.get_ticks_in_wave()}")
        
        # Player
        section("PLAYER")
        hp_color = (0, 255, 0) if state.player_health > 50 else (255, 255, 0) if state.player_health > 25 else (255, 0, 0)
        text(f"HP: {state.player_health}/99", hp_color)
        text(f"Pos: ({state.player_x}, {state.player_y})")
        text(f"Weapon: {state.current_preset.value}")
        
        cd = state.current_tick - state.player_last_attack_tick
        ready = cd >= state.player_attack_speed
        cd_color = (0, 255, 0) if ready else (255, 100, 100)
        text(f"Atk CD: {'READY' if ready else f'{state.player_attack_speed - cd} ticks'}", cd_color)
        
        if state.attack_target:
            text(f"Target: {state.attack_target.entity_type.name}", (255, 200, 100))
        
        # Prayer
        section("PRAYER")
        prayer = state.active_prayer or "None"
        text(f"Active: {prayer}")
        if state.queued_prayer:
            text(f"Queued: {state.queued_prayer}", small=True)
        
        # Last action
        section("LAST ACTION")
        text(f"Action: {self._get_action_name(self.last_action)}")
        reward_color = (0, 255, 0) if self.last_reward > 0 else (255, 0, 0) if self.last_reward < 0 else COLOR_TEXT
        text(f"Reward: {self.last_reward:+.1f}", reward_color)
        
        # Entities
        section("ENTITIES")
        alive = [e for e in state.entities if not e.is_dead()]
        text(f"Alive: {len(alive)}")
        
        for entity in alive[:5]:
            has_los = InfernoLineOfSight.can_entity_attack_player(
                entity, state.player_x, state.player_y, state.pillar_alive
            )
            los_str = "LOS" if has_los else "---"
            atk = "RDY" if entity.attack_delay <= 0 else str(entity.attack_delay)
            name = entity.entity_type.name
            if name.startswith("Jal-"):
                name = name[4:]
            text(f"  {name} HP:{entity.current_health} {los_str} ATK:{atk}", small=True)
        
        # Controls hint
        y = self.window_height - 80
        text("Controls:", (100, 100, 100), small=True)
        text("  SPACE=Step  1-6=Place  Click=Select", (100, 100, 100), small=True)
        text("  Shift+Click=Move  P=State  H=Help", (100, 100, 100), small=True)
    
    def _get_action_name(self, action: int) -> str:
        """Get action name."""
        if action == InfernoAction.STAY:
            return "STAY"
        if InfernoAction.is_attack(action):
            return f"ATK_T{InfernoAction.get_target_index(action) + 1}"
        names = {
            InfernoAction.NO_ACTION_IDX: "NO_ACTION",
            InfernoAction.SWITCH_BOFA: "SW_BOFA",
            InfernoAction.SWITCH_BLOWPIPE: "SW_BPIPE",
            InfernoAction.SWITCH_ICE_BARRAGE: "SW_ICE",
            InfernoAction.SWITCH_BLOOD_BARRAGE: "SW_BLOOD",
        }
        if action in names:
            return names[action]
        if 1 <= action <= 32:
            return f"MOVE_{action}"
        return f"ACTION_{action}"


def main():
    parser = argparse.ArgumentParser(description="Inferno Debug Sandbox")
    parser.add_argument("--wave", type=int, default=35, help="Starting wave")
    args = parser.parse_args()
    
    sandbox = DebugSandbox()
    sandbox.simulator.state.current_wave = args.wave
    
    print("\n=== INFERNO DEBUG SANDBOX ===")
    print("Press H for help, SPACE to step ticks")
    print("Click to select, Shift+Click to move player")
    print("Keys 1-6 place entities at mouse position\n")
    
    sandbox.run()


if __name__ == "__main__":
    main()
