# geometry.pxd — C-level declarations for cross-module cimport.
# Allows pathfinding.pyx to call geometry functions at C speed.

cdef bint c_is_on_pillar(int x, int y, list pillar_alive)
cdef int c_get_los_mask(int x, int y, list pillar_alive)
cdef bint c_is_in_bounds(int x, int y)
cdef bint c_is_valid_tile(int x, int y, list pillar_alive)
cdef bint c_is_valid_tile_for_size(int x, int y, int size, list pillar_alive)
cdef bint c_do_footprints_overlap(int x1, int y1, int s1, int x2, int y2, int s2)
cdef bint c_would_overlap_pillar(int x, int y, int size, list pillar_alive)
cdef bint c_would_npc_overlap_player_at(int nx, int ny, int ns, int px, int py)
cdef (int, int) c_compute_push_out_tile(int px, int py, int nx, int ny, int ns, list pillar_alive)
cdef int c_chebyshev_distance(int x1, int y1, int x2, int y2)
cdef bint c_has_line_of_sight(int x1, int y1, int x2, int y2,
                               int size, int attack_range, bint is_npc,
                               list pillar_alive)
