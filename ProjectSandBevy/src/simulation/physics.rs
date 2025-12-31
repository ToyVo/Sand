use crate::elements::Element;
use crate::simulation::grid::GameGrid;
use crate::particles::ParticleList;
use bevy::prelude::*;
use rand::Rng;

/// Helper functions for physics simulation, ported from TypeScript

/// Pick randomly between two valid indices (returns Option<usize>)
fn pick_rand_valid(a: Option<usize>, b: Option<usize>) -> Option<usize> {
    match (a, b) {
        (Some(a_val), Some(b_val)) => {
            if rand::thread_rng().gen_bool(0.5) {
                Some(a_val)
            } else {
                Some(b_val)
            }
        }
        (Some(a_val), None) => Some(a_val),
        (None, Some(b_val)) => Some(b_val),
        (None, None) => None,
    }
}

/// Check pixel immediately below
fn below(grid: &GameGrid, y: u32, i: usize, target: Element) -> Option<usize> {
    if y >= grid.max_y() {
        return None;
    }
    let below_idx = i + grid.width as usize;
    if below_idx < grid.elements.len() && grid.get_index(below_idx) == target {
        Some(below_idx)
    } else {
        None
    }
}

/// Check pixel below and the 2 diagonally below
fn below_adjacent(grid: &GameGrid, x: u32, y: u32, i: usize, target: Element) -> Option<usize> {
    if y >= grid.max_y() {
        return None;
    }
    let below_idx = i + grid.width as usize;
    
    // Check directly below
    if below_idx < grid.elements.len() && grid.get_index(below_idx) == target {
        return Some(below_idx);
    }
    
    // Check below-left
    if x > 0 {
        let below_left_idx = below_idx - 1;
        if below_left_idx < grid.elements.len() && grid.get_index(below_left_idx) == target {
            return Some(below_left_idx);
        }
    }
    
    // Check below-right
    if x < grid.max_x() {
        let below_right_idx = below_idx + 1;
        if below_right_idx < grid.elements.len() && grid.get_index(below_right_idx) == target {
            return Some(below_right_idx);
        }
    }
    
    None
}

/// Check pixel immediately above
fn above(grid: &GameGrid, y: u32, i: usize, target: Element) -> Option<usize> {
    if y == 0 {
        return None;
    }
    let above_idx = i.saturating_sub(grid.width as usize);
    if grid.get_index(above_idx) == target {
        Some(above_idx)
    } else {
        None
    }
}

/// Check pixel above and the 2 diagonally above
fn above_adjacent(grid: &GameGrid, x: u32, y: u32, i: usize, target: Element) -> Option<usize> {
    if y == 0 {
        return None;
    }
    let above_idx = i.saturating_sub(grid.width as usize);
    
    // Check directly above
    if grid.get_index(above_idx) == target {
        return Some(above_idx);
    }
    
    // Check above-left
    if x > 0 {
        let above_left_idx = above_idx - 1;
        if grid.get_index(above_left_idx) == target {
            return Some(above_left_idx);
        }
    }
    
    // Check above-right
    if x < grid.max_x() {
        let above_right_idx = above_idx + 1;
        if above_right_idx < grid.elements.len() && grid.get_index(above_right_idx) == target {
            return Some(above_right_idx);
        }
    }
    
    None
}

/// Check left and right adjacent pixels
fn adjacent(grid: &GameGrid, x: u32, i: usize, target: Element) -> Option<usize> {
    let left_idx = if x > 0 { Some(i - 1) } else { None };
    let right_idx = if x < grid.max_x() { Some(i + 1) } else { None };
    
    let left_match = left_idx.and_then(|idx| {
        if grid.get_index(idx) == target {
            Some(idx)
        } else {
            None
        }
    });
    
    let right_match = right_idx.and_then(|idx| {
        if grid.get_index(idx) == target {
            Some(idx)
        } else {
            None
        }
    });
    
    pick_rand_valid(left_match, right_match)
}

/// Apply gravity to an element
/// Returns true if the element moved
/// fall_into_void: if true, elements disappear at bottom edge; if false, they stop
pub fn do_gravity(
    grid: &mut GameGrid,
    x: u32,
    y: u32,
    i: usize,
    fall_adjacent: bool,
    chance: f64,
    fall_into_void: bool,
    rainbow_sand_times: &mut Option<&mut std::collections::HashMap<usize, u32>>,
) -> bool {
    if !rand::thread_rng().gen_bool(chance) {
        return false;
    }

    if y >= grid.max_y() {
        if fall_into_void {
            let element = grid.get_index(i);
            grid.set_index(i, Element::Background);
            // Remove placement time if RainbowSand falls into void
            if let Some(times) = rainbow_sand_times.as_mut() {
                if element == Element::RainbowSand {
                    times.remove(&i);
                }
            }
            return true;
        }
        // Stop at edge, don't fall into void
        return false;
    }

    let new_i = if fall_adjacent {
        below_adjacent(grid, x, y, i, Element::Background)
    } else {
        below(grid, y, i, Element::Background)
    };

    let new_i = new_i.or_else(|| {
        if fall_adjacent {
            adjacent(grid, x, i, Element::Background)
        } else {
            None
        }
    });

    if let Some(new_idx) = new_i {
        let element = grid.get_index(i);
        grid.set_index(new_idx, element);
        grid.set_index(i, Element::Background);
        
        // Transfer placement time if RainbowSand moved
        if let Some(times) = rainbow_sand_times.as_mut() {
            if element == Element::RainbowSand {
                if let Some(placement_time) = times.remove(&i) {
                    times.insert(new_idx, placement_time);
                }
            }
        }
        
        return true;
    }

    false
}

/// Density sink for solid elements (e.g., sand sinking through water)
/// The current element sinks through the lighter element below it
/// Returns true if the element moved
pub fn do_density_sink(
    grid: &mut GameGrid,
    x: u32,
    y: u32,
    i: usize,
    lighter_than: Element,
    sink_adjacent: bool,
    chance: f64,
    _fall_into_void: bool,
    rainbow_sand_times: &mut Option<&mut std::collections::HashMap<usize, u32>>,
) -> bool {
    if !rand::thread_rng().gen_bool(chance) {
        return false;
    }

    if y >= grid.max_y() {
        return false;
    }

    let new_i = if sink_adjacent {
        below_adjacent(grid, x, y, i, lighter_than)
    } else {
        below(grid, y, i, lighter_than)
    };

    if let Some(new_idx) = new_i {
        let current_element = grid.get_index(i);
        grid.set_index(new_idx, current_element);
        grid.set_index(i, lighter_than);
        
        // Transfer placement time if RainbowSand moved
        if let Some(times) = rainbow_sand_times.as_mut() {
            if current_element == Element::RainbowSand {
                if let Some(placement_time) = times.remove(&i) {
                    times.insert(new_idx, placement_time);
                }
            }
        }
        
        return true;
    }

    false
}

/// Density-based liquid interaction (e.g., water sinking through oil)
/// Returns true if the element moved
pub fn do_density_liquid(
    grid: &mut GameGrid,
    x: u32,
    y: u32,
    i: usize,
    heavier_than: Element,
    sink_chance: f64,
    equalize_chance: f64,
) -> bool {
    let mut new_i = None;

    if rand::thread_rng().gen_bool(sink_chance) {
        new_i = below_adjacent(grid, x, y, i, heavier_than);
    }

    if new_i.is_none() && rand::thread_rng().gen_bool(equalize_chance) {
        new_i = adjacent(grid, x, i, heavier_than);
    }

    if let Some(new_idx) = new_i {
        let current_element = grid.get_index(i);
        grid.set_index(new_idx, current_element);
        grid.set_index(i, heavier_than);
        return true;
    }

    false
}

/// Transform element when touching another element
/// Returns true if transformation occurred
fn do_transform(
    grid: &mut GameGrid,
    x: u32,
    y: u32,
    i: usize,
    transform_by: Element,
    transform_into: Element,
    transform_chance: f64,
    consume_chance: f64,
) -> bool {
    let mut rng = rand::thread_rng();
    if !rng.gen_bool(transform_chance) {
        return false;
    }
    
    if let Some(transform_loc) = bordering(grid, x, y, i, transform_by) {
        grid.set_index(i, transform_into);
        if rng.gen_bool(consume_chance) {
            grid.set_index(transform_loc, transform_into);
        }
        return true;
    }
    
    false
}

/// Grow element by converting adjacent target element to current element
/// Returns true if growth occurred
#[allow(dead_code)]
fn do_grow(
    grid: &mut GameGrid,
    x: u32,
    y: u32,
    i: usize,
    into: Element,
    chance: f64,
) -> bool {
    let mut rng = rand::thread_rng();
    if !rng.gen_bool(chance) {
        return false;
    }
    
    if let Some(grow_loc) = bordering_adjacent(grid, x, y, i, into) {
        let current_element = grid.get_index(i);
        grid.set_index(grow_loc, current_element);
        return true;
    }
    
    false
}

/// Tree branch generation state
#[derive(Clone)]
pub struct TreeBranch {
    x: f32,
    y: f32,
    angle: f32, // Angle in radians
    generation: u32,
    max_branches: u32,
    branch_spacing: u32,
    next_branch: u32,
    branches_created: u32,
    tree_type: u32,
    iterations: u32, // Track iterations for this branch
}

/// Resource to store active tree branches for incremental growth
#[derive(Resource, Default)]
pub struct ActiveTreeBranches {
    pub branches: Vec<TreeBranch>,
}

/// Start a new tree generation (adds initial branch to active branches)
/// The tree will grow incrementally over multiple frames
pub fn start_tree_generation(active_branches: &mut ActiveTreeBranches, start_x: u32, start_y: u32) {
    let mut rng = rand::thread_rng();
    
    // Tree parameters (similar to TREE_PARTICLE_INIT)
    let initial_angle = -std::f32::consts::PI / 2.0 - std::f32::consts::PI / 8.0 + rng.gen_range(0.0..1.0) * std::f32::consts::PI / 4.0;
    
    let branch_spacing = 15 + rng.gen_range(0..=45);
    let max_branches = 1 + rng.gen_range(0..=2);
    let tree_type = if rng.gen_bool(0.62) { 0 } else { 1 };
    
    // Add initial branch
    active_branches.branches.push(TreeBranch {
        x: start_x as f32,
        y: start_y as f32,
        angle: initial_angle,
        generation: 1,
        max_branches,
        branch_spacing,
        next_branch: branch_spacing,
        branches_created: 0,
        tree_type,
        iterations: 0,
    });
}

/// Process tree branches incrementally (called each frame)
/// Similar to TREE_PARTICLE_ACTION in TypeScript
pub fn process_tree_branches(grid: &mut GameGrid, active_branches: &mut ActiveTreeBranches) {
    let mut rng = rand::thread_rng();
    
    let mut new_branches = Vec::new();
    let mut branches_to_remove = Vec::new();
    
    for (idx, branch) in active_branches.branches.iter_mut().enumerate() {
        branch.iterations += 1;
        
        // Move branch forward along its angle (one step per frame, like particle system)
        // TypeScript setVelocity: xVelocity = velocity * cos(angle), yVelocity = velocity * sin(angle)
        // velocity = 1 + Math.random() * 0.5
        // For angle = -90°: cos(-90°) = 0, sin(-90°) = -1
        // So xVelocity = 0, yVelocity = -velocity (moves up, since y increases downward)
        let velocity = 1.0 + rng.gen_range(0.0..1.0) * 0.5;
        let dx = branch.angle.cos() * velocity;
        let dy = branch.angle.sin() * velocity; // In TypeScript, y increases downward, so negative y is up
        
        let new_x = branch.x + dx;
        let new_y = branch.y + dy;
        
        // Check bounds
        if new_x < 0.0 || new_x >= grid.width as f32 || new_y < 0.0 || new_y >= grid.height as f32 {
            branches_to_remove.push(idx);
            continue;
        }
        
        let new_x_int = new_x as u32;
        let new_y_int = new_y as u32;
        let new_idx = grid.xy_to_index(new_x_int, new_y_int);
        
        if new_idx >= grid.elements.len() {
            branches_to_remove.push(idx);
            continue;
        }
        
        // Check if we hit a wall
        if grid.get_index(new_idx) == Element::Wall {
            branches_to_remove.push(idx);
            continue;
        }
        
        // Place branch element
        if grid.get_index(new_idx) == Element::Background {
            grid.set_index(new_idx, Element::Branch);
        }
        
        // Update branch position
        branch.x = new_x;
        branch.y = new_y;
        
        // Check if it's time to create sub-branches
        if branch.iterations >= branch.next_branch {
            branch.branches_created += 1;
            
            if branch.max_branches == 0 {
                // End of branch - place leaf
                if grid.get_index(new_idx) == Element::Branch {
                    grid.set_index(new_idx, Element::Leaf);
                }
                branches_to_remove.push(idx);
                continue;
            }
            
            // Calculate branch angles (similar to Tree0.branchAngles)
            let branch_angle = std::f32::consts::PI / 8.0 + rng.gen_range(0.0..1.0) * std::f32::consts::PI / 4.0;
            let left_angle = branch.angle + branch_angle;
            let right_angle = branch.angle - branch_angle;
            
            // Create left branch
            let left_branch_spacing = (branch.branch_spacing as f32 * 0.9) as u32;
            new_branches.push(TreeBranch {
                x: branch.x,
                y: branch.y,
                angle: left_angle,
                generation: branch.generation + 1,
                max_branches: branch.max_branches.saturating_sub(1),
                branch_spacing: left_branch_spacing,
                next_branch: left_branch_spacing,
                branches_created: 0,
                tree_type: branch.tree_type,
                iterations: 0,
            });
            
            // Create right branch
            new_branches.push(TreeBranch {
                x: branch.x,
                y: branch.y,
                angle: right_angle,
                generation: branch.generation + 1,
                max_branches: branch.max_branches.saturating_sub(1),
                branch_spacing: left_branch_spacing,
                next_branch: left_branch_spacing,
                branches_created: 0,
                tree_type: branch.tree_type,
                iterations: 0,
            });
            
            // Update next branch time
            if branch.branch_spacing > 45 {
                branch.branch_spacing = (branch.branch_spacing as f32 * 0.8) as u32;
            }
            branch.next_branch = branch.iterations + (branch.branch_spacing as f32 * (0.65 + rng.gen_range(0.0..1.0) * 0.35)) as u32;
        }
        
        // If branch has created all its sub-branches, end it with a leaf
        if branch.branches_created >= branch.max_branches {
            if grid.get_index(new_idx) == Element::Branch {
                grid.set_index(new_idx, Element::Leaf);
            }
            branches_to_remove.push(idx);
        }
    }
    
    // Remove finished branches (in reverse order to maintain indices)
    for &idx in branches_to_remove.iter().rev() {
        active_branches.branches.remove(idx);
    }
    
    // Add new branches
    active_branches.branches.extend(new_branches);
}

/// Create a multi-directional explosion pattern (for magic effects, napalm, methane, etc.)
/// Creates fire in multiple directions radiating from the center
fn create_radial_explosion(
    grid: &mut GameGrid,
    center_x: u32,
    center_y: u32,
    radius: u32,
    num_directions: u32,
) {
    let mut rng = rand::thread_rng();
    
    // Create fire in multiple directions
    for dir in 0..num_directions {
        let angle = (dir as f32 / num_directions as f32) * 2.0 * std::f32::consts::PI;
        let dx = angle.cos();
        let dy = angle.sin();
        
        // Create fire along the ray
        for step in 1..=radius {
            let offset_x = (dx * step as f32).round() as i32;
            let offset_y = (dy * step as f32).round() as i32;
            
            let new_x = center_x as i32 + offset_x;
            let new_y = center_y as i32 + offset_y;
            
            if new_x >= 0 && new_x < grid.width as i32 && new_y >= 0 && new_y < grid.height as i32 {
                let idx = grid.xy_to_index(new_x as u32, new_y as u32);
                if idx < grid.elements.len() {
                    let elem = grid.get_index(idx);
                    // Only place fire on background or flammable materials
                    if elem == Element::Background || matches!(
                        elem,
                        Element::Plant | Element::Wax | Element::Oil | Element::Napalm
                    ) {
                        grid.set_index(idx, Element::Fire);
                    }
                }
            }
        }
    }
}

/// Create a vertical column of fire going upward (for charged nitro explosion)
#[allow(dead_code)]
fn create_vertical_fire_column(grid: &mut GameGrid, start_x: u32, start_y: u32) {
    // Search upward for wall or top of screen
    let mut y = start_y;
    while y > 0 {
        y -= 1;
        let idx = grid.xy_to_index(start_x, y);
        if idx >= grid.elements.len() {
            break;
        }
        
        let elem = grid.get_index(idx);
        if elem == Element::Wall {
            break;
        }
        
        // Place fire (overwrite most things except wall)
        if elem != Element::Wall {
            grid.set_index(idx, Element::Fire);
        }
    }
}

/// Create a large expanding explosion pattern (for C4)
#[allow(dead_code)]
fn create_c4_explosion(grid: &mut GameGrid, center_x: u32, center_y: u32) {
    let mut rng = rand::thread_rng();
    
    // Create multiple expanding rings of fire
    let max_radius = 8;
    for radius in 1..=max_radius {
        // Create a circular pattern
        let num_points = radius * 8; // More points for larger radius
        for i in 0..num_points {
            let angle = (i as f32 / num_points as f32) * 2.0 * std::f32::consts::PI;
            let dx = angle.cos() * radius as f32;
            let dy = angle.sin() * radius as f32;
            
            let new_x = center_x as i32 + dx.round() as i32;
            let new_y = center_y as i32 + dy.round() as i32;
            
            if new_x >= 0 && new_x < grid.width as i32 && new_y >= 0 && new_y < grid.height as i32 {
                let idx = grid.xy_to_index(new_x as u32, new_y as u32);
                if idx < grid.elements.len() {
                    let elem = grid.get_index(idx);
                    // Only place fire on background or flammable materials
                    if elem == Element::Background || matches!(
                        elem,
                        Element::Plant | Element::Wax | Element::Oil | Element::Napalm | Element::Gunpowder
                    ) {
                        // Random chance to place fire (creates more interesting pattern)
                        if rng.gen_bool(0.7) {
                            grid.set_index(idx, Element::Fire);
                        }
                    }
                }
            }
        }
    }
}

/// Producer element - generates target element in adjacent positions
/// Returns true if production occurred
fn do_producer(
    grid: &mut GameGrid,
    x: u32,
    y: u32,
    i: usize,
    produce: Element,
    overwrite_adjacent: bool,
    chance: f64,
) -> bool {
    let mut rng = rand::thread_rng();
    if !rng.gen_bool(chance) {
        return false;
    }
    
    // Produce in up, down, left, right directions
    if y > 0 {
        let up_idx = i.saturating_sub(grid.width as usize);
        if overwrite_adjacent || grid.get_index(up_idx) == Element::Background {
            grid.set_index(up_idx, produce);
        }
    }
    if y < grid.max_y() {
        let down_idx = i + grid.width as usize;
        if down_idx < grid.elements.len() {
            if overwrite_adjacent || grid.get_index(down_idx) == Element::Background {
                grid.set_index(down_idx, produce);
            }
        }
    }
    if x > 0 {
        let left_idx = i - 1;
        if overwrite_adjacent || grid.get_index(left_idx) == Element::Background {
            grid.set_index(left_idx, produce);
        }
    }
    if x < grid.max_x() {
        let right_idx = i + 1;
        if right_idx < grid.elements.len() {
            if overwrite_adjacent || grid.get_index(right_idx) == Element::Background {
                grid.set_index(right_idx, produce);
            }
        }
    }
    
    true
}

/// Check if an element is bordering (up, down, left, right) a target element
fn bordering(grid: &GameGrid, x: u32, y: u32, i: usize, target: Element) -> Option<usize> {
    // Check below
    if y < grid.max_y() {
        let below_idx = i + grid.width as usize;
        if grid.get_index(below_idx) == target {
            return Some(below_idx);
        }
    }
    
    // Check adjacent (left/right)
    if let Some(adj_idx) = adjacent(grid, x, i, target) {
        return Some(adj_idx);
    }
    
    // Check above
    if y > 0 {
        if let Some(above_idx) = above(grid, y, i, target) {
            return Some(above_idx);
        }
    }
    
    None
}

/// Check if an element is bordering adjacent (all 8 directions including corners) a target element
fn bordering_adjacent(grid: &GameGrid, x: u32, y: u32, i: usize, target: Element) -> Option<usize> {
    // Check below adjacent
    if y < grid.max_y() {
        if let Some(below_idx) = below_adjacent(grid, x, y, i, target) {
            return Some(below_idx);
        }
    }
    
    // Check adjacent (left/right)
    if let Some(adj_idx) = adjacent(grid, x, i, target) {
        return Some(adj_idx);
    }
    
    // Check above adjacent
    if y > 0 {
        if let Some(above_idx) = above_adjacent(grid, x, y, i, target) {
            return Some(above_idx);
        }
    }
    
    None
}

/// Check if element is surrounded by target element (up, down, left, right only, not corners)
fn surrounded_by(grid: &GameGrid, x: u32, y: u32, i: usize, target: Element) -> bool {
    if y < grid.max_y() {
        let below_idx = i + grid.width as usize;
        if grid.get_index(below_idx) != target {
            return false;
        }
    }
    if y > 0 {
        let above_idx = i.saturating_sub(grid.width as usize);
        if grid.get_index(above_idx) != target {
            return false;
        }
    }
    if x > 0 {
        let left_idx = i - 1;
        if grid.get_index(left_idx) != target {
            return false;
        }
    }
    if x < grid.max_x() {
        let right_idx = i + 1;
        if right_idx < grid.elements.len() && grid.get_index(right_idx) != target {
            return false;
        }
    }
    true
}

/// Check if element is surrounded by target element (all 8 directions including corners)
fn surrounded_by_adjacent(grid: &GameGrid, x: u32, y: u32, i: usize, target: Element) -> bool {
    let at_bottom = y >= grid.max_y();
    let at_top = y == 0;
    
    let below_idx = if !at_bottom { i + grid.width as usize } else { usize::MAX };
    let above_idx = if !at_top { i.saturating_sub(grid.width as usize) } else { usize::MAX };
    
    if !at_bottom {
        if grid.get_index(below_idx) != target {
            return false;
        }
    }
    if !at_top {
        if grid.get_index(above_idx) != target {
            return false;
        }
    }
    
    if x > 0 {
        let left_idx = i - 1;
        if grid.get_index(left_idx) != target {
            return false;
        }
        if !at_top {
            let above_left_idx = above_idx - 1;
            if grid.get_index(above_left_idx) != target {
                return false;
            }
        }
        if !at_bottom {
            let below_left_idx = below_idx - 1;
            if below_left_idx < grid.elements.len() && grid.get_index(below_left_idx) != target {
                return false;
            }
        }
    }
    
    if x < grid.max_x() {
        let right_idx = i + 1;
        if right_idx < grid.elements.len() && grid.get_index(right_idx) != target {
            return false;
        }
        if !at_top {
            let above_right_idx = above_idx + 1;
            if above_right_idx < grid.elements.len() && grid.get_index(above_right_idx) != target {
                return false;
            }
        }
        if !at_bottom {
            let below_right_idx = below_idx + 1;
            if below_right_idx < grid.elements.len() && grid.get_index(below_right_idx) != target {
                return false;
            }
        }
    }
    
    true
}

/// Make element rise (opposite of gravity, for gases)
/// Returns true if the element moved
pub fn do_rise(
    grid: &mut GameGrid,
    x: u32,
    y: u32,
    i: usize,
    rise_chance: f64,
    adjacent_chance: f64,
    fall_into_void: bool,
) -> bool {
    let mut rng = rand::thread_rng();
    let mut new_i = None;
    
    if rng.gen_bool(rise_chance) {
        if y == 0 {
            if fall_into_void {
                grid.set_index(i, Element::Background);
                return true;
            }
            return false;
        }
        new_i = above_adjacent(grid, x, y, i, Element::Background);
    }
    
    if new_i.is_none() && rng.gen_bool(adjacent_chance) {
        new_i = adjacent(grid, x, i, Element::Background);
    }
    
    if let Some(new_idx) = new_i {
        let current_element = grid.get_index(i);
        grid.set_index(new_idx, current_element);
        grid.set_index(i, Element::Background);
        return true;
    }
    
    false
}

/// Execute element action based on element type
pub fn execute_element_action(
    grid: &mut GameGrid,
    x: u32,
    y: u32,
    i: usize,
    fall_into_void: bool,
    particle_list: Option<&mut ParticleList>,
    active_branches: Option<&mut ActiveTreeBranches>,
    rainbow_sand_times: &mut Option<&mut std::collections::HashMap<usize, u32>>,
) {
    let element = grid.get_index(i);
    
    match element {
        Element::Background => {
            // Background does nothing
        }
        Element::Wall => {
            // Wall is static
        }
        Element::Sand => {
            // Sand can sink through liquids (sand is heavier)
            if y < grid.max_y() {
                if do_density_sink(grid, x, y, i, Element::Water, true, 0.25, fall_into_void, rainbow_sand_times) {
                    return;
                }
                if do_density_sink(grid, x, y, i, Element::SaltWater, true, 0.25, fall_into_void, rainbow_sand_times) {
                    return;
                }
            }
            // Sand falls with gravity, can fall diagonally (fall_adjacent = true)
            do_gravity(grid, x, y, i, true, 0.95, fall_into_void, rainbow_sand_times);
        }
        Element::Water => {
            // Water falls with gravity (95% chance), can flow adjacent
            // Water can sink through oil (water is heavier than oil)
            if !do_density_liquid(grid, x, y, i, Element::Oil, 0.25, 0.50) {
                do_gravity(grid, x, y, i, true, 0.95, fall_into_void, rainbow_sand_times);
            }
        }
        Element::Fire => {
            // Fire spreads and can be extinguished by water
            let mut rng = rand::thread_rng();
            
            // Check for water or salt water to extinguish (80% chance)
            if rng.gen_bool(0.80) {
                if let Some(water_loc) = bordering(grid, x, y, i, Element::Water) {
                    // Extinguish fire, turn water to steam
                    grid.set_index(water_loc, Element::Steam);
                    grid.set_index(i, Element::Background);
                    return;
                }
                if let Some(salt_water_loc) = bordering(grid, x, y, i, Element::SaltWater) {
                    // Extinguish fire, turn salt water to steam
                    grid.set_index(salt_water_loc, Element::Steam);
                    grid.set_index(i, Element::Background);
                    return;
                }
            }
            
            // Fire can spread to plant (20% chance)
            if rng.gen_bool(0.20) {
                if let Some(plant_loc) = bordering_adjacent(grid, x, y, i, Element::Plant) {
                    grid.set_index(plant_loc, Element::Fire);
                    return;
                }
            }
            
            // Fire can spread to fuse (80% chance)
            if rng.gen_bool(0.80) {
                if let Some(fuse_loc) = bordering_adjacent(grid, x, y, i, Element::Fuse) {
                    grid.set_index(fuse_loc, Element::Fire);
                    return;
                }
            }
            
            // Fire can spread to branch (20% chance)
            if rng.gen_bool(0.20) {
                if let Some(branch_loc) = bordering_adjacent(grid, x, y, i, Element::Branch) {
                    grid.set_index(branch_loc, Element::Fire);
                    return;
                }
            }
            
            // Fire can spread to leaf (20% chance)
            if rng.gen_bool(0.20) {
                if let Some(leaf_loc) = bordering_adjacent(grid, x, y, i, Element::Leaf) {
                    grid.set_index(leaf_loc, Element::Fire);
                    return;
                }
            }
            
            // Fire can spread to wax (1% chance, bordering not adjacent - only direct neighbors)
            if rng.gen_bool(0.01) {
                if let Some(wax_loc) = bordering(grid, x, y, i, Element::Wax) {
                    grid.set_index(wax_loc, Element::Fire);
                    // Create falling wax below the wax if there's space
                    let (_wax_x, wax_y) = grid.index_to_xy(wax_loc);
                    if let Some(below_idx) = below(grid, wax_y.max(y), wax_loc.max(i), Element::Background) {
                        grid.set_index(below_idx, Element::FallingWax);
                    }
                    return;
                }
            }
            
            // Fire can rise upward (50% chance)
            if rng.gen_bool(0.50) {
                if let Some(above_idx) = above(grid, y, i, Element::Background) {
                    grid.set_index(above_idx, Element::Fire);
                    return;
                }
            }
            
            // Fire can spread to oil (20% chance)
            if rng.gen_bool(0.20) {
                if let Some(oil_loc) = bordering_adjacent(grid, x, y, i, Element::Oil) {
                    grid.set_index(oil_loc, Element::Fire);
                    return;
                }
            }
            
            // Fire can flame out (40% chance) if no flammable materials nearby
            // Check all 8 adjacent positions for flammable materials
            if rng.gen_bool(0.40) {
                let mut has_flammable = false;
                
                // Check all 8 directions (including corners)
                let x_start = x.saturating_sub(1);
                let y_start = y.saturating_sub(1);
                let x_end = (x + 2).min(grid.max_x() + 1);
                let y_end = (y + 2).min(grid.max_y() + 1);
                
                for y_iter in y_start..y_end {
                    for x_iter in x_start..x_end {
                        if y_iter == y && x_iter == x {
                            continue;
                        }
                        
                        let idx = grid.xy_to_index(x_iter, y_iter);
                        if idx >= grid.elements.len() {
                            continue;
                        }
                        
                        let elem = grid.get_index(idx);
                        
                        // Skip if it's already fire
                        if elem == Element::Fire {
                            continue;
                        }
                        
                        // Check for flammable materials
                        if matches!(
                            elem,
                            Element::Plant | Element::Fuse | Element::Branch | Element::Leaf
                        ) {
                            has_flammable = true;
                            break;
                        }
                        
                        // Wax only counts if directly adjacent (not corners)
                        if (x_iter == x || y_iter == y) && elem == Element::Wax {
                            has_flammable = true;
                            break;
                        }
                        
                        // Oil has 50% chance to prevent flameout
                        if elem == Element::Oil && rng.gen_bool(0.50) {
                            has_flammable = true;
                            break;
                        }
                    }
                    if has_flammable {
                        break;
                    }
                }
                
                // Flame out if no flammable materials nearby
                if !has_flammable {
                    grid.set_index(i, Element::Background);
                    return;
                }
            }
        }
        Element::Salt => {
            // Salt falls with gravity
            if do_gravity(grid, x, y, i, true, 0.95, fall_into_void, rainbow_sand_times) {
                return;
            }
            // Salt can dissolve in water to create salt water (25% chance, 50% consume)
            if do_transform(grid, x, y, i, Element::Water, Element::SaltWater, 0.25, 0.50) {
                return;
            }
            // Salt can sink through salt water
            if y < grid.max_y() {
                if do_density_sink(grid, x, y, i, Element::SaltWater, true, 0.25, fall_into_void, rainbow_sand_times) {
                    return;
                }
            }
        }
        Element::Oil => {
            // Oil can catch fire (30% chance)
            let mut rng = rand::thread_rng();
            if rng.gen_bool(0.30) {
                if let Some(_fire_loc) = bordering(grid, x, y, i, Element::Fire) {
                    // Set surrounding pixels on fire
                    if y > 0 {
                        let above_idx = i.saturating_sub(grid.width as usize);
                        if grid.get_index(above_idx) == Element::Background {
                            grid.set_index(above_idx, Element::Fire);
                        }
                    }
                    if y < grid.max_y() {
                        let below_idx = i + grid.width as usize;
                        if below_idx < grid.elements.len() && grid.get_index(below_idx) == Element::Background {
                            grid.set_index(below_idx, Element::Fire);
                        }
                    }
                    if x > 0 {
                        let left_idx = i - 1;
                        if grid.get_index(left_idx) == Element::Background {
                            grid.set_index(left_idx, Element::Fire);
                        }
                    }
                    if x < grid.max_x() {
                        let right_idx = i + 1;
                        if right_idx < grid.elements.len() && grid.get_index(right_idx) == Element::Background {
                            grid.set_index(right_idx, Element::Fire);
                        }
                    }
                    grid.set_index(i, Element::Fire);
                    return;
                }
            }
            // Oil falls with gravity (lighter than water, so floats)
            do_gravity(grid, x, y, i, true, 0.95, fall_into_void, rainbow_sand_times);
        }
        Element::Rock => {
            // Rock is heavy and sinks through liquids
            if y < grid.max_y() {
                // Rock sinks through water, oil (95% chance)
                if do_density_sink(grid, x, y, i, Element::Water, false, 0.95, fall_into_void, rainbow_sand_times) {
                    return;
                }
                if do_density_sink(grid, x, y, i, Element::Oil, false, 0.95, fall_into_void, rainbow_sand_times) {
                    return;
                }
            }
            // Rock falls with gravity (99% chance, no diagonal falling)
            do_gravity(grid, x, y, i, false, 0.99, fall_into_void, rainbow_sand_times);
            
            // Rock produces methane when in contact with oil above (1% * 20% = 0.2% chance)
            let mut rng = rand::thread_rng();
            if rng.gen_bool(0.01) && rng.gen_bool(0.20) {
                if let Some(oil_loc) = above(grid, y, i, Element::Oil) {
                    if rng.gen_bool(0.50) {
                        grid.set_index(oil_loc, Element::Methane);
                    } else {
                        grid.set_index(i, Element::Methane);
                    }
                    return;
                }
            }
        }
        Element::Ice => {
            // Ice melts when touching heat sources
            let mut rng = rand::thread_rng();
            
            // Skip if surrounded by ice (optimization)
            if surrounded_by(grid, x, y, i, Element::Ice) {
                return;
            }
            
            // Slow melt from water (1% chance)
            if rng.gen_bool(0.01) {
                if let Some(_water_loc) = bordering(grid, x, y, i, Element::Water) {
                    grid.set_index(i, Element::Water);
                    return;
                }
            }
            
            // Fast melt from steam (70% chance)
            if rng.gen_bool(0.70) {
                if let Some(steam_loc) = bordering(grid, x, y, i, Element::Steam) {
                    grid.set_index(i, Element::Water);
                    if rng.gen_bool(0.50) {
                        grid.set_index(steam_loc, Element::Water);
                    }
                    return;
                }
            }
            
            // Fast melt from salt or salt water (10% chance)
            if rng.gen_bool(0.10) {
                if let Some(_salt_loc) = bordering(grid, x, y, i, Element::Salt) {
                    grid.set_index(i, Element::Water);
                    return;
                }
                if let Some(_salt_water_loc) = bordering(grid, x, y, i, Element::SaltWater) {
                    grid.set_index(i, Element::Water);
                    return;
                }
            }
            
            // Fast melt from fire (50% chance)
            if rng.gen_bool(0.50) {
                if let Some(_fire_loc) = bordering(grid, x, y, i, Element::Fire) {
                    grid.set_index(i, Element::Water);
                    return;
                }
            }
            
            // Fast melt from lava (50% chance)
            if rng.gen_bool(0.50) {
                if let Some(_lava_loc) = bordering(grid, x, y, i, Element::Lava) {
                    grid.set_index(i, Element::Water);
                    return;
                }
            }
        }
        Element::Lava => {
            // Lava falls with gravity and burns things
            let mut rng = rand::thread_rng();
            
            // Lava touching water or salt water turns to rock and liquid to steam
            if let Some(water_loc) = bordering(grid, x, y, i, Element::Water) {
                grid.set_index(water_loc, Element::Steam);
                grid.set_index(i, Element::Rock);
                return;
            }
            if let Some(salt_water_loc) = bordering(grid, x, y, i, Element::SaltWater) {
                grid.set_index(salt_water_loc, Element::Steam);
                grid.set_index(i, Element::Rock);
                return;
            }
            
            // Lava can burn adjacent elements (25% chance)
            if rng.gen_bool(0.25) {
                let burn_locs = [
                    if y > 0 { Some(i.saturating_sub(grid.width as usize)) } else { None },
                    if y < grid.max_y() { Some(i + grid.width as usize) } else { None },
                    if x > 0 { Some(i - 1) } else { None },
                    if x < grid.max_x() { Some(i + 1) } else { None },
                ];
                
                for burn_loc_opt in burn_locs.iter() {
                    if let Some(burn_loc) = burn_loc_opt {
                        if *burn_loc < grid.elements.len() {
                            let elem = grid.get_index(*burn_loc);
                            // Lava immune elements: Lava, Background, Fire, Wall, Rock, Water, Steam
                            let should_burn = !matches!(
                                elem,
                                Element::Lava | Element::Background | Element::Fire
                                    | Element::Wall | Element::Rock | Element::Water | Element::Steam
                            );
                            if should_burn {
                                grid.set_index(*burn_loc, Element::Fire);
                            }
                        }
                    }
                }
            }
            
            // Lava can create fire above (6% chance)
            if rng.gen_bool(0.06) && y > 0 {
                let above_idx = i.saturating_sub(grid.width as usize);
                if grid.get_index(above_idx) == Element::Background {
                    grid.set_index(above_idx, Element::Fire);
                }
            }
            
            // Allow steam to pass through (95% chance)
            if y < grid.max_y() {
                let below_idx = i + grid.width as usize;
                if below_idx < grid.elements.len() {
                    let below_elem = grid.get_index(below_idx);
                    if below_elem == Element::Steam && rng.gen_bool(0.95) {
                        grid.set_index(below_idx, Element::Lava);
                        grid.set_index(i, Element::Steam);
                        return;
                    }
                }
            }
            
            // Lava falls with gravity (100% chance, can fall diagonally)
            do_gravity(grid, x, y, i, true, 1.0, fall_into_void, rainbow_sand_times);
        }
        Element::Steam => {
            // Steam rises and condenses
            let mut rng = rand::thread_rng();
            
            // Steam rises (70% chance)
            if do_rise(grid, x, y, i, 0.70, 0.60, fall_into_void) {
                return;
            }
            
            // Condense due to water (5% chance)
            if rng.gen_bool(0.05) {
                if let Some(_water_loc) = bordering(grid, x, y, i, Element::Water) {
                    grid.set_index(i, Element::Water);
                    return;
                }
            }
            
            // Condense/disappear due to air cooling (5% * 40% = 2% chance)
            if rng.gen_bool(0.05) && rng.gen_bool(0.40) {
                let below_bg = below(grid, y, i, Element::Background);
                let above_bg = if y > 0 { above(grid, y, i, Element::Background) } else { None };
                if below_bg.is_some() && above_bg.is_none() {
                    if rng.gen_bool(0.30) {
                        grid.set_index(i, Element::Water);
                    } else {
                        grid.set_index(i, Element::Background);
                    }
                    return;
                }
            }
            
            // Condense due to spout (5% chance)
            if rng.gen_bool(0.05) {
                if let Some(_spout_loc) = bordering(grid, x, y, i, Element::Spout) {
                    grid.set_index(i, Element::Water);
                    return;
                }
            }
            
            // Steam may be trapped; disappear slowly (1% * 5% = 0.05% chance)
            if rng.gen_bool(0.01) && rng.gen_bool(0.05) {
                if below(grid, y, i, Element::Steam).is_none() {
                    grid.set_index(i, Element::Background);
                    return;
                }
            }
        }
        Element::SaltWater => {
            // Salt water falls with gravity (95% chance)
            // Can mix with water (50% chance each direction)
            if !do_density_liquid(grid, x, y, i, Element::Water, 0.50, 0.50) {
                do_gravity(grid, x, y, i, true, 0.95, fall_into_void, rainbow_sand_times);
            }
        }
        Element::Plant => {
            // Plant grows with water (50% chance)
            // But don't grow into water that is directly above soil (let soil handle that)
            let mut rng = rand::thread_rng();
            if rng.gen_bool(0.50) {
                if let Some(grow_loc) = bordering_adjacent(grid, x, y, i, Element::Water) {
                    // Check if this water is directly above soil - if so, don't convert it
                    // Calculate the y position of the water
                    let grow_y = (grow_loc / grid.width as usize) as u32;
                    
                    // Check if there's soil below this water
                    if grow_y < grid.max_y() {
                        let below_water_idx = grow_loc + grid.width as usize;
                        if below_water_idx < grid.elements.len() {
                            let below_elem = grid.get_index(below_water_idx);
                            if below_elem == Element::Soil || below_elem == Element::WetSoil {
                                // Don't convert water that's above soil - let soil create the plant
                                // Skip this growth
                            } else {
                                // Normal plant growth
                                let current_element = grid.get_index(i);
                                grid.set_index(grow_loc, current_element);
                                return;
                            }
                        } else {
                            // Normal plant growth
                            let current_element = grid.get_index(i);
                            grid.set_index(grow_loc, current_element);
                            return;
                        }
                    } else {
                        // Normal plant growth
                        let current_element = grid.get_index(i);
                        grid.set_index(grow_loc, current_element);
                        return;
                    }
                }
            }
            
            // Plant dies from salt (5% chance)
            if rng.gen_bool(0.05) {
                if let Some(_salt_loc) = bordering(grid, x, y, i, Element::Salt) {
                    grid.set_index(i, Element::Background);
                    return;
                }
            }
        }
        Element::Gunpowder => {
            // Gunpowder explodes when touched by fire (95% chance)
            let mut rng = rand::thread_rng();
            if rng.gen_bool(0.95) {
                if let Some(_fire_loc) = bordering(grid, x, y, i, Element::Fire) {
                    // Create explosion pattern - set surrounding pixels on fire
                    let burn = rng.gen_bool(0.60);
                    let replace = if burn { Element::Fire } else { Element::Gunpowder };
                    
                    // Set center
                    grid.set_index(i, replace);
                    
                    // Set 8 surrounding pixels
                    let positions = [
                        if y > 0 { Some(i.saturating_sub(grid.width as usize)) } else { None },
                        if y < grid.max_y() { Some(i + grid.width as usize) } else { None },
                        if x > 0 { Some(i - 1) } else { None },
                        if x < grid.max_x() { Some(i + 1) } else { None },
                        if y > 0 && x > 0 { Some(i.saturating_sub(grid.width as usize) - 1) } else { None },
                        if y > 0 && x < grid.max_x() { Some(i.saturating_sub(grid.width as usize) + 1) } else { None },
                        if y < grid.max_y() && x > 0 { Some(i + grid.width as usize - 1) } else { None },
                        if y < grid.max_y() && x < grid.max_x() { Some(i + grid.width as usize + 1) } else { None },
                    ];
                    
                    for pos_opt in positions.iter() {
                        if let Some(pos) = pos_opt {
                            if *pos < grid.elements.len() {
                                grid.set_index(*pos, replace);
                            }
                        }
                    }
                    
                    // Extended explosion (40% chance, 2 pixels away)
                    if burn && rng.gen_bool(0.40) {
                        let extended_positions = [
                            if y >= 2 { Some(i.saturating_sub(2 * grid.width as usize)) } else { None },
                            if y + 2 <= grid.max_y() { Some(i + 2 * grid.width as usize) } else { None },
                            if x >= 2 { Some(i - 2) } else { None },
                            if x + 2 <= grid.max_x() { Some(i + 2) } else { None },
                        ];
                        
                        for pos_opt in extended_positions.iter() {
                            if let Some(pos) = pos_opt {
                                if *pos < grid.elements.len() {
                                    let elem = grid.get_index(*pos);
                                    if elem != Element::Gunpowder || rng.gen_bool(0.50) {
                                        grid.set_index(*pos, Element::Fire);
                                    }
                                }
                            }
                        }
                    }
                    
                    return;
                }
            }
            
            // Gunpowder falls with gravity
            do_gravity(grid, x, y, i, true, 0.95, fall_into_void, rainbow_sand_times);
        }
        Element::Wax => {
            // Wax is static, but can burn and turn into falling wax
            let mut rng = rand::thread_rng();
            if rng.gen_bool(0.01) {
                if let Some(_fire_loc) = bordering(grid, x, y, i, Element::Fire) {
                    // Wax burns - turn to falling wax
                    grid.set_index(i, Element::FallingWax);
                    return;
                }
            }
        }
        Element::FallingWax => {
            // Falling wax falls with gravity (no diagonal), then turns back to wax
            if do_gravity(grid, x, y, i, false, 1.0, fall_into_void, rainbow_sand_times) {
                return;
            }
            // If it stopped falling, turn back to wax
            grid.set_index(i, Element::Wax);
        }
        Element::ChilledIce => {
            // Chilled ice thaws back to regular ice (6% chance)
            let mut rng = rand::thread_rng();
            if rng.gen_bool(0.06) {
                grid.set_index(i, Element::Ice);
                return;
            }
            
            // Thaw immediately if bordering salt, salt water, lava, fire, or steam
            if let Some(_) = bordering(grid, x, y, i, Element::Salt) {
                grid.set_index(i, Element::Ice);
                return;
            }
            if let Some(_) = bordering(grid, x, y, i, Element::SaltWater) {
                grid.set_index(i, Element::Ice);
                return;
            }
            if let Some(_) = bordering(grid, x, y, i, Element::Lava) {
                grid.set_index(i, Element::Ice);
                return;
            }
            if let Some(_) = bordering(grid, x, y, i, Element::Fire) {
                grid.set_index(i, Element::Ice);
                return;
            }
            if let Some(_) = bordering(grid, x, y, i, Element::Steam) {
                grid.set_index(i, Element::Ice);
                return;
            }
        }
        Element::Mystery => {
            // Mystery element - falls with gravity, special interactions
            // For now, simplified - just falls (particle effects would be added later)
            let mut rng = rand::thread_rng();
            
            // Reduce computation cost (50% chance to skip)
            if rng.gen_bool(0.50) {
                return;
            }
            
            // Check for sand - create multi-pronged star explosion (MAGIC1_PARTICLE effect)
            if let Some(_) = bordering_adjacent(grid, x, y, i, Element::Sand) {
                // Create radial explosion pattern (5-18 spokes)
                let num_spokes = 5 + rng.gen_range(0..=13);
                create_radial_explosion(grid, x, y, 10, num_spokes);
                grid.set_index(i, Element::Background);
                return;
            }
            // Check for salt - create spiral/circular explosion (MAGIC2_PARTICLE effect)
            if let Some(_) = bordering_adjacent(grid, x, y, i, Element::Salt) {
                // Create circular explosion pattern
                create_radial_explosion(grid, x, y, 15, 16);
                grid.set_index(i, Element::Background);
                return;
            }
            
            // Falls with gravity
            do_gravity(grid, x, y, i, true, 0.95, fall_into_void, rainbow_sand_times);
        }
        Element::ChargedNitro => {
            // Charged nitro - falls with gravity, sinks through lighter elements, explodes on fire
            if do_gravity(grid, x, y, i, true, 0.95, fall_into_void, rainbow_sand_times) {
                return;
            }
            
            // Sink through lighter elements
            if y < grid.max_y() {
                if do_density_sink(grid, x, y, i, Element::Soil, true, 0.25, fall_into_void, rainbow_sand_times) {
                    return;
                }
                if do_density_sink(grid, x, y, i, Element::WetSoil, true, 0.25, fall_into_void, rainbow_sand_times) {
                    return;
                }
                if do_density_sink(grid, x, y, i, Element::Nitro, true, 0.25, fall_into_void, rainbow_sand_times) {
                    return;
                }
                if do_density_sink(grid, x, y, i, Element::Pollen, true, 0.25, fall_into_void, rainbow_sand_times) {
                    return;
                }
            }
            
            // Explode when touching fire - create vertical fire column (CHARGED_NITRO_PARTICLE effect)
            if let Some(_) = bordering_adjacent(grid, x, y, i, Element::Fire) {
                // Create CHARGED_NITRO_PARTICLE (matches TypeScript: particles.addActiveParticle(CHARGED_NITRO_PARTICLE, x, y, i))
                if let Some(plist) = particle_list {
                    plist.add_active_particle(
                        crate::particles::ParticleType::ChargedNitro,
                        x as f32,
                        y as f32,
                        i,
                    );
                }
                grid.set_index(i, Element::Fire);
                return;
            }
        }
        Element::BurningThermite => {
            // Burning thermite - burns adjacent elements, can create charged nitro, consumes itself, burns through walls
            let mut rng = rand::thread_rng();
            
            // Burn adjacent elements (up, left, right) - except thermite, burning thermite, lava, wall
            if y > 0 {
                let above_idx = i.saturating_sub(grid.width as usize);
                let elem = grid.get_index(above_idx);
                if elem != Element::Thermite && elem != Element::BurningThermite && elem != Element::Lava && elem != Element::Wall {
                    grid.set_index(above_idx, Element::Fire);
                }
            }
            if x > 0 {
                let left_idx = i - 1;
                let elem = grid.get_index(left_idx);
                if elem != Element::Thermite && elem != Element::BurningThermite && elem != Element::Lava && elem != Element::Wall {
                    grid.set_index(left_idx, Element::Fire);
                }
            }
            if x < grid.max_x() {
                let right_idx = i + 1;
                if right_idx < grid.elements.len() {
                    let elem = grid.get_index(right_idx);
                    if elem != Element::Thermite && elem != Element::BurningThermite && elem != Element::Lava && elem != Element::Wall {
                        grid.set_index(right_idx, Element::Fire);
                    }
                }
            }
            
            // Chance to create charged nitro explosion (2% * 7% = 0.14% chance)
            if rng.gen_bool(0.02) && rng.gen_bool(0.07) {
                // Create CHARGED_NITRO_PARTICLE (matches TypeScript: particles.addActiveParticle(CHARGED_NITRO_PARTICLE, x, y, i))
                if let Some(plist) = particle_list {
                    plist.add_active_particle(
                        crate::particles::ParticleType::ChargedNitro,
                        x as f32,
                        y as f32,
                        i,
                    );
                }
                grid.set_index(i, Element::Fire);
                return;
            }
            
            // Chance to consume itself (2% chance)
            if rng.gen_bool(0.02) {
                grid.set_index(i, Element::Fire);
                return;
            }
            
            // Burn through walls (8% chance)
            if rng.gen_bool(0.08) {
                // Check adjacent walls
                if let Some(wall_loc) = adjacent(grid, x, i, Element::Wall) {
                    grid.set_index(wall_loc, Element::Background);
                }
                if let Some(wall_loc) = below(grid, y, i, Element::Wall) {
                    grid.set_index(wall_loc, Element::Background);
                }
            }
            
            // Clear fire below (to allow falling through)
            if let Some(fire_loc) = below(grid, y, i, Element::Fire) {
                grid.set_index(fire_loc, Element::Background);
            }
            
            // Falls with gravity
            if do_gravity(grid, x, y, i, false, 0.99, fall_into_void, rainbow_sand_times) {
                return;
            }
            
            // Sink through liquids
            if y < grid.max_y() {
                if do_density_sink(grid, x, y, i, Element::Water, false, 0.95, fall_into_void, rainbow_sand_times) {
                    return;
                }
                if do_density_sink(grid, x, y, i, Element::SaltWater, false, 0.95, fall_into_void, rainbow_sand_times) {
                    return;
                }
                if do_density_sink(grid, x, y, i, Element::Oil, false, 0.95, fall_into_void, rainbow_sand_times) {
                    return;
                }
            }
        }
        Element::Concrete => {
            // Concrete can sink through water and salt water
            if y < grid.max_y() {
                if do_density_sink(grid, x, y, i, Element::Water, true, 0.35, fall_into_void, rainbow_sand_times) {
                    return;
                }
                if do_density_sink(grid, x, y, i, Element::SaltWater, true, 0.35, fall_into_void, rainbow_sand_times) {
                    return;
                }
            }
            
            // Concrete hardens to wall when next to wall (10% * 10% = 1% chance)
            let mut rng = rand::thread_rng();
            if rng.gen_bool(0.10) && rng.gen_bool(0.10) {
                if let Some(_wall_loc) = bordering_adjacent(grid, x, y, i, Element::Wall) {
                    grid.set_index(i, Element::Wall);
                    return;
                }
            }
            
            // Concrete falls with gravity
            if do_gravity(grid, x, y, i, true, 0.95, fall_into_void, rainbow_sand_times) {
                return;
            }
            
            // Concrete can harden on its own (10% * 10% * 5% = 0.05% chance)
            if rng.gen_bool(0.10) && rng.gen_bool(0.10) && rng.gen_bool(0.05) {
                grid.set_index(i, Element::Wall);
            }
        }
        Element::Nitro => {
            // Nitro falls with gravity
            if do_gravity(grid, x, y, i, true, 0.95, fall_into_void, rainbow_sand_times) {
                return;
            }
            
            // Optimize: skip if surrounded by nitro
            if surrounded_by(grid, x, y, i, Element::Nitro) {
                return;
            }
            
            // Nitro explodes when touched by fire (30% chance)
            let mut rng = rand::thread_rng();
            if let Some(_fire_loc) = bordering_adjacent(grid, x, y, i, Element::Fire) {
                if rng.gen_bool(0.30) {
                    // Create border burn (set surrounding pixels on fire)
                    if y > 0 {
                        let above_idx = i.saturating_sub(grid.width as usize);
                        if grid.get_index(above_idx) == Element::Background {
                            grid.set_index(above_idx, Element::Fire);
                        }
                    }
                    if y < grid.max_y() {
                        let below_idx = i + grid.width as usize;
                        if below_idx < grid.elements.len() && grid.get_index(below_idx) == Element::Background {
                            grid.set_index(below_idx, Element::Fire);
                        }
                    }
                    if x > 0 {
                        let left_idx = i - 1;
                        if grid.get_index(left_idx) == Element::Background {
                            grid.set_index(left_idx, Element::Fire);
                        }
                    }
                    if x < grid.max_x() {
                        let right_idx = i + 1;
                        if right_idx < grid.elements.len() && grid.get_index(right_idx) == Element::Background {
                            grid.set_index(right_idx, Element::Fire);
                        }
                    }
                    grid.set_index(i, Element::Fire);
                    return;
                } else if rng.gen_bool(0.20) {
                    grid.set_index(i, Element::Fire);
                    return;
                }
            }
            
            // Nitro sinks through lighter liquids and pollen
            if y < grid.max_y() {
                if do_density_sink(grid, x, y, i, Element::Oil, true, 0.25, fall_into_void, rainbow_sand_times) {
                    return;
                }
                if do_density_sink(grid, x, y, i, Element::Water, true, 0.25, fall_into_void, rainbow_sand_times) {
                    return;
                }
                if do_density_sink(grid, x, y, i, Element::SaltWater, true, 0.25, fall_into_void, rainbow_sand_times) {
                    return;
                }
                if do_density_sink(grid, x, y, i, Element::Pollen, true, 0.25, fall_into_void, rainbow_sand_times) {
                    return;
                }
            }
        }
        Element::Napalm => {
            // Napalm catches fire (25% chance) - create spreading fire particles (NAPALM_PARTICLE effect)
            let mut rng = rand::thread_rng();
            if rng.gen_bool(0.25) {
                if let Some(_fire_loc) = bordering(grid, x, y, i, Element::Fire) {
                    // Create NAPALM_PARTICLE (matches TypeScript: particles.addActiveParticle(NAPALM_PARTICLE, x, y, i))
                    if let Some(plist) = particle_list {
                        if plist.add_active_particle(
                            crate::particles::ParticleType::Napalm,
                            x as f32,
                            y as f32,
                            i,
                        ).is_some() {
                            grid.set_index(i, Element::Fire);
                            return;
                        }
                    }
                    // Fallback if particle creation fails
                    grid.set_index(i, Element::Fire);
                    return;
                }
            }
            
            // Napalm falls with gravity
            do_gravity(grid, x, y, i, true, 0.95, fall_into_void, rainbow_sand_times);
        }
        Element::C4 => {
            // C4 explodes when touched by fire (60% chance) - create large expanding explosion (C4_PARTICLE effect)
            let mut rng = rand::thread_rng();
            if rng.gen_bool(0.60) {
                if let Some(_fire_loc) = bordering(grid, x, y, i, Element::Fire) {
                    // Create C4_PARTICLE (matches TypeScript: particles.addActiveParticle(C4_PARTICLE, x, y, i))
                    if let Some(plist) = particle_list {
                        if plist.add_active_particle(
                            crate::particles::ParticleType::C4,
                            x as f32,
                            y as f32,
                            i,
                        ).is_some() {
                            grid.set_index(i, Element::Fire);
                            return;
                        }
                    }
                    // Fallback if particle creation fails
                    grid.set_index(i, Element::Fire);
                    return;
                }
            }
            // C4 is static (doesn't fall)
        }
        Element::Fuse => {
            // Fuse is static (doesn't fall)
            // Fire spreads to it (handled in fire action)
        }
        Element::Acid => {
            // Acid dissolves bordering elements (10% chance)
            let mut rng = rand::thread_rng();
            if rng.gen_bool(0.10) {
                // Check up, down, left, right (not corners)
                let positions = [
                    if y > 0 { Some(i.saturating_sub(grid.width as usize)) } else { None },
                    if y < grid.max_y() { Some(i + grid.width as usize) } else { None },
                    if x > 0 { Some(i - 1) } else { None },
                    if x < grid.max_x() { Some(i + 1) } else { None },
                ];
                
                // Randomize order to avoid bias
                let mut shuffled_positions = positions;
                if rng.gen_bool(0.5) {
                    shuffled_positions.swap(0, 1);
                }
                if rng.gen_bool(0.5) {
                    shuffled_positions.swap(2, 3);
                }
                
                for pos_opt in shuffled_positions.iter() {
                    if let Some(pos) = pos_opt {
                        if *pos < grid.elements.len() {
                            let elem = grid.get_index(*pos);
                            // Acid immune elements: Acid, Background, Water, SaltWater, Ice, Steam
                            let can_dissolve = !matches!(
                                elem,
                                Element::Acid | Element::Background | Element::Water
                                    | Element::SaltWater | Element::Ice | Element::ChilledIce | Element::Steam | Element::Cryo
                            );
                            
                            if can_dissolve {
                                // If dissolving something above or to the side, just remove it
                                if *pos != i + grid.width as usize {
                                    grid.set_index(*pos, Element::Background);
                                    return;
                                } else {
                                    // If dissolving something below, move acid down (75% chance for wall)
                                    grid.set_index(i, Element::Background);
                                    if elem != Element::Wall || rng.gen_bool(0.75) {
                                        grid.set_index(*pos, Element::Acid);
                                    }
                                    return;
                                }
                            }
                        }
                    }
                }
            }
            
            // Acid can mix with water/salt water
            if do_density_liquid(grid, x, y, i, Element::Water, 0.25, 0.30) {
                return;
            }
            if do_density_liquid(grid, x, y, i, Element::SaltWater, 0.25, 0.30) {
                return;
            }
            
            // Acid falls with gravity (100% chance)
            do_gravity(grid, x, y, i, true, 1.0, fall_into_void, rainbow_sand_times);
        }
        Element::Cryo => {
            // Cryo freezes things and falls with gravity
            let mut rng = rand::thread_rng();
            
            // Freeze surrounding surfaces
            let x_start = x.saturating_sub(1);
            let y_start = y.saturating_sub(1);
            let x_end = (x + 2).min(grid.max_x() + 1);
            let y_end = (y + 2).min(grid.max_y() + 1);
            
            for y_iter in y_start..y_end {
                for x_iter in x_start..x_end {
                    if y_iter == y && x_iter == x {
                        continue;
                    }
                    
                    let idx = grid.xy_to_index(x_iter, y_iter);
                    if idx >= grid.elements.len() {
                        continue;
                    }
                    
                    let elem = grid.get_index(idx);
                    
                    // Freeze water to ice
                    if elem == Element::Water {
                        grid.set_index(idx, Element::Ice);
                        grid.set_index(i, Element::Ice);
                        return;
                    }
                    
                    // Freeze ice - can create chilled ice (1% * 5% = 0.05% chance)
                    if elem == Element::Ice {
                        if rng.gen_bool(0.01) && rng.gen_bool(0.05) {
                            grid.set_index(idx, Element::ChilledIce);
                            grid.set_index(i, Element::ChilledIce);
                        } else {
                            grid.set_index(idx, Element::Ice);
                            grid.set_index(i, Element::Ice);
                        }
                        return;
                    }
                    
                    // Freeze certain elements (simplified list)
                    if matches!(elem, Element::Wall | Element::Wax | Element::Plant | Element::C4) {
                        grid.set_index(i, Element::Ice);
                        return;
                    }
                    
                    // Cryo + Lava = Rock
                    if elem == Element::Lava {
                        grid.set_index(i, Element::Background);
                        grid.set_index(idx, Element::Rock);
                        return;
                    }
                }
            }
            
            // Cryo falls with gravity
            do_gravity(grid, x, y, i, true, 0.95, fall_into_void, rainbow_sand_times);
            
            // Can freeze even if no nearby freezable surfaces (1% * 50% = 0.5% chance)
            if rng.gen_bool(0.01) && rng.gen_bool(0.50) {
                if bordering(grid, x, y, i, Element::Background).is_none() && !surrounded_by(grid, x, y, i, Element::Cryo) {
                    grid.set_index(i, Element::Ice);
                }
            }
        }
        Element::Methane => {
            // Methane is a flammable gas that rises
            let mut rng = rand::thread_rng();
            
            // Check if there's a methane particle nearby (for chain reaction spreading)
            // This allows fire to propagate through methane clouds
            // Also check if methane touches fire (25% chance)
            let mut should_create_particle = false;
            if let Some(plist) = particle_list {
                let check_radius = 8.0; // Reduced from 15 to 8 pixels for slower propagation
                let active_indices = plist.active_particles();
                for &particle_idx in active_indices {
                    if let Some(particle) = plist.get_particle(particle_idx) {
                        if particle.particle_type == crate::particles::ParticleType::Methane {
                            let dx = particle.x - x as f32;
                            let dy = particle.y - y as f32;
                            let dist_sq = dx * dx + dy * dy;
                            if dist_sq <= check_radius * check_radius {
                                // Add probability to slow down propagation (50% chance)
                                if rng.gen_bool(0.5) {
                                    should_create_particle = true;
                                    break;
                                }
                            }
                        }
                    }
                }
                
                // Also check if methane touches fire (25% chance)
                if !should_create_particle && rng.gen_bool(0.25) {
                    if let Some(_fire_loc) = bordering(grid, x, y, i, Element::Fire) {
                        should_create_particle = true;
                    }
                }
                
                // Create particle if needed (either from nearby particle or from fire contact)
                if should_create_particle {
                    if plist.add_active_particle(
                        crate::particles::ParticleType::Methane,
                        x as f32,
                        y as f32,
                        i,
                    ).is_some() {
                        grid.set_index(i, Element::Fire);
                        return;
                    }
                    // Fallback if particle creation fails
                    grid.set_index(i, Element::Fire);
                    return;
                }
            } else {
                // No particle_list available, fall back to simple fire conversion
                if rng.gen_bool(0.25) {
                    if let Some(_fire_loc) = bordering(grid, x, y, i, Element::Fire) {
                        grid.set_index(i, Element::Fire);
                        return;
                    }
                }
            }
            
            // Methane rises (25% chance, 65% adjacent)
            if do_rise(grid, x, y, i, 0.25, 0.65, fall_into_void) {
                return;
            }
            
            // Methane can pass through gas-permeable elements (70% chance)
            // Simplified: just check for common permeable elements
            if rng.gen_bool(0.70) {
                if y > 0 {
                    let above_idx = i.saturating_sub(grid.width as usize);
                    let above_elem = grid.get_index(above_idx);
                    // Gas permeable: Sand, Water, Salt, SaltWater, Oil, Gunpowder, Concrete, Rock
                    if matches!(
                        above_elem,
                        Element::Sand | Element::Water | Element::Salt | Element::SaltWater
                            | Element::Oil | Element::Gunpowder | Element::Concrete | Element::Rock
                    ) {
                        grid.set_index(above_idx, Element::Methane);
                        grid.set_index(i, above_elem);
                        return;
                    }
                }
            }
        }
        Element::Soil => {
            // Soil falls with gravity (no diagonal)
            if do_gravity(grid, x, y, i, false, 0.99, fall_into_void, rainbow_sand_times) {
                return;
            }
            
            // Soil can sink through lighter elements
            if y < grid.max_y() {
                if do_density_sink(grid, x, y, i, Element::Water, true, 0.50, fall_into_void, rainbow_sand_times) {
                    return;
                }
                if do_density_sink(grid, x, y, i, Element::SaltWater, true, 0.50, fall_into_void, rainbow_sand_times) {
                    return;
                }
                if do_density_sink(grid, x, y, i, Element::Pollen, true, 0.50, fall_into_void, rainbow_sand_times) {
                    return;
                }
            }
            
            // Soil transforms nitro to charged nitro (25% chance, 100% of the time)
            let mut rng = rand::thread_rng();
            if rng.gen_bool(0.25) {
                if let Some(nitro_loc) = bordering_adjacent(grid, x, y, i, Element::Nitro) {
                    grid.set_index(nitro_loc, Element::ChargedNitro);
                    return;
                }
            }
            
            // Soil absorbs water above (15% chance) to become wet soil
            // Just convert soil to wet soil, no tree creation here (trees come from wet soil later)
            if rng.gen_bool(0.15) {
                if let Some(water_loc) = above_adjacent(grid, x, y, i, Element::Water) {
                    grid.set_index(water_loc, Element::Background);
                    grid.set_index(i, Element::WetSoil);
                    return;
                }
            }
        }
        Element::WetSoil => {
            // Wet soil can absorb more water (15% chance)
            let mut rng = rand::thread_rng();
            if rng.gen_bool(0.15) {
                if let Some(water_loc) = above_adjacent(grid, x, y, i, Element::Water) {
                    grid.set_index(water_loc, Element::Background);
                }
            }
            
            // Wet soil falls with gravity (no diagonal)
            if do_gravity(grid, x, y, i, false, 0.99, fall_into_void, rainbow_sand_times) {
                return;
            }
            
            // Wet soil can sink through lighter elements
            if do_density_sink(grid, x, y, i, Element::Water, true, 0.50, fall_into_void, rainbow_sand_times) {
                return;
            }
            if do_density_sink(grid, x, y, i, Element::SaltWater, true, 0.50, fall_into_void, rainbow_sand_times) {
                return;
            }
            
            // Wet soil can generate trees or dry to soil
            // In TypeScript: if (random() < 5) { if (random() < 97) { dry to soil } else { try tree } }
            // Tree generation: 5% chance, then 3% of that time (not 97%), then 65% after that
            if rng.gen_bool(0.05) {
                if rng.gen_bool(0.97) {
                    // 97% of the time: dry to soil (if no water adjacent)
                    if bordering_adjacent(grid, x, y, i, Element::Water).is_none() {
                        grid.set_index(i, Element::Soil);
                        return;
                    }
                } else {
                    // 3% of the time: try tree generation
                    // Make tree generation less likely (35% chance to skip)
                    if rng.gen_bool(0.35) {
                        return; // Skip tree generation
                    }
                    
                    // 65% of the time: generate tree
                    // Check conditions: space above (any of the 3 positions), and soil or wall below (any of the 3 positions)
                    // TypeScript: aboveAdjacent checks directly above, above-left, above-right
                    // TypeScript: belowAdjacent checks directly below, below-left, below-right
                    if let Some(_) = above_adjacent(grid, x, y, i, Element::Background) {
                        let below_soil = below_adjacent(grid, x, y, i, Element::Soil);
                        let below_wall = below_adjacent(grid, x, y, i, Element::Wall);
                        if below_soil.is_some() || below_wall.is_some() {
                            // Start tree generation using grid-based approach
                            if let Some(active_branches) = active_branches {
                                start_tree_generation(active_branches, x, y);
                                grid.set_index(i, Element::Soil);
                                return;
                            }
                        }
                    }
                }
            }
        }
        Element::Thermite => {
            // Thermite falls and burns through walls
            // Skip if surrounded by thermite (optimization - check all 8 directions)
            if surrounded_by_adjacent(grid, x, y, i, Element::Thermite) {
                return;
            }
            
            // Thermite turns to burning thermite when near fire (50% chance)
            let mut rng = rand::thread_rng();
            if rng.gen_bool(0.50) {
                if let Some(_fire_loc) = bordering_adjacent(grid, x, y, i, Element::Fire) {
                    // Use the BurningThermite element we already have
                    grid.set_index(i, Element::BurningThermite);
                    return;
                }
            }
            
            // Thermite sinks through liquids
            if do_density_sink(grid, x, y, i, Element::Water, false, 0.95, fall_into_void, rainbow_sand_times) {
                return;
            }
            if do_density_sink(grid, x, y, i, Element::SaltWater, false, 0.95, fall_into_void, rainbow_sand_times) {
                return;
            }
            if do_density_sink(grid, x, y, i, Element::Oil, false, 0.95, fall_into_void, rainbow_sand_times) {
                return;
            }
            
            // Thermite falls with gravity (no diagonal, 99% chance)
            do_gravity(grid, x, y, i, false, 0.99, fall_into_void, rainbow_sand_times);
        }
        Element::Spout => {
            // Spout produces water (5% chance, doesn't overwrite)
            do_producer(grid, x, y, i, Element::Water, false, 0.05);
        }
        Element::Well => {
            // Well produces oil (10% chance, doesn't overwrite)
            do_producer(grid, x, y, i, Element::Oil, false, 0.10);
        }
        Element::Torch => {
            // Torch produces fire (25% chance, overwrites adjacent)
            do_producer(grid, x, y, i, Element::Fire, true, 0.25);
        }
        Element::Branch => {
            // Branch is static, burns when touched by fire (3% chance)
            let mut rng = rand::thread_rng();
            if rng.gen_bool(0.03) {
                if let Some(_fire_loc) = bordering_adjacent(grid, x, y, i, Element::Fire) {
                    grid.set_index(i, Element::Fire);
                    return;
                }
            }
        }
        Element::Leaf => {
            // Leaf is static, burns when touched by fire (5% chance)
            let mut rng = rand::thread_rng();
            if rng.gen_bool(0.05) {
                if let Some(_fire_loc) = bordering_adjacent(grid, x, y, i, Element::Fire) {
                    grid.set_index(i, Element::Fire);
                    return;
                }
            }
            
            // Leaf dies from salt (20% chance)
            if rng.gen_bool(0.20) {
                if let Some(_salt_loc) = bordering_adjacent(grid, x, y, i, Element::Salt) {
                    grid.set_index(i, Element::Background);
                    return;
                }
            }
            
            // Leaf produces pollen (1% * 9% = 0.09% chance)
            if rng.gen_bool(0.01) && rng.gen_bool(0.09) {
                do_producer(grid, x, y, i, Element::Pollen, false, 1.0);
            }
        }
        Element::Pollen => {
            // Pollen falls with gravity
            do_gravity(grid, x, y, i, true, 0.95, fall_into_void, rainbow_sand_times);
        }
        Element::RainbowSand => {
            // RainbowSand behaves like sand - can sink through liquids and falls with gravity
            if y < grid.max_y() {
                if do_density_sink(grid, x, y, i, Element::Water, true, 0.25, fall_into_void, rainbow_sand_times) {
                    return;
                }
                if do_density_sink(grid, x, y, i, Element::SaltWater, true, 0.25, fall_into_void, rainbow_sand_times) {
                    return;
                }
            }
            // RainbowSand falls with gravity, can fall diagonally (fall_adjacent = true)
            do_gravity(grid, x, y, i, true, 0.95, fall_into_void, rainbow_sand_times);
        }
    }
}

