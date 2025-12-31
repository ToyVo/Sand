use crate::particles::types::{Particle, ParticleType, MAGIC_COLORS};
use crate::simulation::GameGrid;
use crate::SIZE;
use crate::elements::Element;
use rand::Rng;

/// Initialize a particle based on its type
pub fn particle_init(particle: &mut Particle, grid: &GameGrid) {
    let mut rng = rand::thread_rng();
    
    match particle.particle_type {
        ParticleType::Unknown => {
            // Unknown particles shouldn't be initialized
        }
        ParticleType::Nitro => {
            nitro_particle_init(particle, &mut rng);
        }
        ParticleType::Napalm => {
            napalm_particle_init(particle, &mut rng);
        }
        ParticleType::C4 => {
            c4_particle_init(particle, &mut rng);
        }
        ParticleType::Lava => {
            lava_particle_init(particle, &mut rng);
        }
        ParticleType::Magic1 => {
            magic1_particle_init(particle, &mut rng, grid);
        }
        ParticleType::Magic2 => {
            magic2_particle_init(particle, &mut rng);
        }
        ParticleType::Methane => {
            methane_particle_init(particle, &mut rng);
        }
        ParticleType::Tree => {
            tree_particle_init(particle, &mut rng);
        }
        ParticleType::ChargedNitro => {
            charged_nitro_particle_init(particle, grid);
        }
        ParticleType::Nuke => {
            nuke_particle_init(particle, &mut rng);
        }
    }
}

/// Update a particle each frame
/// Returns true if particle should be removed
/// particle_list is only needed for tree particles (to create branches)
pub fn particle_action(
    particle: &mut Particle,
    particle_list: Option<&mut crate::particles::manager::ParticleList>,
    particle_idx: usize,
    grid: &GameGrid,
) -> bool {
    particle.action_iterations += 1;
    
    match particle.particle_type {
        ParticleType::Unknown => {
            return true; // Remove unknown particles
        }
        ParticleType::Nitro => {
            return nitro_particle_action(particle, grid);
        }
        ParticleType::Napalm => {
            return napalm_particle_action(particle);
        }
        ParticleType::C4 => {
            return c4_particle_action(particle);
        }
        ParticleType::Lava => {
            return lava_particle_action(particle);
        }
        ParticleType::Magic1 => {
            // Magic1 particles - simplified for now (full version would create spokes)
            return magic1_particle_action(particle, None, particle_idx, grid);
        }
        ParticleType::Magic2 => {
            return magic2_particle_action(particle, grid);
        }
        ParticleType::Methane => {
            return methane_particle_action(particle, grid, particle_list);
        }
        ParticleType::Tree => {
            if let Some(plist) = particle_list {
                return tree_particle_action(particle, plist, particle_idx, grid);
            }
            // Tree particle without particle_list - can't create branches, just move
            particle.x += particle.x_velocity;
            particle.y += particle.y_velocity;
            return false;
        }
        ParticleType::ChargedNitro => {
            return charged_nitro_particle_action(particle, grid);
        }
        ParticleType::Nuke => {
            return nuke_particle_action(particle);
        }
    }
}

// NITRO_PARTICLE
fn nitro_particle_init(particle: &mut Particle, rng: &mut impl Rng) {
    particle.set_color(crate::elements::Element::Fire);
    
    let velocity = 5.0 + rng.gen_range(0.0..1.0) * 10.0;
    let angle = rng.gen_range(0.0..1.0) * 2.0 * std::f32::consts::PI;
    particle.set_velocity(velocity, angle);
    
    particle.size = 2.0 + rng.gen_range(0.0..1.0) * 7.0;
}

fn nitro_particle_action(particle: &mut Particle, grid: &GameGrid) -> bool {
    // Move particle
    particle.x += particle.x_velocity;
    particle.y += particle.y_velocity;
    
    // Shrink over time
    if particle.action_iterations % 5 == 0 {
        particle.size /= 1.3;
    }
    
    // Accelerate downward
    if particle.action_iterations % 15 == 0 {
        particle.y_velocity += 10.0 * (particle.action_iterations as f32 / 5.0);
    }
    
    // Remove if too small or off canvas
    if particle.size < 1.75 {
        return true;
    }
    if particle.off_canvas(grid.width as f32, grid.height as f32) {
        return true;
    }
    
    false
}

// NAPALM_PARTICLE
fn napalm_particle_init(particle: &mut Particle, rng: &mut impl Rng) {
    particle.set_color(crate::elements::Element::Fire);
    particle.size = rng.gen_range(0.0..1.0) * 8.0 + 6.0;
    particle.x_velocity = rng.gen_range(0.0..1.0) * 8.0 - 4.0;
    particle.y_velocity = -(rng.gen_range(0.0..1.0) * 4.0 + 4.0);
    particle.max_iterations = Some(rng.gen_range(5..=15));
}

fn napalm_particle_action(particle: &mut Particle) -> bool {
    // Move particle
    particle.x += particle.x_velocity;
    particle.y += particle.y_velocity;
    
    // Grow over time
    particle.size *= 1.0 + rand::thread_rng().gen_range(0.0..1.0) * 0.1;
    
    // Remove after max iterations
    if let Some(max_iter) = particle.max_iterations {
        if particle.action_iterations > max_iter {
            return true;
        }
    }
    
    false
}

// C4_PARTICLE
fn c4_particle_init(particle: &mut Particle, rng: &mut impl Rng) {
    particle.set_color(crate::elements::Element::Fire);
    let rand = rng.gen_range(0.0..1.0) * 10000.0;
    if rand < 9000.0 {
        particle.size = rng.gen_range(0.0..1.0) * 10.0 + 3.0;
    } else if rand < 9500.0 {
        particle.size = rng.gen_range(0.0..1.0) * 32.0 + 3.0;
    } else if rand < 9800.0 {
        particle.size = rng.gen_range(0.0..1.0) * 64.0 + 3.0;
    } else {
        particle.size = rng.gen_range(0.0..1.0) * 128.0 + 3.0;
    }
}

fn c4_particle_action(particle: &mut Particle) -> bool {
    // Shrink over time
    if particle.action_iterations % 3 == 0 {
        particle.size /= 3.0;
        if particle.size <= 1.0 {
            return true;
        }
    }
    
    false
}

// LAVA_PARTICLE
fn lava_particle_init(particle: &mut Particle, rng: &mut impl Rng) {
    particle.set_color(crate::elements::Element::Fire);
    
    // Make it harder for the angle to be steep
    let mut angle = std::f32::consts::PI / 4.0 + rng.gen_range(0.0..1.0) * std::f32::consts::PI / 2.0;
    if rng.gen_bool(0.75) && (std::f32::consts::PI / 2.0 - angle).abs() < std::f32::consts::PI / 18.0 {
        angle += std::f32::consts::PI / 18.0 * if angle > std::f32::consts::PI / 2.0 { 1.0 } else { -1.0 };
    }
    
    particle.x_velocity = (1.0 + rng.gen_range(0.0..1.0) * 3.0) * angle.cos();
    particle.y_velocity = (-4.0 * rng.gen_range(0.0..1.0) - 3.0) * angle.sin();
    particle.init_y_velocity = Some(particle.y_velocity);
    particle.y_acceleration = Some(0.06);
    
    particle.size = 4.0 + rng.gen_range(0.0..1.0) * 3.0;
    particle.y -= particle.size;
}

fn lava_particle_action(particle: &mut Particle) -> bool {
    // Move with acceleration
    particle.x += particle.x_velocity;
    if let (Some(init_y_vel), Some(y_accel)) = (particle.init_y_velocity, particle.y_acceleration) {
        let iterations = particle.action_iterations as f32;
        particle.y = particle.init_y + init_y_vel * iterations + (y_accel * iterations * iterations) / 2.0;
    } else {
        particle.y += particle.y_velocity;
    }
    
    // Check for collisions (simplified - would check grid in full version)
    if particle.off_canvas(SIZE.x as f32, SIZE.y as f32) {
        return true;
    }
    
    false
}

// MAGIC1_PARTICLE (multi-pronged star)
fn magic1_particle_init(particle: &mut Particle, rng: &mut impl Rng, _grid: &GameGrid) {
    // Set random color from magic colors
    let color_idx = rng.gen_range(0..MAGIC_COLORS.len());
    particle.set_color(MAGIC_COLORS[color_idx]);
    
    let num_spokes = 5 + rng.gen_range(0..=13);
    // Note: In full version, would create multiple particles for each spoke
    // For now, we'll create a single particle that represents one spoke
    
    let _angle = 2.0 * std::f32::consts::PI / num_spokes as f32;
    let velocity = 7.0 + rng.gen_range(0.0..1.0) * 3.0;
    let spoke_size = 4.0 + rng.gen_range(0.0..1.0) * 4.0;
    
    // For simplicity, create one spoke - in full version would create all spokes
    particle.set_velocity(velocity, 0.0); // Start at angle 0, caller can adjust
    particle.size = spoke_size;
}

fn magic1_particle_action(
    particle: &mut Particle,
    _particle_list: Option<&mut crate::particles::manager::ParticleList>,
    _particle_idx: usize,
    grid: &GameGrid,
) -> bool {
    // Move particle
    particle.x += particle.x_velocity;
    particle.y += particle.y_velocity;
    
    // Remove if off canvas
    if particle.off_canvas(grid.width as f32, grid.height as f32) {
        return true;
    }
    
    false
}

// MAGIC2_PARTICLE (spiral)
fn magic2_particle_init(particle: &mut Particle, rng: &mut impl Rng) {
    // Set random color from magic colors
    let color_idx = rng.gen_range(0..MAGIC_COLORS.len());
    particle.set_color(MAGIC_COLORS[color_idx]);
    
    particle.size = 4.0 + rng.gen_range(0.0..1.0) * 8.0;
    particle.x = SIZE.x as f32 / 2.0;
    particle.y = SIZE.y as f32 / 2.0;
    particle.init_x = particle.x;
    particle.init_y = particle.y;
    
    let max_dimension = SIZE.x.max(SIZE.y) as f32;
    particle.magic_2_max_radius = Some((max_dimension * max_dimension + max_dimension * max_dimension).sqrt() / 2.0 + particle.size);
    particle.magic_2_theta = Some(0.0);
    particle.magic_2_speed = Some(20.0);
    particle.magic_2_radius_spacing = Some(25.0 + rng.gen_range(0.0..1.0) * 55.0);
    particle.magic_2_radius = Some(particle.magic_2_radius_spacing.unwrap());
}

fn magic2_particle_action(particle: &mut Particle, grid: &GameGrid) -> bool {
    if let (Some(theta), Some(speed), Some(radius_spacing)) = (
        particle.magic_2_theta,
        particle.magic_2_speed,
        particle.magic_2_radius_spacing,
    ) {
        let new_theta = theta + speed / particle.magic_2_radius.unwrap_or(1.0);
        particle.magic_2_theta = Some(new_theta);
        
        let new_radius = (new_theta / (2.0 * std::f32::consts::PI)) * radius_spacing;
        particle.magic_2_radius = Some(new_radius);
        
        particle.x = new_radius * new_theta.cos() + particle.init_x;
        particle.y = new_radius * new_theta.sin() + particle.init_y;
        
        if let Some(max_radius) = particle.magic_2_max_radius {
            if new_radius > max_radius {
                return true;
            }
        }
    }
    
    if particle.off_canvas(grid.width as f32, grid.height as f32) {
        return true;
    }
    
    false
}

// METHANE_PARTICLE
fn methane_particle_init(particle: &mut Particle, rng: &mut impl Rng) {
    particle.set_color(crate::elements::Element::Fire);
    particle.size = 10.0 + rng.gen_range(0.0..1.0) * 10.0;
}

fn methane_particle_action(
    particle: &mut Particle,
    _grid: &GameGrid,
    _particle_list: Option<&mut crate::particles::manager::ParticleList>,
) -> bool {
    // Remove after 2 iterations (matches TypeScript)
    // Note: Fire spreading to adjacent methane is handled in the methane element action
    // by checking for nearby methane particles
    if particle.action_iterations > 2 {
        return true;
    }
    
    false
}

// TREE_PARTICLE
fn tree_particle_init(particle: &mut Particle, rng: &mut impl Rng) {
    particle.set_color(Element::Branch);
    particle.size = if rng.gen_bool(0.5) { 3.0 } else { 4.0 };
    
    let velocity = 1.0 + rng.gen_range(0.0..1.0) * 0.5;
    // Angle: -HALF_PI - EIGHTH_PI + random * QUARTER_PI
    // This makes trees grow upward with slight variation
    let angle = -std::f32::consts::PI / 2.0 - std::f32::consts::PI / 8.0 + rng.gen_range(0.0..1.0) * std::f32::consts::PI / 4.0;
    particle.set_velocity(velocity, angle);
    
    particle.tree_generation = Some(1);
    particle.tree_branch_spacing = Some(15 + rng.gen_range(0..=45));
    particle.tree_max_branches = Some(1 + rng.gen_range(0..=2));
    particle.tree_next_branch = particle.tree_branch_spacing;
    particle.tree_branches = Some(0);
    
    // Make it more likely to be a standard tree (Tree0)
    if rng.gen_bool(0.62) {
        particle.tree_type = Some(0);
    } else {
        particle.tree_type = Some(1); // Tree2 (Tree1 is excluded)
    }
    
}

fn tree_particle_action(
    particle: &mut Particle,
    particle_list: &mut crate::particles::manager::ParticleList,
    _particle_idx: usize,
    grid: &GameGrid,
) -> bool {
    // Store previous position for line drawing
    particle.prev_x = particle.x;
    particle.prev_y = particle.y;
    
    // Store velocity before moving (needed for branch creation)
    let particle_velocity = particle.velocity;
    let particle_size = particle.size;
    
    // Move particle (draws line from previous position to current)
    particle.x += particle.x_velocity;
    particle.y += particle.y_velocity;
    
    // Check if particle went off canvas (should be removed)
    if particle.off_canvas(grid.width as f32, grid.height as f32) {
        return true; // Remove particle if off canvas
    }
    
    // Check if about to hit wall (similar to TypeScript aboutToHit)
    let radius = particle.size / 2.0;
    let theta = particle.y_velocity.atan2(particle.x_velocity); // atan2(y, x) for direction
    let x_prime = particle.x + theta.cos() * radius;
    let y_prime = particle.y + theta.sin() * radius;
    let idx = (x_prime.round() as u32) + (y_prime.round() as u32) * grid.width;
    
    if idx < grid.elements.len() as u32 && grid.get_index(idx as usize) == Element::Wall {
        return true; // Remove particle if hitting wall
    }
    
    let iterations = particle.action_iterations;
    
    // Check if it's time to create branches
    if let (Some(next_branch), Some(branches), Some(max_branches), Some(branch_spacing), Some(generation), Some(tree_type)) = (
        particle.tree_next_branch,
        particle.tree_branches,
        particle.tree_max_branches,
        particle.tree_branch_spacing,
        particle.tree_generation,
        particle.tree_type,
    ) {
        if iterations >= next_branch {
            let new_branches = branches + 1;
            particle.tree_branches = Some(new_branches);
            
            if max_branches == 0 {
                return true; // End of branch
            }
            
            let leaf_branch = particle.color == Element::Leaf || new_branches >= max_branches;
            
            // Collect all data we need before creating new particles
            let current_angle = particle.angle;
            let current_x = particle.x;
            let current_y = particle.y;
            let current_init_i = particle.init_i;
            
            // Calculate branch angles based on tree type
            let branch_angles = match tree_type {
                0 => {
                    // Tree0: two branches (left and right)
                    let branch_angle = std::f32::consts::PI / 8.0 + rand::thread_rng().gen_range(0.0..1.0) * std::f32::consts::PI / 4.0;
                    vec![current_angle + branch_angle, current_angle - branch_angle]
                }
                1 => {
                    // Tree2: three branches (straight, left, right)
                    let branch_angle = rand::thread_rng().gen_range(0.0..1.0) * std::f32::consts::PI / 16.0 + std::f32::consts::PI / 8.0;
                    vec![current_angle, current_angle + branch_angle, current_angle - branch_angle]
                }
                _ => vec![current_angle], // Fallback
            };
            
            let spacing_factor = match tree_type {
                0 => 0.9,  // Tree0 spacing factor
                1 => 0.6,  // Tree2 spacing factor
                _ => 0.9,
            };
            let new_branch_spacing = (branch_spacing as f32 * spacing_factor) as u32;
            
            // Now create particles (we can borrow particle_list because we're not using particle anymore)
            for branch_angle in branch_angles {
                if let Some(new_particle_idx) = particle_list.add_active_particle(
                    ParticleType::Tree,
                    current_x,
                    current_y,
                    current_init_i,
                ) {
                    if let Some(new_particle) = particle_list.get_particle_mut(new_particle_idx) {
                        new_particle.tree_generation = Some(generation + 1);
                        new_particle.tree_max_branches = Some(max_branches.saturating_sub(1));
                        new_particle.tree_branch_spacing = Some(new_branch_spacing);
                        new_particle.tree_next_branch = Some(new_branch_spacing);
                        new_particle.angle = branch_angle;
                        new_particle.set_velocity(particle_velocity, branch_angle);
                        new_particle.size = (particle_size - 1.0).max(2.0);
                        new_particle.tree_type = Some(tree_type);
                        new_particle.tree_branches = Some(0);
                        
                        if leaf_branch {
                            new_particle.set_color(Element::Leaf);
                        }
                    }
                }
            }
            
            // Check if we've reached max branches (matches TypeScript: if (branches >= maxBranches))
            if new_branches >= max_branches {
                return true; // End of branch - remove particle
            }
            
            // Update next branch time (we can modify particle again now)
            let mut updated_branch_spacing = branch_spacing;
            if updated_branch_spacing > 45 {
                updated_branch_spacing = (updated_branch_spacing as f32 * 0.8) as u32;
            }
            let next_branch_time = iterations + (updated_branch_spacing as f32 * (0.65 + rand::thread_rng().gen_range(0.0..1.0) * 0.35)) as u32;
            particle.tree_next_branch = Some(next_branch_time);
            particle.tree_branch_spacing = Some(updated_branch_spacing);
        }
    }
    
    false
}

// CHARGED_NITRO_PARTICLE
fn charged_nitro_particle_init(particle: &mut Particle, grid: &GameGrid) {
    particle.set_color(crate::elements::Element::Fire);
    // Make the line thinner - just 1-2 pixels wide
    particle.size = 1.5;
    particle.x_velocity = 0.0;
    // Reduce velocity to make it less jarring (TypeScript uses -100, but that's too fast)
    // Use -30 instead to make it more visible and less "floating"
    particle.y_velocity = -30.0;
    
    // Search upwards for a WALL collision (but don't check every pixel)
    particle.min_y = Some(-1.0);
    let step = (3 + rand::thread_rng().gen_range(0..=2)) * grid.width as usize;
    let mut idx = particle.init_i;
    while idx > 0 {
        if grid.get_index(idx) == crate::elements::Element::Wall {
            particle.min_y = Some((idx / grid.width as usize) as f32);
            break;
        }
        if idx < step {
            break;
        }
        idx -= step;
    }
}

fn charged_nitro_particle_action(particle: &mut Particle, grid: &GameGrid) -> bool {
    // Store previous position for line drawing (though we use init position, not prev)
    // Move particle upward (creates vertical fire column)
    particle.x += particle.x_velocity;
    let old_y = particle.y;
    if let Some(min_y) = particle.min_y {
        particle.y = (particle.y + particle.y_velocity).max(min_y);
    } else {
        particle.y += particle.y_velocity;
    }
    
    // Remove if hit wall (y stopped at min_y) or off canvas
    if let Some(min_y) = particle.min_y {
        if old_y > min_y && particle.y <= min_y {
            return true; // Hit wall (stopped at min_y)
        }
    }
    // Remove if off canvas or moved too far upward
    if particle.y < 0.0 || particle.off_canvas(grid.width as f32, grid.height as f32) {
        return true;
    }
    
    false
}

// NUKE_PARTICLE
fn nuke_particle_init(particle: &mut Particle, rng: &mut impl Rng) {
    particle.set_color(crate::elements::Element::Fire);
    let max_dimension = SIZE.x.max(SIZE.y) as f32;
    particle.size = max_dimension / 4.0 + (rng.gen_range(0.0..1.0) * max_dimension) / 8.0;
}

fn nuke_particle_action(particle: &mut Particle) -> bool {
    // Remove after 4 iterations
    if particle.action_iterations > 4 {
        return true;
    }
    
    false
}
