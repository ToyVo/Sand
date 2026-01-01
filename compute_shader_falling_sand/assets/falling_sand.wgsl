// The shader reads the previous frame's state from the `input` texture, and writes the new state of
// each pixel to the `output` texture. The textures are flipped each step to progress the
// simulation.
// Two textures are needed for falling sand as each pixel of step N depends on the state of its
// neighbors at step N-1.
// Uses gather/advection approach: each thread determines what should be at its location by reading
// from source locations (above, diagonal above). This eliminates race conditions and ensures full
// parallelism without atomic operations.

@group(0) @binding(0) var input: texture_storage_2d<rgba32float, read>;

@group(0) @binding(1) var output: texture_storage_2d<rgba32float, write>;

@group(0) @binding(2) var element_type_input: texture_storage_2d<r32uint, read>;

@group(0) @binding(3) var element_type_output: texture_storage_2d<r32uint, write>;

@group(0) @binding(4) var<uniform> config: FallingSandUniforms;

struct FallingSandUniforms {
    sand_color: vec4<f32>, // Single color for all regular sand
    size: vec2<u32>,
    click_position: vec2<i32>,
    spigot_sizes: vec4<u32>, // 0 = disabled, 1-6 = spigot size
    spigot_elements: vec4<u32>, // 0 = regular sand, 1 = rainbow sand
    click_radius: f32, // Radius of the circle for placing/removing sand
    click_action: u32, // 0 = no action, 1 = add element, 2 = remove element
    selected_element: u32,
    wall_color: vec4<f32>, // Color for walls
    color_shift_enabled: u32, // 1 = true, 0 = false
    sim_step: u32, // Simulation step counter for alternating diagonal movement
    overwrite_mode: u32, // 1 = true, 0 = false - when enabled, drawing overwrites existing materials
    fall_into_void: u32, // 1 = true, 0 = false - when enabled, elements fall off screen edges
}

// Background color (black)
const BG_COLOR: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 1.0);
const COLOR_THRESHOLD: f32 = 0.05;

// Element type IDs (stored in separate texture)
const ELEMENT_BACKGROUND: u32 = 0u;
const ELEMENT_WALL: u32 = 1u;
const ELEMENT_SAND: u32 = 2u;
const ELEMENT_RAINBOW_SAND: u32 = 3u;

fn color_equal(a: vec4<f32>, b: vec4<f32>) -> bool {
    return abs(a.x - b.x) < COLOR_THRESHOLD && 
           abs(a.y - b.y) < COLOR_THRESHOLD && 
           abs(a.z - b.z) < COLOR_THRESHOLD;
}

// Get element type from element type texture
fn get_element_type(location: vec2<i32>) -> u32 {
    return textureLoad(element_type_input, location).r;
}

// Check if a pixel is background
fn is_background_element(location: vec2<i32>) -> bool {
    return get_element_type(location) == ELEMENT_BACKGROUND;
}

// Check if a pixel is sand
fn is_sand_element(location: vec2<i32>) -> bool {
    return get_element_type(location) == ELEMENT_SAND;
}

// Check if a pixel is rainbow sand
fn is_rainbow_sand_element(location: vec2<i32>) -> bool {
    return get_element_type(location) == ELEMENT_RAINBOW_SAND;
}

// Check if a pixel is a wall
fn is_wall(location: vec2<i32>) -> bool {
    return get_element_type(location) == ELEMENT_WALL;
}

// Calculate rainbow sand color based on sim_step and position
// This creates a shifting rainbow effect that changes over time
fn calculate_rainbow_sand_color(sim_step: u32, pos_x: i32, pos_y: i32, use_position_variation: bool) -> vec4<f32> {
    // Slow down the color shift by dividing sim_step - this makes color changes more gradual
    // Using / 5 instead of / 10 makes it shift faster
    let time_shift = sim_step % 360u;
    var hue: f32;
    if (use_position_variation) {
        // For spigot-spawned sand: add position variation for per-particle differences
        let pos_variation = (u32(pos_x) * 73856093u + u32(pos_y) * 19349663u) % 60u;
        let combined = (time_shift + pos_variation) % 360u;
        hue = f32(combined);
    } else {
        // For click-placed sand: use only time-based shifting for smooth, consistent color
        hue = f32(time_shift);
    }
    let saturation = 0.8; // High saturation for vibrant colors
    let value = 0.9; // Bright value
    
    // HSV to RGB conversion
    let c = value * saturation;
    let x = c * (1.0 - abs((hue / 60.0) % 2.0 - 1.0));
    let m = value - c;
    
    var r: f32;
    var g: f32;
    var b: f32;
    
    if (hue < 60.0) {
        r = c;
        g = x;
        b = 0.0;
    } else if (hue < 120.0) {
        r = x;
        g = c;
        b = 0.0;
    } else if (hue < 180.0) {
        r = 0.0;
        g = c;
        b = x;
    } else if (hue < 240.0) {
        r = 0.0;
        g = x;
        b = c;
    } else if (hue < 300.0) {
        r = x;
        g = 0.0;
        b = c;
    } else {
        r = c;
        g = 0.0;
        b = x;
    }
    
    return vec4<f32>(r + m, g + m, b + m, 1.0);
}

fn is_background(location: vec2<i32>, offset_x: i32, offset_y: i32) -> bool {
    let check_location = location + vec2<i32>(offset_x, offset_y);
    return is_background_element(check_location);
}

@compute @workgroup_size(8, 8, 1)
fn init(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));
    
    // Start with empty grid - background color and element type
    let color = BG_COLOR;
    textureStore(output, location, color);
    textureStore(element_type_output, location, vec4<u32>(ELEMENT_BACKGROUND, 0u, 0u, 0u));
}

@compute @workgroup_size(8, 8, 1)
fn update(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));
    let size = vec2<i32>(i32(config.size.x), i32(config.size.y));
    
    // Check bounds
    if (location.x < 0 || location.x >= size.x || location.y < 0 || location.y >= size.y) {
        return;
    }
    
    // Priority 1: Handle click actions (highest priority)
    let click_pos = config.click_position;
    let click_action = config.click_action;
    if (click_pos.x >= 0 && click_pos.y >= 0 && click_action > 0u) {
        let dx = f32(location.x - click_pos.x);
        let dy = f32(location.y - click_pos.y);
        let distance_squared = dx * dx + dy * dy;
        let radius_squared = config.click_radius * config.click_radius;
        
        // Check if this pixel is within the circle
        if (distance_squared <= radius_squared) {
            if (click_action == 1u) {
                // Add element based on selected_element - respect overwrite mode
                let current_element_type = get_element_type(location);
                let is_bg = current_element_type == ELEMENT_BACKGROUND;
                let is_wall_pixel = current_element_type == ELEMENT_WALL;
                
                // Don't overwrite walls unless overwrite mode is enabled
                if (config.overwrite_mode == 1u || (is_bg && !is_wall_pixel)) {
                    var element_color: vec4<f32>;
                    var element_type_id: u32;
                    if (config.selected_element == ELEMENT_SAND) {
                        // Sand - use single sand color
                        element_color = vec4(config.sand_color.rgb, 1.0);
                        element_type_id = ELEMENT_SAND;
                    } else if (config.selected_element == ELEMENT_RAINBOW_SAND) {
                        // Rainbow sand - all pixels in circle have same color, shifts over time
                        let rainbow_color = calculate_rainbow_sand_color(config.sim_step, click_pos.x, click_pos.y, false);
                        element_color = vec4(rainbow_color.rgb, 1.0);
                        element_type_id = ELEMENT_RAINBOW_SAND;
                    } else if (config.selected_element == ELEMENT_WALL) {
                        // Wall - use wall_color
                        element_color = vec4(config.wall_color.rgb, 1.0);
                        element_type_id = ELEMENT_WALL;
                    } else {
                        // Remove element (clear the pixel)
                        element_color = BG_COLOR;
                        element_type_id = ELEMENT_BACKGROUND;
                    }
                    textureStore(output, location, element_color);
                    textureStore(element_type_output, location, vec4<u32>(element_type_id, 0u, 0u, 0u));
                    return;
                }
            } else if (click_action == 2u) {
                // Remove element (clear the pixel)
                textureStore(output, location, BG_COLOR);
                textureStore(element_type_output, location, vec4<u32>(ELEMENT_BACKGROUND, 0u, 0u, 0u));
                return;
            }
        }
    }
    
    // Priority 2: Handle spigot spawning (if at top)
    // 4 spigots evenly spaced across the top, each with independent size control
    let spigot_height = 10i; // Match ProjectSandBevy SPIGOT_HEIGHT
    
    // Calculate positions for 4 spigots (at 1/5, 2/5, 3/5, 4/5 of screen width)
    let spigot_1_x = size.x / 5i;
    let spigot_2_x = (size.x * 2i) / 5i;
    let spigot_3_x = (size.x * 3i) / 5i;
    let spigot_4_x = (size.x * 4i) / 5i;
    
    // Check if location is within any of the 4 spigots (each with its own size)
    if (location.y >= 0 && location.y < spigot_height) {
        var in_spigot = false;
        
        // Check spigot 1
        if (config.spigot_sizes.x > 0u) {
            let spigot_half_width = i32(config.spigot_sizes.x) / 2i;
            if (location.x >= spigot_1_x - spigot_half_width && location.x <= spigot_1_x + spigot_half_width) {
                in_spigot = true;
            }
        }
        
        // Check spigot 2
        if (!in_spigot && config.spigot_sizes.y > 0u) {
            let spigot_half_width = i32(config.spigot_sizes.y) / 2i;
            if (location.x >= spigot_2_x - spigot_half_width && location.x <= spigot_2_x + spigot_half_width) {
                in_spigot = true;
            }
        }
        
        // Check spigot 3
        if (!in_spigot && config.spigot_sizes.z > 0u) {
            let spigot_half_width = i32(config.spigot_sizes.z) / 2i;
            if (location.x >= spigot_3_x - spigot_half_width && location.x <= spigot_3_x + spigot_half_width) {
                in_spigot = true;
            }
        }
        
        // Check spigot 4
        if (!in_spigot && config.spigot_sizes.w > 0u) {
            let spigot_half_width = i32(config.spigot_sizes.w) / 2i;
            if (location.x >= spigot_4_x - spigot_half_width && location.x <= spigot_4_x + spigot_half_width) {
                in_spigot = true;
            }
        }
        
        if (in_spigot) {
            let current_element_type = get_element_type(location);
            let is_bg = current_element_type == ELEMENT_BACKGROUND;
            // Spawn sand if location is empty (10% chance per frame, matching ProjectSandBevy)
            // Use a simple hash-based random to get ~10% chance
            let hash = u32(location.x) * 73856093u + u32(location.y) * 19349663u + config.sim_step;
            if ((hash % 10u) == 0u && is_bg) {
                // Determine which spigot this location belongs to and use its type
                var spigot_type = config.spigot_elements.x;
                
                if (location.x >= spigot_1_x - i32(config.spigot_sizes.x) / 2i && location.x <= spigot_1_x + i32(config.spigot_sizes.x) / 2i && config.spigot_sizes.x > 0u) {
                    spigot_type = config.spigot_elements.x;
                } else if (location.x >= spigot_2_x - i32(config.spigot_sizes.y) / 2i && location.x <= spigot_2_x + i32(config.spigot_sizes.y) / 2i && config.spigot_sizes.y > 0u) {
                    spigot_type = config.spigot_elements.y;
                } else if (location.x >= spigot_3_x - i32(config.spigot_sizes.z) / 2i && location.x <= spigot_3_x + i32(config.spigot_sizes.z) / 2i && config.spigot_sizes.z > 0u) {
                    spigot_type = config.spigot_elements.z;
                } else if (location.x >= spigot_4_x - i32(config.spigot_sizes.w) / 2i && location.x <= spigot_4_x + i32(config.spigot_sizes.w) / 2i && config.spigot_sizes.w > 0u) {
                    spigot_type = config.spigot_elements.w;
                }
                
                // Use rainbow sand color if spigot type is rainbow, otherwise use regular sand color
                var final_color: vec4<f32>;
                var element_type_id: u32;
                if (spigot_type == ELEMENT_RAINBOW_SAND) {
                    // Rainbow sand - calculate color based on sim_step and position (shifts over time)
                    let rainbow_color = calculate_rainbow_sand_color(config.sim_step, location.x, location.y, true);
                    final_color = vec4(rainbow_color.rgb, 1.0);
                    element_type_id = ELEMENT_RAINBOW_SAND;
                } else {
                    // Regular sand - use single sand color
                    final_color = vec4(config.sand_color.rgb, 1.0);
                    element_type_id = ELEMENT_SAND;
                }
                textureStore(output, location, final_color);
                textureStore(element_type_output, location, vec4<u32>(element_type_id, 0u, 0u, 0u));
                return;
            }
        }
    }
    
    // Priority 3: Gather/advection - determine what should be at this location
    // by reading from source locations (above, diagonal above)
    
    // Read current element type to check if it's empty
    let current_element_type = get_element_type(location);
    let current_is_bg = current_element_type == ELEMENT_BACKGROUND;
    
    // If current location has a wall, keep it in place (walls don't move)
    if (current_element_type == ELEMENT_WALL) {
        let current_pixel = textureLoad(input, location);
        textureStore(output, location, current_pixel);
        textureStore(element_type_output, location, vec4<u32>(ELEMENT_WALL, 0u, 0u, 0u));
        return;
    }
    
    // If current location has sand (regular or rainbow), check if it should move (and clear if it does)
    let current_is_sand = current_element_type == ELEMENT_SAND || current_element_type == ELEMENT_RAINBOW_SAND;
    if (current_is_sand) {
        // This location has sand - check if it can move down
        let below = vec2<i32>(location.x, location.y + 1);
        if (below.y < size.y) {
            let below_element_type = get_element_type(below);
            let below_is_bg = below_element_type == ELEMENT_BACKGROUND;
            let below_is_wall = below_element_type == ELEMENT_WALL;
            
            // Sand can't fall through walls - if below is a wall, sand stays in place
            if (below_is_wall) {
                // Sand is directly above a wall - it stays in place (piles on top)
                let current_pixel = textureLoad(input, location);
                textureStore(output, location, current_pixel);
                textureStore(element_type_output, location, vec4<u32>(current_element_type, 0u, 0u, 0u));
                return;
            }
            
            // Space below is empty - sand will move down
            if (below_is_bg) {
                // Clear this location (the thread below will write the sand)
                textureStore(output, location, BG_COLOR);
                textureStore(element_type_output, location, vec4<u32>(ELEMENT_BACKGROUND, 0u, 0u, 0u));
                return;
            }
            
            // Space below is occupied (by sand) - check diagonal movement
            // But first, check if the space directly below the diagonal is also a wall
            // If so, don't allow diagonal movement (prevents infinite falling off walls)
            let odd_frame = config.sim_step % 2u == 1u;
            let below_left = vec2<i32>(location.x - 1, location.y + 1);
            let below_right = vec2<i32>(location.x + 1, location.y + 1);
            let first_diag = select(below_left, below_right, odd_frame);
            let second_diag = select(below_right, below_left, odd_frame);
            
            // Check first diagonal
            if (first_diag.y < size.y && first_diag.x >= 0 && first_diag.x < size.x) {
                let first_diag_element_type = get_element_type(first_diag);
                let first_diag_is_bg = first_diag_element_type == ELEMENT_BACKGROUND;
                let first_diag_is_wall = first_diag_element_type == ELEMENT_WALL;
                
                // Check what's directly below the diagonal position
                let below_first_diag = vec2<i32>(first_diag.x, first_diag.y + 1);
                var below_first_diag_is_wall = false;
                if (below_first_diag.y < size.y) {
                    let below_first_diag_element_type = get_element_type(below_first_diag);
                    below_first_diag_is_wall = below_first_diag_element_type == ELEMENT_WALL;
                }
                
                // Only allow diagonal movement if:
                // 1. Diagonal space is empty and not a wall
                // 2. Space directly below diagonal is NOT a wall (prevents infinite falling off walls)
                if (first_diag_is_bg && !first_diag_is_wall && !below_first_diag_is_wall) {
                    // Can move diagonally - clear this location
                    textureStore(output, location, BG_COLOR);
                    textureStore(element_type_output, location, vec4<u32>(ELEMENT_BACKGROUND, 0u, 0u, 0u));
                    return;
                }
            }
            
            // Check second diagonal
            if (second_diag.y < size.y && second_diag.x >= 0 && second_diag.x < size.x) {
                let second_diag_element_type = get_element_type(second_diag);
                let second_diag_is_bg = second_diag_element_type == ELEMENT_BACKGROUND;
                let second_diag_is_wall = second_diag_element_type == ELEMENT_WALL;
                
                // Check what's directly below the diagonal position
                let below_second_diag = vec2<i32>(second_diag.x, second_diag.y + 1);
                var below_second_diag_is_wall = false;
                if (below_second_diag.y < size.y) {
                    let below_second_diag_element_type = get_element_type(below_second_diag);
                    below_second_diag_is_wall = below_second_diag_element_type == ELEMENT_WALL;
                }
                
                // Only allow diagonal movement if:
                // 1. Diagonal space is empty and not a wall
                // 2. Space directly below diagonal is NOT a wall (prevents infinite falling off walls)
                if (second_diag_is_bg && !second_diag_is_wall && !below_second_diag_is_wall) {
                    // Can move diagonally - clear this location
                    textureStore(output, location, BG_COLOR);
                    textureStore(element_type_output, location, vec4<u32>(ELEMENT_BACKGROUND, 0u, 0u, 0u));
                    return;
                }
            }
        } else {
            // At bottom edge - check fall_into_void
            if (config.fall_into_void == 1u) {
                textureStore(output, location, BG_COLOR);
                textureStore(element_type_output, location, vec4<u32>(ELEMENT_BACKGROUND, 0u, 0u, 0u));
                return;
            }
        }
        
        // Sand can't move - stays in place
        let current_pixel = textureLoad(input, location);
        textureStore(output, location, current_pixel);
        textureStore(element_type_output, location, vec4<u32>(current_element_type, 0u, 0u, 0u));
        return;
    }
    
    // Current location is empty - check if sand above would fall here
    // Note: If current location is a wall, we already handled it above (walls stay in place)
    let above = vec2<i32>(location.x, location.y - 1);
    if (above.y >= 0) {
        let above_element_type = get_element_type(above);
        let above_is_sand = above_element_type == ELEMENT_SAND || above_element_type == ELEMENT_RAINBOW_SAND;
        
        // Sand can fall onto empty spaces (including spaces above walls)
        if (above_is_sand) {
            // Sand exists above - check if it can fall straight here
            // Check if we're at the bottom edge and fall_into_void is enabled
            if (location.y >= size.y - 1 && config.fall_into_void == 1u) {
                // Sand would fall into void - disappear (write background)
                textureStore(output, location, BG_COLOR);
                textureStore(element_type_output, location, vec4<u32>(ELEMENT_BACKGROUND, 0u, 0u, 0u));
                return;
            }
            // Space below sand is empty - sand falls straight down here
            let above_pixel = textureLoad(input, above);
            textureStore(output, location, above_pixel);
            textureStore(element_type_output, location, vec4<u32>(above_element_type, 0u, 0u, 0u));
            return;
        }
    }
    
    // Check diagonal movement (alternating preference based on frame)
    // Only check diagonals if no sand fell straight down AND location is empty
    let odd_frame = config.sim_step % 2u == 1u;
    let above_left = vec2<i32>(location.x - 1, location.y - 1);
    let above_right = vec2<i32>(location.x + 1, location.y - 1);
    
    // Determine which diagonal to check first based on frame alternation
    // Odd frame: prefer right diagonal first, then left
    // Even frame: prefer left diagonal first, then right
    let first_diag = select(above_left, above_right, odd_frame);
    let second_diag = select(above_right, above_left, odd_frame);
    
    // Check first diagonal (preferred direction for this frame)
    if (first_diag.y >= 0 && first_diag.x >= 0 && first_diag.x < size.x) {
        let first_diag_element_type = get_element_type(first_diag);
        let first_diag_is_sand = first_diag_element_type == ELEMENT_SAND || first_diag_element_type == ELEMENT_RAINBOW_SAND;
        
        if (first_diag_is_sand) {
            // Sand exists at first diagonal above
            // Check if space directly below it is occupied (forcing diagonal fall)
            let directly_below_first = vec2<i32>(first_diag.x, first_diag.y + 1);
            if (directly_below_first.y < size.y) {
                let below_first_element_type = get_element_type(directly_below_first);
                let below_first_is_bg = below_first_element_type == ELEMENT_BACKGROUND;
                let below_first_is_wall = below_first_element_type == ELEMENT_WALL;
                
                // Only allow diagonal fall if:
                // 1. Space directly below the diagonal source is occupied (can't fall straight)
                // 2. Space directly below the diagonal source is NOT a wall (prevents infinite falling off walls)
                // 3. This location is empty (already checked above)
                if (!below_first_is_bg && !below_first_is_wall) {
                    // Check if we're at the edge and fall_into_void
                    if (location.y >= size.y - 1 && config.fall_into_void == 1u) {
                        // Sand would fall into void - disappear
                        textureStore(output, location, BG_COLOR);
                        textureStore(element_type_output, location, vec4<u32>(ELEMENT_BACKGROUND, 0u, 0u, 0u));
                        return;
                    }
                    // Check if we're at left/right edge and fall_into_void
                    if ((location.x <= 0 || location.x >= size.x - 1) && config.fall_into_void == 1u) {
                        // Sand would fall off side edge - disappear
                        textureStore(output, location, BG_COLOR);
                        textureStore(element_type_output, location, vec4<u32>(ELEMENT_BACKGROUND, 0u, 0u, 0u));
                        return;
                    }
                    // Sand falls diagonally here - only from preferred direction
                    let first_diag_pixel = textureLoad(input, first_diag);
                    textureStore(output, location, first_diag_pixel);
                    textureStore(element_type_output, location, vec4<u32>(first_diag_element_type, 0u, 0u, 0u));
                    return;
                }
            } else {
                // Sand at first diagonal is at bottom edge - would fall into void
                if (config.fall_into_void == 1u) {
                    textureStore(output, location, BG_COLOR);
                    textureStore(element_type_output, location, vec4<u32>(ELEMENT_BACKGROUND, 0u, 0u, 0u));
                    return;
                }
            }
        }
    }
    
    // Check second diagonal (alternate direction) - only if first didn't work
    if (second_diag.y >= 0 && second_diag.x >= 0 && second_diag.x < size.x) {
        let second_diag_element_type = get_element_type(second_diag);
        let second_diag_is_sand = second_diag_element_type == ELEMENT_SAND || second_diag_element_type == ELEMENT_RAINBOW_SAND;
        
        if (second_diag_is_sand) {
            // Sand exists at second diagonal above
            // Check if space directly below it is occupied (forcing diagonal fall)
            let directly_below_second = vec2<i32>(second_diag.x, second_diag.y + 1);
            if (directly_below_second.y < size.y) {
                let below_second_element_type = get_element_type(directly_below_second);
                let below_second_is_bg = below_second_element_type == ELEMENT_BACKGROUND;
                let below_second_is_wall = below_second_element_type == ELEMENT_WALL;
                
                // Only allow diagonal fall if:
                // 1. Space directly below the diagonal source is occupied (can't fall straight)
                // 2. Space directly below the diagonal source is NOT a wall (prevents infinite falling off walls)
                // 3. This location is empty (already checked above)
                // 4. This location is not a wall (sand can't fall through walls)
                if (!below_second_is_bg && !below_second_is_wall) {
                    // Check if we're at the edge and fall_into_void
                    if (location.y >= size.y - 1 && config.fall_into_void == 1u) {
                        // Sand would fall into void - disappear
                        textureStore(output, location, BG_COLOR);
                        textureStore(element_type_output, location, vec4<u32>(ELEMENT_BACKGROUND, 0u, 0u, 0u));
                        return;
                    }
                    // Check if we're at left/right edge and fall_into_void
                    if ((location.x <= 0 || location.x >= size.x - 1) && config.fall_into_void == 1u) {
                        // Sand would fall off side edge - disappear
                        textureStore(output, location, BG_COLOR);
                        textureStore(element_type_output, location, vec4<u32>(ELEMENT_BACKGROUND, 0u, 0u, 0u));
                        return;
                    }
                    // Sand falls diagonally here from alternate direction
                    let second_diag_pixel = textureLoad(input, second_diag);
                    textureStore(output, location, second_diag_pixel);
                    textureStore(element_type_output, location, vec4<u32>(second_diag_element_type, 0u, 0u, 0u));
                    return;
                }
            } else {
                // Sand at second diagonal is at bottom edge - would fall into void
                if (config.fall_into_void == 1u) {
                    textureStore(output, location, BG_COLOR);
                    textureStore(element_type_output, location, vec4<u32>(ELEMENT_BACKGROUND, 0u, 0u, 0u));
                    return;
                }
            }
        }
    }
    
    // No sand moving here - write background
    // (We already checked current_is_bg above, so if we reach here, location is empty)
    textureStore(output, location, BG_COLOR);
    textureStore(element_type_output, location, vec4<u32>(ELEMENT_BACKGROUND, 0u, 0u, 0u));
}

