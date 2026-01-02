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
    size: vec2<u32>,
    click_position: vec2<i32>,
    spigot_sizes: vec4<u32>, // 0 = disabled, 1-6 = spigot size
    spigot_elements: vec4<u32>, // 0 = regular sand, 1 = rainbow sand
    click_radius: f32, // Radius of the circle for placing/removing sand
    selected_element: u32,
    sim_step: u32, // Simulation step counter for alternating diagonal movement
    bit_field: u32, // Bit field for various flags, 0 = overwrite mode, 1 = fall into void
}

const OVERWRITE_MODE_BIT: u32 = 0u;
const FALL_INTO_VOID_BIT: u32 = 1u;

const BACKGROUND_COLOR: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 1.0);
const WALL_COLOR: vec4<f32> = vec4<f32>(0.49803922, 0.49803922, 0.49803922, 1.0);
const SAND_COLOR: vec4<f32> = vec4<f32>(0.87450980, 0.75686275, 0.38823529, 1.0);
const WATER_COLOR: vec4<f32> = vec4<f32>(0.0, 0.03921569, 1.0, 1.0);
const PLANT_COLOR: vec4<f32> = vec4<f32>(0.0, 0.86274510, 0.0, 1.0);
const FIRE_COLOR: vec4<f32> = vec4<f32>(1.0, 0.0, 0.03921569, 1.0);
const SALT_COLOR: vec4<f32> = vec4<f32>(0.99215686, 0.99215686, 0.99215686, 1.0);
const SALT_WATER_COLOR: vec4<f32> = vec4<f32>(0.49803922, 0.68627451, 1.0, 1.0);
const OIL_COLOR: vec4<f32> = vec4<f32>(0.58823529, 0.23529412, 0.0, 1.0);
const SPOUT_COLOR: vec4<f32> = vec4<f32>(0.45882353, 0.74117647, 0.98823529, 1.0);
const WELL_COLOR: vec4<f32> = vec4<f32>(0.51372549, 0.04313725, 0.10980392, 1.0);
const TORCH_COLOR: vec4<f32> = vec4<f32>(0.78431373, 0.01960784, 0.0, 1.0);
const GUNPOWDER_COLOR: vec4<f32> = vec4<f32>(0.66666667, 0.66666667, 0.54901961, 1.0);
const WAX_COLOR: vec4<f32> = vec4<f32>(0.93725490, 0.88235294, 0.82745098, 1.0);
const FALLING_WAX_COLOR: vec4<f32> = vec4<f32>(0.94117647, 0.88235294, 0.82745098, 1.0);
const NITRO_COLOR: vec4<f32> = vec4<f32>(0.0, 0.58823529, 0.10196078, 1.0);
const NAPALM_COLOR: vec4<f32> = vec4<f32>(0.86274510, 0.50196078, 0.27450980, 1.0);
const C4_COLOR: vec4<f32> = vec4<f32>(0.94117647, 0.90196078, 0.58823529, 1.0);
const CONCRETE_COLOR: vec4<f32> = vec4<f32>(0.70588235, 0.70588235, 0.70588235, 1.0);
const FUSE_COLOR: vec4<f32> = vec4<f32>(0.85882353, 0.68627451, 0.78039216, 1.0);
const ICE_COLOR: vec4<f32> = vec4<f32>(0.63137255, 0.90980392, 1.0, 1.0);
const CHILLED_ICE_COLOR: vec4<f32> = vec4<f32>(0.07843137, 0.60000000, 0.86274510, 1.0);
const LAVA_COLOR: vec4<f32> = vec4<f32>(0.96078431, 0.43137255, 0.15686275, 1.0);
const ROCK_COLOR: vec4<f32> = vec4<f32>(0.26666667, 0.15686275, 0.03137255, 1.0);
const STEAM_COLOR: vec4<f32> = vec4<f32>(0.76470588, 0.83921569, 0.92156863, 1.0);
const CRYO_COLOR: vec4<f32> = vec4<f32>(0.0, 0.83529412, 1.0, 1.0);
const MYSTERY_COLOR: vec4<f32> = vec4<f32>(0.63529412, 0.90980392, 0.76862745, 1.0);
const METHANE_COLOR: vec4<f32> = vec4<f32>(0.54901961, 0.54901961, 0.54901961, 1.0);
const SOIL_COLOR: vec4<f32> = vec4<f32>(0.47058824, 0.29411765, 0.12941176, 1.0);
const WET_SOIL_COLOR: vec4<f32> = vec4<f32>(0.27450980, 0.13725490, 0.03921569, 1.0);
const BRANCH_COLOR: vec4<f32> = vec4<f32>(0.65098039, 0.50196078, 0.39215686, 1.0);
const LEAF_COLOR: vec4<f32> = vec4<f32>(0.32156863, 0.41960784, 0.17647059, 1.0);
const POLLEN_COLOR: vec4<f32> = vec4<f32>(0.90196078, 0.92156863, 0.43137255, 1.0);
const CHARGED_NITRO_COLOR: vec4<f32> = vec4<f32>(0.96078431, 0.38431373, 0.30588235, 1.0);
const ACID_COLOR: vec4<f32> = vec4<f32>(0.61568627, 0.94117647, 0.15686275, 1.0);
const THERMITE_COLOR: vec4<f32> = vec4<f32>(0.76470588, 0.54901961, 0.27450980, 1.0);
const BURNING_THERMITE_COLOR: vec4<f32> = vec4<f32>(1.0, 0.50980392, 0.50980392, 1.0);
const ZOMBIE_COLOR: vec4<f32> = vec4<f32>(0.92549020, 0.87450980, 0.96078431, 1.0);
const ZOMBIE_WET_COLOR: vec4<f32> = vec4<f32>(0.92549020, 0.87450980, 0.96078431, 1.0);
const ZOMBIE_BURNING_COLOR: vec4<f32> = vec4<f32>(0.98039216, 0.50980392, 0.50980392, 1.0);
const ZOMBIE_FROZEN_COLOR: vec4<f32> = vec4<f32>(0.74509804, 0.74509804, 0.98039216, 1.0);

// Element type IDs (stored in separate texture)
const BACKGROUND_ID: u32 = 0u;
const WALL_ID: u32 = 1u;
const SAND_ID: u32 = 2u;
const RAINBOW_SAND_ID: u32 = 3u;

fn bit_field_get(bit: u32) -> bool {
    return ((config.bit_field >> bit) & 1u) == 1u;
}

// Get element type from element type texture
fn get_element_type(location: vec2<i32>) -> u32 {
    return textureLoad(element_type_input, location).r;
}

// Hash-based PRNG to generate a deterministic random value between 0.0 and 1.0
// Based on position and sim_step to ensure each particle has its own random sequence
fn hash_random(pos_x: i32, pos_y: i32, sim_step: u32) -> f32 {
    // Use large prime numbers for better hash distribution
    let hash = u32(pos_x) * 73856093u + u32(pos_y) * 19349663u + sim_step * 83492791u;
    // Convert to float in range [0.0, 1.0)
    // Use modulo with a large number to get good distribution
    return f32(hash % 1000000u) / 1000000.0;
}

// Check if a sand particle should be affected by gravity (95% chance)
// Returns true if gravity should apply, false if particle should stay in place
fn should_apply_gravity(pos_x: i32, pos_y: i32, sim_step: u32) -> bool {
    let random_value = hash_random(pos_x, pos_y, sim_step);
    // 95% chance to be affected by gravity (0.95 threshold)
    return random_value < 0.95;
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
    return get_element_type(check_location) == BACKGROUND_ID;
}

@compute @workgroup_size(8, 8, 1)
fn init(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));
    
    // Start with empty grid - background color and element type
    let color = BACKGROUND_COLOR;
    textureStore(output, location, color);
    textureStore(element_type_output, location, vec4<u32>(BACKGROUND_ID, 0u, 0u, 0u));
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
    if (click_pos.x >= 0 && click_pos.y >= 0) {
        let dx = f32(location.x - click_pos.x);
        let dy = f32(location.y - click_pos.y);
        let distance_squared = dx * dx + dy * dy;
        let radius_squared = config.click_radius * config.click_radius;
        
        // Check if this pixel is within the circle
        if (distance_squared <= radius_squared) {
            // Add element based on selected_element - respect overwrite mode
            let current_element_type = get_element_type(location);
            let is_bg = current_element_type == BACKGROUND_ID;
            let is_wall_pixel = current_element_type == WALL_ID;
            
            let overwrite_mode = bit_field_get(OVERWRITE_MODE_BIT);
            // Don't overwrite walls unless overwrite mode is enabled
            if (overwrite_mode || (is_bg && !is_wall_pixel)) {
                var element_color: vec4<f32>;
                var element_type_id: u32;
                if (config.selected_element == SAND_ID) {
                    element_color = SAND_COLOR;
                    element_type_id = SAND_ID;
                } else if (config.selected_element == RAINBOW_SAND_ID) {
                    // Rainbow sand - all pixels in circle have same color, shifts over time
                    let rainbow_color = calculate_rainbow_sand_color(config.sim_step, click_pos.x, click_pos.y, false);
                    element_color = vec4(rainbow_color.rgb, 1.0);
                    element_type_id = RAINBOW_SAND_ID;
                } else if (config.selected_element == WALL_ID) {
                    element_color = WALL_COLOR;
                    element_type_id = WALL_ID;
                } else {
                    // Anything unimplemented should be treated as background / eraser
                    element_color = BACKGROUND_COLOR;
                    element_type_id = BACKGROUND_ID;
                }
                textureStore(output, location, element_color);
                textureStore(element_type_output, location, vec4<u32>(element_type_id, 0u, 0u, 0u));
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
            let is_bg = current_element_type == BACKGROUND_ID;
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
                if (spigot_type == RAINBOW_SAND_ID) {
                    // Rainbow sand - calculate color based on sim_step and position (shifts over time)
                    let rainbow_color = calculate_rainbow_sand_color(config.sim_step, location.x, location.y, true);
                    final_color = vec4(rainbow_color.rgb, 1.0);
                    element_type_id = RAINBOW_SAND_ID;
                } else {
                    final_color = SAND_COLOR;
                    element_type_id = SAND_ID;
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
    let current_is_bg = current_element_type == BACKGROUND_ID;
    
    // If current location has a wall, keep it in place (walls don't move)
    if (current_element_type == WALL_ID) {
        let current_pixel = textureLoad(input, location);
        textureStore(output, location, current_pixel);
        textureStore(element_type_output, location, vec4<u32>(WALL_ID, 0u, 0u, 0u));
        return;
    }
    
    let fall_into_void = bit_field_get(FALL_INTO_VOID_BIT);
    // If current location has sand (regular or rainbow), check if it should move (and clear if it does)
    let current_is_sand = current_element_type == SAND_ID || current_element_type == RAINBOW_SAND_ID;
    if (current_is_sand) {
        // Check if this sand particle should be affected by gravity (95% chance)
        // If not affected by gravity, it stays in place regardless of whether it could move
        if (!should_apply_gravity(location.x, location.y, config.sim_step)) {
            // 5% chance: sand is not affected by gravity this frame - stays in place
            let current_pixel = textureLoad(input, location);
            textureStore(output, location, current_pixel);
            textureStore(element_type_output, location, vec4<u32>(current_element_type, 0u, 0u, 0u));
            return;
        }
        
        // 95% chance: sand is affected by gravity - check if it can move
        // Match ProjectSandBevy's do_gravity priority: below_adjacent (down, below-left, below-right), then adjacent (left/right)
        
        if (location.y >= size.y - 1) {
            // At bottom edge - check fall_into_void
            if (fall_into_void) {
                textureStore(output, location, BACKGROUND_COLOR);
                textureStore(element_type_output, location, vec4<u32>(BACKGROUND_ID, 0u, 0u, 0u));
                return;
            }
            // Can't move - stays in place
            let current_pixel = textureLoad(input, location);
            textureStore(output, location, current_pixel);
            textureStore(element_type_output, location, vec4<u32>(current_element_type, 0u, 0u, 0u));
            return;
        }
        
        // Check below_adjacent: directly below, then below-left, then below-right
        let below = vec2<i32>(location.x, location.y + 1);
        let below_element_type = get_element_type(below);
        let below_is_bg = below_element_type == BACKGROUND_ID;
        let below_is_wall = below_element_type == WALL_ID;
        
        // Priority 1: Directly below (if not a wall)
        if (below_is_bg && !below_is_wall) {
            // Clear this location (the thread below will write the sand)
            textureStore(output, location, BACKGROUND_COLOR);
            textureStore(element_type_output, location, vec4<u32>(BACKGROUND_ID, 0u, 0u, 0u));
            return;
        }
        
        // Priority 2: Below-left (if directly below is occupied and not a wall)
        if (!below_is_wall) {
            let below_left = vec2<i32>(location.x - 1, location.y + 1);
            if (below_left.x >= 0 && below_left.y < size.y) {
                let below_left_element_type = get_element_type(below_left);
                let below_left_is_bg = below_left_element_type == BACKGROUND_ID;
                let below_left_is_wall = below_left_element_type == WALL_ID;
                
                if (below_left_is_bg && !below_left_is_wall) {
                    // Clear this location (the thread at below_left will write the sand)
                    textureStore(output, location, BACKGROUND_COLOR);
                    textureStore(element_type_output, location, vec4<u32>(BACKGROUND_ID, 0u, 0u, 0u));
                    return;
                }
            }
            
            // Priority 3: Below-right
            let below_right = vec2<i32>(location.x + 1, location.y + 1);
            if (below_right.x < size.x && below_right.y < size.y) {
                let below_right_element_type = get_element_type(below_right);
                let below_right_is_bg = below_right_element_type == BACKGROUND_ID;
                let below_right_is_wall = below_right_element_type == WALL_ID;
                
                if (below_right_is_bg && !below_right_is_wall) {
                    // Clear this location (the thread at below_right will write the sand)
                    textureStore(output, location, BACKGROUND_COLOR);
                    textureStore(element_type_output, location, vec4<u32>(BACKGROUND_ID, 0u, 0u, 0u));
                    return;
                }
            }
        }
        
        // Priority 4: Horizontal adjacent (left/right) - DISABLED for now to test
        // Horizontal movement might be causing the collapse issue
        // TODO: Re-enable with stricter conditions if needed
        
        // Sand can't move - stays in place
        let current_pixel = textureLoad(input, location);
        textureStore(output, location, current_pixel);
        textureStore(element_type_output, location, vec4<u32>(current_element_type, 0u, 0u, 0u));
        return;
    }
    
    // Current location is empty - check if sand would fall here
    // Match ProjectSandBevy's below_adjacent then adjacent priority
    // Note: If current location is a wall, we already handled it above (walls stay in place)
    
    // Priority 1: Directly above (below_adjacent checks directly below first)
    let above = vec2<i32>(location.x, location.y - 1);
    if (above.y >= 0) {
        let above_element_type = get_element_type(above);
        let above_is_sand = above_element_type == SAND_ID || above_element_type == RAINBOW_SAND_ID;
        
        if (above_is_sand) {
            // Check if the source particle (above) is affected by gravity (95% chance)
            if (should_apply_gravity(above.x, above.y, config.sim_step)) {
                // Check if space directly below the source is empty (can fall straight)
                let directly_below_above = vec2<i32>(above.x, above.y + 1);
                if (directly_below_above.y < size.y) {
                    let below_above_element_type = get_element_type(directly_below_above);
                    let below_above_is_bg = below_above_element_type == BACKGROUND_ID;
                    let below_above_is_wall = below_above_element_type == WALL_ID;
                    
                    // Only gather if space directly below source is empty (not a wall)
                    if (below_above_is_bg && !below_above_is_wall) {
                        // Check if we're at the bottom edge and fall_into_void is enabled
                        if (location.y >= size.y - 1 && fall_into_void) {
                            // Sand would fall into void - disappear (write background)
                            textureStore(output, location, BACKGROUND_COLOR);
                            textureStore(element_type_output, location, vec4<u32>(BACKGROUND_ID, 0u, 0u, 0u));
                            return;
                        }
                        // Sand falls straight down here
                        let above_pixel = textureLoad(input, above);
                        textureStore(output, location, above_pixel);
                        textureStore(element_type_output, location, vec4<u32>(above_element_type, 0u, 0u, 0u));
                        return;
                    }
                } else {
                    // Source is at bottom edge - would fall into void
                    if (fall_into_void) {
                        textureStore(output, location, BACKGROUND_COLOR);
                        textureStore(element_type_output, location, vec4<u32>(BACKGROUND_ID, 0u, 0u, 0u));
                        return;
                    }
                }
            }
        }
    }
    
    // Priority 2: Above-left (below_adjacent checks below-left second)
    // Only gather from diagonal if the source particle can't move straight down
    let above_left = vec2<i32>(location.x - 1, location.y - 1);
    if (above_left.y >= 0 && above_left.x >= 0) {
        let above_left_element_type = get_element_type(above_left);
        let above_left_is_sand = above_left_element_type == SAND_ID || above_left_element_type == RAINBOW_SAND_ID;
        
        if (above_left_is_sand) {
            // Check if the source particle is affected by gravity (95% chance)
            if (should_apply_gravity(above_left.x, above_left.y, config.sim_step)) {
                // Check if space directly below the source is occupied (forcing diagonal fall)
                let directly_below_above_left = vec2<i32>(above_left.x, above_left.y + 1);
                if (directly_below_above_left.y < size.y) {
                    let below_above_left_element_type = get_element_type(directly_below_above_left);
                    let below_above_left_is_bg = below_above_left_element_type == BACKGROUND_ID;
                    let below_above_left_is_wall = below_above_left_element_type == WALL_ID;
                    
                    // Only allow diagonal fall if:
                    // 1. Space directly below source is occupied (not background, not wall)
                    // 2. This ensures the source particle truly can't move straight down
                    if (!below_above_left_is_bg && !below_above_left_is_wall) {
                        // Additional check: make sure the source particle would actually move diagonally
                        // by verifying it's not trying to move straight down (which would be handled by Priority 1)
                        // Check if we're at the edge and fall_into_void
                        if (location.y >= size.y - 1 && fall_into_void) {
                            textureStore(output, location, BACKGROUND_COLOR);
                            textureStore(element_type_output, location, vec4<u32>(BACKGROUND_ID, 0u, 0u, 0u));
                            return;
                        }
                        // Sand falls diagonally from above-left
                        let above_left_pixel = textureLoad(input, above_left);
                        textureStore(output, location, above_left_pixel);
                        textureStore(element_type_output, location, vec4<u32>(above_left_element_type, 0u, 0u, 0u));
                        return;
                    }
                } else {
                    // Source is at bottom edge - would fall into void
                    if (fall_into_void) {
                        textureStore(output, location, BACKGROUND_COLOR);
                        textureStore(element_type_output, location, vec4<u32>(BACKGROUND_ID, 0u, 0u, 0u));
                        return;
                    }
                }
            }
        }
    }
    
    // Priority 3: Above-right (below_adjacent checks below-right third)
    // Only gather from diagonal if the source particle can't move straight down
    let above_right = vec2<i32>(location.x + 1, location.y - 1);
    if (above_right.y >= 0 && above_right.x < size.x) {
        let above_right_element_type = get_element_type(above_right);
        let above_right_is_sand = above_right_element_type == SAND_ID || above_right_element_type == RAINBOW_SAND_ID;
        
        if (above_right_is_sand) {
            // Check if the source particle is affected by gravity (95% chance)
            if (should_apply_gravity(above_right.x, above_right.y, config.sim_step)) {
                // Check if space directly below the source is occupied (forcing diagonal fall)
                let directly_below_above_right = vec2<i32>(above_right.x, above_right.y + 1);
                if (directly_below_above_right.y < size.y) {
                    let below_above_right_element_type = get_element_type(directly_below_above_right);
                    let below_above_right_is_bg = below_above_right_element_type == BACKGROUND_ID;
                    let below_above_right_is_wall = below_above_right_element_type == WALL_ID;
                    
                    // Only allow diagonal fall if:
                    // 1. Space directly below source is occupied (not background, not wall)
                    // 2. This ensures the source particle truly can't move straight down
                    if (!below_above_right_is_bg && !below_above_right_is_wall) {
                        // Additional check: make sure the source particle would actually move diagonally
                        // by verifying it's not trying to move straight down (which would be handled by Priority 1)
                        // Check if we're at the edge and fall_into_void
                        if (location.y >= size.y - 1 && fall_into_void) {
                            textureStore(output, location, BACKGROUND_COLOR);
                            textureStore(element_type_output, location, vec4<u32>(BACKGROUND_ID, 0u, 0u, 0u));
                            return;
                        }
                        // Sand falls diagonally from above-right
                        let above_right_pixel = textureLoad(input, above_right);
                        textureStore(output, location, above_right_pixel);
                        textureStore(element_type_output, location, vec4<u32>(above_right_element_type, 0u, 0u, 0u));
                        return;
                    }
                } else {
                    // Source is at bottom edge - would fall into void
                    if (fall_into_void) {
                        textureStore(output, location, BACKGROUND_COLOR);
                        textureStore(element_type_output, location, vec4<u32>(BACKGROUND_ID, 0u, 0u, 0u));
                        return;
                    }
                }
            }
        }
    }
    
    // Priority 4: Horizontal adjacent (left/right) - DISABLED for now to test
    // Horizontal movement might be causing the collapse issue
    // TODO: Re-enable with stricter conditions if needed
    
    // No sand moving here - write background
    // (We already checked current_is_bg above, so if we reach here, location is empty)
    textureStore(output, location, BACKGROUND_COLOR);
    textureStore(element_type_output, location, vec4<u32>(BACKGROUND_ID, 0u, 0u, 0u));
}

