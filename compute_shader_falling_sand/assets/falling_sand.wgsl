// The shader reads the previous frame's state from the `input` texture, and writes the new state of
// each pixel to the `output` texture. The textures are flipped each step to progress the
// simulation.
// Two textures are needed for falling sand as each pixel of step N depends on the state of its
// neighbors at step N-1.

@group(0) @binding(0) var input: texture_storage_2d<rgba32float, read>;

@group(0) @binding(1) var output: texture_storage_2d<rgba32float, write>;

@group(0) @binding(2) var<uniform> config: FallingSandUniforms;

struct FallingSandUniforms {
    sand_color: vec4<f32>,
    size: vec2<u32>,
    click_position: vec2<i32>,
    continuous_spawn: u32, // 1 = true, 0 = false
    click_radius: f32, // Radius of the circle for placing/removing sand
    click_action: u32, // 0 = no action, 1 = add sand, 2 = remove sand
}

fn is_sand(location: vec2<i32>, offset_x: i32, offset_y: i32) -> bool {
    let value: vec4<f32> = textureLoad(input, location + vec2<i32>(offset_x, offset_y));
    return value.a > 0.5;
}

fn is_empty(location: vec2<i32>, offset_x: i32, offset_y: i32) -> bool {
    return !is_sand(location, offset_x, offset_y);
}

@compute @workgroup_size(8, 8, 1)
fn init(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));
    
    // Start with empty grid
    let color = vec4(0.0, 0.0, 0.0, 0.0);
    textureStore(output, location, color);
}

@compute @workgroup_size(8, 8, 1)
fn update(@builtin(global_invocation_id) invocation_id: vec3<u32>) {
    let location = vec2<i32>(i32(invocation_id.x), i32(invocation_id.y));
    let size = vec2<i32>(i32(config.size.x), i32(config.size.y));
    
    // Check bounds
    if (location.x < 0 || location.x >= size.x || location.y < 0 || location.y >= size.y) {
        return;
    }
    
    // Handle click actions (add or remove sand in circular area)
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
                // Add sand only if the location is empty
                if (is_empty(location, 0, 0)) {
                    textureStore(output, location, vec4(config.sand_color.rgb, 1.0));
                    return;
                }
            } else if (click_action == 2u) {
                // Remove sand (clear the pixel)
                textureStore(output, location, vec4(0.0, 0.0, 0.0, 0.0));
                return;
            }
        }
    }
    
    // Spawn sand at top center (continuous, if enabled)
    if (config.continuous_spawn == 1u) {
        let center_x = size.x / 2i;
        if (location.y == 0 && location.x >= center_x - 2 && location.x <= center_x + 2) {
            let color = vec4(config.sand_color.rgb, 1.0);
            textureStore(output, location, color);
            return;
        }
    }
    
    let currently_sand = is_sand(location, 0, 0);
    let below = location + vec2<i32>(0, 1);
    
    // If this pixel is sand, check if it should fall
    if (currently_sand) {
        let current_color = textureLoad(input, location);
        
        // Check if we can fall straight down
        if (below.y < size.y && is_empty(location, 0, 1)) {
            // Sand falls down - clear this pixel (destination will handle receiving it)
            textureStore(output, location, vec4(0.0, 0.0, 0.0, 0.0));
            return;
        }
        
        // Check if we can fall diagonally (space below is occupied, but diagonal is empty)
        if (below.y < size.y && !is_empty(location, 0, 1)) {
            let below_left = location + vec2<i32>(-1, 1);
            let below_right = location + vec2<i32>(1, 1);
            
            // Try falling down-left first
            if (below_left.x >= 0 && is_empty(location, -1, 1)) {
                textureStore(output, location, vec4(0.0, 0.0, 0.0, 0.0));
                return;
            }
            
            // Try falling down-right
            if (below_right.x < size.x && is_empty(location, 1, 1)) {
                textureStore(output, location, vec4(0.0, 0.0, 0.0, 0.0));
                return;
            }
        }
        
        // Sand stays in place - preserve its color
        textureStore(output, location, vec4(current_color.rgb, 1.0));
        return;
    }
    
    // This pixel is empty - check if sand is falling into it
    // Check if sand is falling straight down from above
    let above = location + vec2<i32>(0, -1);
    if (above.y >= 0 && is_sand(location, 0, -1)) {
        // Sand above is falling down - preserve its color
        let above_color = textureLoad(input, above);
        textureStore(output, location, vec4(above_color.rgb, 1.0));
        return;
    }
    
    // Check if sand is falling diagonally into this position
    // Sand falls diagonally when: there's sand above-left/above-right,
    // the space directly below that sand is occupied, and this diagonal space is empty
    let above_left = location + vec2<i32>(-1, -1);
    if (above_left.y >= 0 && above_left.x >= 0) {
        let above_left_sand = is_sand(location, -1, -1);
        // Check if sand above-left has something directly below it (so it would fall diagonally)
        let directly_below_above_left = location + vec2<i32>(-1, 0);
        if (above_left_sand && directly_below_above_left.y < size.y && !is_empty(location, -1, 0)) {
            // Sand is falling diagonally from above-left into this position - preserve its color
            let above_left_color = textureLoad(input, above_left);
            textureStore(output, location, vec4(above_left_color.rgb, 1.0));
            return;
        }
    }
    
    let above_right = location + vec2<i32>(1, -1);
    if (above_right.y >= 0 && above_right.x < size.x) {
        let above_right_sand = is_sand(location, 1, -1);
        // Check if sand above-right has something directly below it (so it would fall diagonally)
        let directly_below_above_right = location + vec2<i32>(1, 0);
        if (above_right_sand && directly_below_above_right.y < size.y && !is_empty(location, 1, 0)) {
            // Sand is falling diagonally from above-right into this position - preserve its color
            let above_right_color = textureLoad(input, above_right);
            textureStore(output, location, vec4(above_right_color.rgb, 1.0));
            return;
        }
    }
    
    // Keep empty space empty
    textureStore(output, location, vec4(0.0, 0.0, 0.0, 0.0));
}

