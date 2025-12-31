use bevy::prelude::*;
use crate::particles::types::{Particle, PAINTABLE_PARTICLE_COLORS};
use crate::particles::manager::ParticleList;
use crate::elements::Element;

/// Resource to store the particle texture handle
#[derive(Resource)]
pub struct ParticleTexture(pub Handle<Image>);

/// Render particles to the particle texture
/// Particles are drawn as circles or lines depending on type
pub fn render_particles_to_texture(
    particle_list: Res<ParticleList>,
    grid: Res<crate::simulation::GameGrid>,
    mut images: ResMut<Assets<Image>>,
    mut particle_texture: ResMut<ParticleTexture>,
) {
    // Create pixel data for particle texture (transparent black background)
    // IMPORTANT: We need to clear the texture each frame, otherwise old particles will remain
    let mut particle_pixels = vec![0u8; (grid.width * grid.height * 4) as usize];
    
    // Draw each active particle
    for &particle_idx in particle_list.active_particles() {
        if let Some(particle) = particle_list.get_particle(particle_idx) {
            if particle.particle_type == crate::particles::types::ParticleType::Tree {
                // For tree particles, draw line from previous position to current
                if particle.prev_x >= 0.0 && particle.prev_y >= 0.0 {
                    draw_line(particle.prev_x, particle.prev_y, particle.x, particle.y, particle.size, &mut particle_pixels, grid.width, grid.height, particle.color);
                } else {
                    // First frame - just draw a circle
                    draw_circle_helper(particle.x, particle.y, particle.size, &mut particle_pixels, grid.width, grid.height, particle.color);
                }
            } else if particle.particle_type == crate::particles::types::ParticleType::ChargedNitro {
                // ChargedNitro particles draw a vertical fire column from init position to current position
                // This creates the upward fire column effect
                draw_line(particle.init_x, particle.init_y, particle.x, particle.y, particle.size, &mut particle_pixels, grid.width, grid.height, particle.color);
            } else {
                draw_particle(particle, &mut particle_pixels, grid.width, grid.height);
            }
        }
    }
    
    // Update particle texture using the same pattern as main render texture
    // Create a new Image each frame to force Bevy to re-upload the texture
    let mut new_particle_image = Image::new_target_texture(grid.width, grid.height, bevy::render::render_resource::TextureFormat::Rgba8Unorm);
    new_particle_image.data = Some(particle_pixels);
    new_particle_image.asset_usage = bevy::asset::RenderAssetUsages::RENDER_WORLD;
    new_particle_image.texture_descriptor.usage = bevy::render::render_resource::TextureUsages::COPY_DST | bevy::render::render_resource::TextureUsages::TEXTURE_BINDING;
    
    // Add the new image and update the resource handle (same pattern as main render texture)
    let new_handle = images.add(new_particle_image);
    particle_texture.0 = new_handle;
}

/// Draw a single particle to the pixel buffer
fn draw_particle(particle: &Particle, pixels: &mut [u8], width: u32, height: u32) {
    let color = particle.color.color();
    let r = (color.red * 255.0) as u8;
    let g = (color.green * 255.0) as u8;
    let b = (color.blue * 255.0) as u8;
    let a = (color.alpha * 255.0) as u8;
    
    match particle.particle_type {
        crate::particles::types::ParticleType::Nitro
        | crate::particles::types::ParticleType::Lava
        | crate::particles::types::ParticleType::Magic1
        | crate::particles::types::ParticleType::ChargedNitro
        | crate::particles::types::ParticleType::Tree => {
            // Draw as line (from previous position to current)
            // For tree particles, we need to track previous position
            // For now, draw as circle at current position (will be improved)
            draw_circle_internal(particle.x, particle.y, particle.size, r, g, b, a, pixels, width, height);
        }
        crate::particles::types::ParticleType::Napalm
        | crate::particles::types::ParticleType::C4
        | crate::particles::types::ParticleType::Methane
        | crate::particles::types::ParticleType::Nuke => {
            // Draw as circle
            draw_circle_internal(particle.x, particle.y, particle.size, r, g, b, a, pixels, width, height);
        }
        crate::particles::types::ParticleType::Magic2 => {
            // Draw as line for spiral
            // For simplicity, draw as small circle
            draw_circle_internal(particle.x, particle.y, particle.size, r, g, b, a, pixels, width, height);
        }
        _ => {
            // Default: draw as circle
            draw_circle_internal(particle.x, particle.y, particle.size, r, g, b, a, pixels, width, height);
        }
    }
}

/// Draw a line from (x1, y1) to (x2, y2) with given width
fn draw_line(x1: f32, y1: f32, x2: f32, y2: f32, width: f32, pixels: &mut [u8], canvas_width: u32, canvas_height: u32, color: Element) {
    // Simple line drawing using Bresenham-like algorithm
    let dx = x2 - x1;
    let dy = y2 - y1;
    let dist = (dx * dx + dy * dy).sqrt();
    
    if dist < 0.1 {
        // Points are too close, just draw a circle
        draw_circle_helper(x1, y1, width, pixels, canvas_width, canvas_height, color);
        return;
    }
    
    // Draw line by drawing circles along the path
    // Use enough steps to ensure continuous coverage (at least 2 pixels per step)
    let min_steps = (dist / (width / 2.0)).ceil().max(2.0) as usize;
    for i in 0..=min_steps {
        let t = if min_steps > 0 { i as f32 / min_steps as f32 } else { 0.0 };
        let x = x1 + dx * t;
        let y = y1 + dy * t;
        // Draw circles with radius = width/2 to create a continuous line
        draw_circle_helper(x, y, width / 2.0, pixels, canvas_width, canvas_height, color);
    }
}

/// Draw a circle helper that takes Element color
fn draw_circle_helper(x: f32, y: f32, radius: f32, pixels: &mut [u8], canvas_width: u32, canvas_height: u32, color: Element) {
    let color_rgba = color.color();
    let r = (color_rgba.red * 255.0) as u8;
    let g = (color_rgba.green * 255.0) as u8;
    let b = (color_rgba.blue * 255.0) as u8;
    let a = (color_rgba.alpha * 255.0) as u8;
    draw_circle_internal(x, y, radius, r, g, b, a, pixels, canvas_width, canvas_height);
}

/// Draw a filled circle at the given position (internal helper)
fn draw_circle_internal(
    x: f32,
    y: f32,
    radius: f32,
    r: u8,
    g: u8,
    b: u8,
    a: u8,
    pixels: &mut [u8],
    width: u32,
    height: u32,
) {
    let radius_sq = radius * radius;
    let x_center = x.round() as i32;
    let y_center = y.round() as i32;
    let radius_int = radius.ceil() as i32;
    
    // Draw circle
    for dy in -radius_int..=radius_int {
        for dx in -radius_int..=radius_int {
            let dist_sq = (dx * dx + dy * dy) as f32;
            if dist_sq <= radius_sq {
                let px = x_center + dx;
                let py = y_center + dy;
                
                if px >= 0 && px < width as i32 && py >= 0 && py < height as i32 {
                    let idx = ((py as u32 * width + px as u32) * 4) as usize;
                    if idx + 3 < pixels.len() {
                        pixels[idx] = r;
                        pixels[idx + 1] = g;
                        pixels[idx + 2] = b;
                        pixels[idx + 3] = a;
                    }
                }
            }
        }
    }
}

/// Composite particle texture onto main texture
/// Only copies pixels that match paintable particle colors
pub fn composite_particles_to_main(
    grid: Res<crate::simulation::GameGrid>,
    _particle_list: Res<ParticleList>,
    mut images: ResMut<Assets<Image>>,
    particle_texture: ResMut<ParticleTexture>,
    render_texture: Res<crate::systems::RenderTexture>,
) {
    // Get particle texture data (clone to avoid borrow issues)
    let particle_data = {
        if let Some(particle_image) = images.get(&particle_texture.0) {
            if let Some(data) = particle_image.data.as_ref() {
                data.clone()
            } else {
                return; // No particle data yet
            }
        } else {
            return; // Particle texture not found
        }
    };
    
    // Get main texture
    let (main_data, width, height) = {
        if let Some(main_image) = images.get_mut(&render_texture.0) {
            if let Some(data) = main_image.data.as_mut() {
                (data, grid.width, grid.height)
            } else {
                return; // No main texture data yet
            }
        } else {
            return; // Main texture not found
        }
    };
    
    // Composite particles onto main texture
    // Only copy pixels that match paintable colors
    for y in 0..height {
        for x in 0..width {
            let idx = ((y * width + x) * 4) as usize;
            if idx + 3 >= particle_data.len() || idx + 3 >= main_data.len() {
                continue;
            }
            
            // Check if particle pixel is non-black
            let pr = particle_data[idx];
            let pg = particle_data[idx + 1];
            let pb = particle_data[idx + 2];
            let pa = particle_data[idx + 3];
            
            if pr == 0 && pg == 0 && pb == 0 && pa == 0 {
                continue; // Skip black pixels (background)
            }
            
            // Check if color matches a paintable particle color
            // For simplicity, check if it matches any element color closely
            let mut matches = false;
            for &color_elem in PAINTABLE_PARTICLE_COLORS {
                let color = color_elem.color();
                let cr = (color.red * 255.0) as u8;
                let cg = (color.green * 255.0) as u8;
                let cb = (color.blue * 255.0) as u8;
                
                // Allow some tolerance for anti-aliasing (increased from 10 to 20)
                if (pr as i16 - cr as i16).abs() < 20
                    && (pg as i16 - cg as i16).abs() < 20
                    && (pb as i16 - cb as i16).abs() < 20
                {
                    matches = true;
                    break;
                }
            }
            
            if matches {
                // Copy particle pixel to main texture
                main_data[idx] = pr;
                main_data[idx + 1] = pg;
                main_data[idx + 2] = pb;
                main_data[idx + 3] = pa;
            } else {
                // Try to find nearby valid color (anti-aliasing fix)
                let aliasing_search = 3;
                let mut found_color = None;
                
                // Search left
                if x >= aliasing_search {
                    let search_idx = (((y * width + (x - aliasing_search)) * 4)) as usize;
                    if search_idx + 3 < particle_data.len() {
                        found_color = check_paintable_color(
                            &particle_data[search_idx..search_idx + 4],
                        );
                    }
                }
                
                // Search right
                if found_color.is_none() && x + aliasing_search < width {
                    let search_idx = (((y * width + (x + aliasing_search)) * 4)) as usize;
                    if search_idx + 3 < particle_data.len() {
                        found_color = check_paintable_color(
                            &particle_data[search_idx..search_idx + 4],
                        );
                    }
                }
                
                // Search up
                if found_color.is_none() && y >= aliasing_search {
                    let search_idx = ((((y - aliasing_search) * width + x) * 4)) as usize;
                    if search_idx + 3 < particle_data.len() {
                        found_color = check_paintable_color(
                            &particle_data[search_idx..search_idx + 4],
                        );
                    }
                }
                
                // Search down
                if found_color.is_none() && y + aliasing_search < height {
                    let search_idx = ((((y + aliasing_search) * width + x) * 4)) as usize;
                    if search_idx + 3 < particle_data.len() {
                        found_color = check_paintable_color(
                            &particle_data[search_idx..search_idx + 4],
                        );
                    }
                }
                
                if let Some((r, g, b, a)) = found_color {
                    main_data[idx] = r;
                    main_data[idx + 1] = g;
                    main_data[idx + 2] = b;
                    main_data[idx + 3] = a;
                }
            }
        }
    }
    
}

/// Check if a pixel color matches a paintable particle color
fn check_paintable_color(pixel: &[u8]) -> Option<(u8, u8, u8, u8)> {
    if pixel.len() < 4 {
        return None;
    }
    
    let pr = pixel[0];
    let pg = pixel[1];
    let pb = pixel[2];
    let pa = pixel[3];
    
    if pr == 0 && pg == 0 && pb == 0 {
        return None; // Black background
    }
    
    // Check against paintable colors
    for &color_elem in PAINTABLE_PARTICLE_COLORS {
        let color = color_elem.color();
        let cr = (color.red * 255.0) as u8;
        let cg = (color.green * 255.0) as u8;
        let cb = (color.blue * 255.0) as u8;
        
        // Allow tolerance for anti-aliasing
        if (pr as i16 - cr as i16).abs() < 10
            && (pg as i16 - cg as i16).abs() < 10
            && (pb as i16 - cb as i16).abs() < 10
        {
            return Some((cr, cg, cb, pa));
        }
    }
    
    None
}

