#![allow(
    clippy::needless_pass_by_value,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::similar_names
)]

use crate::elements::Element;

/// Get a description for an element (for tooltips)
fn get_element_description(element: Element) -> &'static str {
    match element {
        Element::Background => "Empty space",
        Element::Wall => "Solid barrier that doesn't move",
        Element::Sand => "Falls down, sinks through liquids",
        Element::RainbowSand => "Falls like sand, with rainbow colors",
        Element::Water => "Flows and spreads, freezes into ice",
        Element::Fire => "Spreads to flammable materials, extinguished by water",
        Element::Salt => "Falls down, dissolves in water",
        Element::Oil => "Flammable liquid, floats on water",
        Element::Rock => "Heavy, sinks through liquids",
        Element::Ice => "Melts with heat, freezes water",
        Element::Lava => "Burns things, creates rock when touching water",
        Element::Steam => "Rises up, condenses to water",
        Element::SaltWater => "Water with salt, conducts electricity",
        Element::Plant => "Grows from water and soil",
        Element::Gunpowder => "Explodes when touched by fire",
        Element::Wax => "Melts with heat, burns with fire",
        Element::Concrete => "Hardens when touching water",
        Element::Nitro => "Highly explosive liquid",
        Element::Napalm => "Sticky flammable liquid",
        Element::C4 => "Powerful explosive",
        Element::Fuse => "Burns and ignites nearby explosives",
        Element::Acid => "Dissolves most materials",
        Element::Cryo => "Freezes water instantly",
        Element::Methane => "Flammable gas that rises",
        Element::Soil => "Falls down, can grow plants",
        Element::WetSoil => "Soil with water, grows plants faster",
        Element::Thermite => "Burns very hot, melts through materials",
        Element::Spout => "Sprays water upward",
        Element::Well => "Generates water",
        Element::Torch => "Burns continuously, ignites flammable materials",
        Element::Branch => "Part of tree structure",
        Element::Leaf => "Part of tree structure",
        Element::Pollen => "Light powder that floats",
        Element::FallingWax => "Wax that's falling",
        Element::ChilledIce => "Very cold ice",
        Element::Mystery => "Mysterious element with unknown properties",
        Element::ChargedNitro => "Nitro that's been charged",
        Element::BurningThermite => "Thermite that's actively burning",
    }
}
use crate::particles::{ParticleList, ParticleTexture};
use crate::particles::actions::particle_init;
use crate::simulation::{execute_element_action, GameGrid, ActiveTreeBranches};
use crate::spigots::{Spigots, NUM_SPIGOTS};
use crate::{DISPLAY_FACTOR, SIZE};
use std::collections::HashMap;
use bevy::{
    asset::RenderAssetUsages,
    input::mouse::MouseWheel,
    prelude::*,
    render::render_resource::{TextureFormat, TextureUsages},
    window::{PrimaryWindow, WindowResized},
};
use bevy_egui::{EguiContexts, egui};
use rand::Rng;

/// Resource to track the currently selected element for placement
#[derive(Resource, Clone, Copy)]
pub struct SelectedElement(pub Element);

/// Resource to track whether to overwrite existing materials when drawing
#[derive(Resource, Clone, Copy)]
pub struct OverwriteMode(pub bool);

/// Resource to track whether elements fall into the void or stop at edges
#[derive(Resource, Clone, Copy)]
pub struct FallIntoVoid(pub bool);

/// Resource to track the drawing radius
#[derive(Resource, Clone, Copy)]
pub struct DrawRadius(pub f32);

/// Resource to signal that the grid should be cleared
#[derive(Resource, Default)]
pub struct ClearGrid(pub bool);

/// Resource to signal that the grid should be saved
#[derive(Resource, Default)]
pub struct SaveGrid(pub bool);

/// Resource to signal that the grid should be loaded
#[derive(Resource, Default)]
pub struct LoadGrid(pub bool);

/// Resource to track line drawing state for shift-key straight lines
#[derive(Resource, Default)]
pub struct LineDrawingState {
    pub start_x: Option<u32>,
    pub start_y: Option<u32>,
    pub shift_pressed: bool,
}

/// Resource to track frame count for time-based effects (like rainbow sand animation)
#[derive(Resource, Default)]
pub struct FrameCount(pub u32);

/// Resource to track simulation speed (0.0 = paused, 1.0 = normal, 2.0 = 2x speed)
#[derive(Resource)]
pub struct SimulationSpeed(pub f32);

impl Default for SimulationSpeed {
    fn default() -> Self {
        Self(1.0) // Normal speed by default
    }
}

/// Resource to track placement counter for RainbowSand gradient effect
/// This increments each time RainbowSand is placed, creating a gradient over time
#[derive(Resource, Default)]
pub struct RainbowSandPlacementCounter {
    pub counter: u32,
    pub last_mouse_pressed: bool,
    pub frame_since_last_increment: u32,
}

/// Resource to track placement time for each RainbowSand element
/// Maps grid index to the placement counter value when that position was last set to RainbowSand
#[derive(Resource, Default)]
pub struct RainbowSandPlacementTimes(pub HashMap<usize, u32>);

pub fn setup(mut commands: Commands, mut image_assets: ResMut<Assets<Image>>) {
    // Create a single image for rendering (CPU-based, no double buffering needed)
    // Use Rgba8Unorm for simpler byte-based updates
    let mut image = Image::new_target_texture(SIZE.x, SIZE.y, TextureFormat::Rgba8Unorm);
    image.asset_usage = RenderAssetUsages::RENDER_WORLD;
    image.texture_descriptor.usage = TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING;
    let texture_handle = image_assets.add(image);

    commands.spawn((
        Sprite {
            image: texture_handle.clone(),
            custom_size: Some(SIZE.as_vec2()),
            ..default()
        },
        // DISPLAY_FACTOR is u32, cast to f32 for scale
        Transform::from_scale(Vec3::splat(DISPLAY_FACTOR as f32)),
    ));
    commands.spawn(Camera2d);

    // Store texture handle for rendering (will be updated each frame)
    commands.insert_resource(RenderTexture(texture_handle.clone()));

    // Initialize game grid (CPU-based simulation)
    commands.insert_resource(GameGrid::default());

    // Initialize spigots resource
    commands.insert_resource(Spigots::default());

    // Resource to track selected element (for UI)
    commands.insert_resource(SelectedElement(Element::RainbowSand));
    
    // Resource to track fall into void setting (default: true)
    commands.insert_resource(FallIntoVoid(false));
    
    // Resource to track draw radius (default: 5.0)
    commands.insert_resource(DrawRadius(5.0));
    
    // Resource to track overwrite mode (default: true, overwrite existing materials)
    commands.insert_resource(OverwriteMode(true));
    
    // Resource to signal grid clearing
    commands.insert_resource(ClearGrid::default());
    
    // Resources for save/load
    commands.insert_resource(SaveGrid::default());
    commands.insert_resource(LoadGrid::default());
    
    // Resource to track frame count for time-based effects
    commands.insert_resource(FrameCount::default());
    
    // Resource to track simulation speed (0.0 = paused, 1.0 = normal, 2.0 = 2x speed)
    commands.insert_resource(SimulationSpeed::default());
    
    // Resource to track RainbowSand placement counter for gradient effect
    commands.insert_resource(RainbowSandPlacementCounter::default());
    
    // Resource to track RainbowSand placement times
    commands.insert_resource(RainbowSandPlacementTimes::default());
    
    // Resource to track line drawing state for shift-key straight lines
    commands.insert_resource(LineDrawingState::default());
    
    // Resource to track active tree branches for incremental growth
    commands.insert_resource(ActiveTreeBranches::default());
    
    // Initialize particle system
    commands.insert_resource(ParticleList::default());
    
    // Create particle texture (offscreen canvas for particles)
    // Initialize with black pixels (transparent background)
    let particle_pixel_data = vec![0u8; (SIZE.x * SIZE.y * 4) as usize];
    let mut particle_image = Image::new_target_texture(SIZE.x, SIZE.y, TextureFormat::Rgba8Unorm);
    particle_image.data = Some(particle_pixel_data);
    particle_image.asset_usage = RenderAssetUsages::RENDER_WORLD;
    particle_image.texture_descriptor.usage = TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING;
    let particle_texture_handle = image_assets.add(particle_image);
    commands.insert_resource(ParticleTexture(particle_texture_handle));
}

/// Resource to store the render texture handle
#[derive(Resource)]
pub struct RenderTexture(pub Handle<Image>);

/// UI system for the egui controls window.
///
/// # Errors
/// Returns an error if the egui context cannot be accessed.
pub fn ui_system(
    mut contexts: EguiContexts,
    mut selected_element: ResMut<SelectedElement>,
    mut spigots: ResMut<Spigots>,
    mut fall_into_void: ResMut<FallIntoVoid>,
    mut draw_radius: ResMut<DrawRadius>,
    mut overwrite_mode: ResMut<OverwriteMode>,
    mut clear_grid: ResMut<ClearGrid>,
    mut simulation_speed: ResMut<SimulationSpeed>,
    mut save_grid: ResMut<SaveGrid>,
    mut load_grid: ResMut<LoadGrid>,
) {
    if let Ok(ctx) = contexts.ctx_mut() {
        egui::Window::new("Controls").show(ctx, |ui| {
        // Element selection
        ui.label("Selected Element:");
        ui.horizontal_wrapped(|ui| {
            for element in [Element::Sand, Element::RainbowSand, Element::Water, Element::Wall, Element::Fire, Element::Salt, Element::Oil, Element::Rock, Element::Ice, Element::Lava, Element::Steam, Element::SaltWater, Element::Plant, Element::Gunpowder, Element::Wax, Element::Concrete, Element::Nitro, Element::Napalm, Element::C4, Element::Fuse, Element::Acid, Element::Cryo, Element::Methane, Element::Soil, Element::WetSoil, Element::Thermite, Element::Spout, Element::Well, Element::Torch, Element::Branch, Element::Leaf, Element::Pollen, Element::FallingWax, Element::ChilledIce, Element::Mystery, Element::ChargedNitro, Element::BurningThermite] {
                let is_selected = selected_element.0 == element;
                let button_text = format!("{:?}", element);
                let response = ui.selectable_label(is_selected, &button_text);
                if response.clicked() {
                    selected_element.0 = element;
                }
                // Show tooltip on hover
                response.on_hover_text(get_element_description(element));
            }
        });

        ui.separator();

        // Draw radius
        ui.horizontal(|ui| {
            ui.label("Draw Radius:");
            let mut radius = draw_radius.0;
            if ui.add(egui::Slider::new(&mut radius, 1.0..=50.0)).changed() {
                draw_radius.0 = radius;
            }
        });

        ui.separator();

        // Fall into void toggle
        let mut fall_void = fall_into_void.0;
        if ui.checkbox(&mut fall_void, "Fall Into Void").changed() {
            fall_into_void.0 = fall_void;
        }
        ui.label("When enabled, elements fall off screen edges. When disabled, elements stop at edges.");

        ui.separator();

        // Overwrite mode toggle
        let mut overwrite = overwrite_mode.0;
        if ui.checkbox(&mut overwrite, "Overwrite").changed() {
            overwrite_mode.0 = overwrite;
        }
        ui.label("When enabled, drawing overwrites existing materials. When disabled, only draws on empty spaces.");

        ui.separator();

        // Simulation speed slider
        ui.horizontal(|ui| {
            ui.label("Speed:");
            let mut speed = simulation_speed.0;
            if ui.add(egui::Slider::new(&mut speed, 0.0..=2.0)).changed() {
                simulation_speed.0 = speed;
            }
            if speed == 0.0 {
                ui.label("(Paused)");
            } else {
                ui.label(format!("{:.1}x", speed));
            }
        });
        ui.label("0.0 = Paused, 1.0 = Normal Speed, 2.0 = 2x Speed");

        ui.separator();

        // Save/Load buttons
        ui.horizontal(|ui| {
            if ui.button("Save").clicked() {
                save_grid.0 = true;
            }
            if ui.button("Load").clicked() {
                load_grid.0 = true;
            }
        });

        ui.separator();

        // Clear button
        if ui.button("Clear Grid").clicked() {
            clear_grid.0 = true;
        }

        ui.separator();

        // Spigot controls
        ui.collapsing("Spigots", |ui| {
            let valid_elements = Element::spigot_valid_elements();
            let element_names: Vec<String> = valid_elements.iter().map(|e| format!("{:?}", e)).collect();
            
            for i in 0..NUM_SPIGOTS {
                ui.group(|ui| {
                    ui.label(format!("Spigot {}", i + 1));
                    
                    // Size slider (0 = disabled, 1-6 = enabled with that size)
                    let mut size = spigots.sizes[i] as f32;
                    ui.horizontal(|ui| {
                        ui.label("Size:");
                        if ui.add(egui::Slider::new(&mut size, 0.0..=6.0).step_by(1.0)).changed() {
                            spigots.sizes[i] = size as u32;
                        }
                    });
                    if spigots.sizes[i] == 0 {
                        ui.label("(Size 0 = disabled)");
                    }
                    
                    if spigots.sizes[i] > 0 {
                        // Element selection dropdown
                        let current_element = spigots.elements[i];
                        let current_idx = valid_elements
                            .iter()
                            .position(|&e| e == current_element)
                            .unwrap_or(0);
                        
                        ui.horizontal(|ui| {
                            ui.label("Element:");
                            bevy_egui::egui::ComboBox::from_id_salt(format!("spigot_{}_element", i))
                                .selected_text(&element_names[current_idx])
                                .show_ui(ui, |ui| {
                                    for (idx, element) in valid_elements.iter().enumerate() {
                                        if ui
                                            .selectable_label(idx == current_idx, &element_names[idx])
                                            .clicked()
                                        {
                                            spigots.elements[i] = *element;
                                        }
                                    }
                                });
                        });
                    }
                });
                
                if i < NUM_SPIGOTS - 1 {
                    ui.separator();
                }
            }
        });
        });
    }
}

/// Resource to track accumulated simulation frames for speed control
#[derive(Resource, Default)]
pub struct SimulationFrameAccumulator(pub f32);

/// Handle save/load operations
pub fn handle_save_load(
    mut save_grid: ResMut<SaveGrid>,
    mut load_grid: ResMut<LoadGrid>,
    grid: Res<GameGrid>,
    mut commands: Commands,
) {
    // Handle save
    if save_grid.0 {
        save_grid.0 = false;
        if let Ok(data) = bincode::serialize(&*grid) {
            if let Err(e) = std::fs::write("sand_save.bin", data) {
                bevy::log::error!("Failed to save grid: {}", e);
            } else {
                bevy::log::info!("Grid saved to sand_save.bin");
            }
        }
    }
    
    // Handle load
    if load_grid.0 {
        load_grid.0 = false;
        if let Ok(data) = std::fs::read("sand_save.bin") {
            if let Ok(loaded_grid) = bincode::deserialize::<GameGrid>(&data) {
                commands.insert_resource(loaded_grid);
                bevy::log::info!("Grid loaded from sand_save.bin");
            } else {
                bevy::log::error!("Failed to deserialize grid data");
            }
        } else {
            bevy::log::warn!("No save file found (sand_save.bin)");
        }
    }
}

/// Update the game simulation (CPU-based, ported from TypeScript)
/// Iterates bottom-to-top, zigzagging left-right/right-left
/// Speed control: accumulates frames based on speed setting, only runs when >= 1.0
pub fn update_game_simulation(
    mut grid: ResMut<GameGrid>,
    spigots: Res<Spigots>,
    fall_into_void: Res<FallIntoVoid>,
    mut active_branches: ResMut<ActiveTreeBranches>,
    mut particle_list: ResMut<ParticleList>,
    mut clear_grid: ResMut<ClearGrid>,
    mut rainbow_sand_counter: ResMut<RainbowSandPlacementCounter>,
    mut rainbow_sand_times: ResMut<RainbowSandPlacementTimes>,
    simulation_speed: Res<SimulationSpeed>,
    mut frame_accumulator: Local<SimulationFrameAccumulator>,
) {
    // Handle simulation speed: accumulate frames and only run when we've accumulated >= 1.0
    // Speed 0.0 = paused (never accumulate, never run)
    // Speed 1.0 = normal (accumulate 1.0 per frame, run every frame)
    // Speed 2.0 = 2x (accumulate 2.0 per frame, run twice per frame)
    if simulation_speed.0 <= 0.0 {
        return; // Paused
    }
    
    frame_accumulator.0 += simulation_speed.0;
    
    // Only run simulation when we've accumulated at least 1.0 frames
    // If speed > 1.0, we might run multiple times per frame
    while frame_accumulator.0 >= 1.0 {
        frame_accumulator.0 -= 1.0;
        
        // Run one frame of simulation
        run_simulation_frame(
            &mut grid,
            &spigots,
            &fall_into_void,
            &mut active_branches,
            &mut particle_list,
            &mut clear_grid,
            &mut rainbow_sand_counter,
            &mut rainbow_sand_times,
        );
    }
}

/// Run a single frame of simulation
fn run_simulation_frame(
    grid: &mut GameGrid,
    spigots: &Spigots,
    fall_into_void: &FallIntoVoid,
    active_branches: &mut ActiveTreeBranches,
    particle_list: &mut ParticleList,
    clear_grid: &mut ClearGrid,
    rainbow_sand_counter: &mut RainbowSandPlacementCounter,
    rainbow_sand_times: &mut RainbowSandPlacementTimes,
) {
    // Check if grid should be cleared
    if clear_grid.0 {
        grid.clear();
        active_branches.branches.clear();
        clear_grid.0 = false;
        // Also clear RainbowSand placement times
        rainbow_sand_times.0.clear();
    }
    
    // Process tree branches incrementally (like particle system)
    use crate::simulation::process_tree_branches;
    process_tree_branches(grid, active_branches);
    
    // Update spigots first
    update_spigots_cpu(grid, spigots, rainbow_sand_counter, rainbow_sand_times);
    

    // Iterate from bottom to top, zigzagging rows
    // This matches the TypeScript implementation
    let max_y = grid.max_y();
    let max_x = grid.max_x();
    let direction = max_y & 1; // Start direction based on bottom row
    
    for y in (0..=max_y).rev() {
        let y_parity = y & 1;
        if y_parity == direction {
            // Right to left
            for x in (0..=max_x).rev() {
                let i = grid.xy_to_index(x, y);
                let element = grid.get_index(i);
                if element == Element::Background {
                    continue; // Skip background for optimization
                }
                
                let mut times_opt = Some(&mut rainbow_sand_times.0);
                execute_element_action(grid, x, y, i, fall_into_void.0, Some(particle_list), Some(active_branches), &mut times_opt);
            }
        } else {
            // Left to right
            for x in 0..=max_x {
                let i = grid.xy_to_index(x, y);
                let element = grid.get_index(i);
                if element == Element::Background {
                    continue; // Skip background for optimization
                }
                
                let mut times_opt = Some(&mut rainbow_sand_times.0);
                execute_element_action(grid, x, y, i, fall_into_void.0, Some(particle_list), Some(active_branches), &mut times_opt);
            }
        }
    }
}

/// Update spigots (CPU version)
fn update_spigots_cpu(
    grid: &mut GameGrid,
    spigots: &Spigots,
    rainbow_sand_counter: &mut RainbowSandPlacementCounter,
    rainbow_sand_times: &mut RainbowSandPlacementTimes,
) {
    let positions = spigots.get_spigot_positions(grid.width);
    let spigot_height = 10u32; // SPIGOT_HEIGHT from TypeScript

    for (x, width, idx) in positions {
        // Size 0 means disabled, skip it
        if spigots.sizes[idx as usize] == 0 {
            continue;
        }

        let element = spigots.elements[idx as usize];
        
        // Increment RainbowSand counter every few frames for spigots
        // This ensures colors change at a moderate pace
        let current_placement_time = if element == Element::RainbowSand {
            // Increment counter every 3 frames (same as mouse placement)
            rainbow_sand_counter.frame_since_last_increment += 1;
            if rainbow_sand_counter.frame_since_last_increment >= 3 {
                rainbow_sand_counter.counter = rainbow_sand_counter.counter.wrapping_add(1);
                rainbow_sand_counter.frame_since_last_increment = 0;
            }
            Some(rainbow_sand_counter.counter)
        } else {
            None
        };
        
        // Spawn elements at the top rows with 10% chance (matching TypeScript)
        for h in 0..spigot_height.min(grid.height) {
            for w in x..(x + width).min(grid.width) {
                if rand::thread_rng().gen_bool(0.10) {
                    let spawn_y = h;
                    let spawn_idx = grid.xy_to_index(w, spawn_y);
                    grid.set_index(spawn_idx, element);
                    
                    // Store placement time for RainbowSand from spigots
                    if let Some(placement_time) = current_placement_time {
                        rainbow_sand_times.0.insert(spawn_idx, placement_time);
                    } else {
                        // Remove from placement times if not RainbowSand
                        rainbow_sand_times.0.remove(&spawn_idx);
                    }
                }
            }
        }
    }
}

/// Helper function to update a single particle, handling borrow conflicts
fn update_particle_safe(
    particle_list: &mut ParticleList,
    particle_idx: usize,
    grid: &GameGrid,
) -> bool {
    use crate::particles::actions::{particle_init, particle_action};
    
    // Initialize particle if needed (first frame)
    {
        let particle = particle_list.get_particle_mut(particle_idx);
        if let Some(particle) = particle {
            if particle.active && particle.action_iterations == 0 && !particle.reinitialized {
                particle_init(particle, grid);
                particle.reinitialized = true; // Mark as initialized
            }
        }
    }
    
    // Update particle - handle tree particles specially
    let is_tree = {
        let particle = particle_list.get_particle(particle_idx);
        particle.map(|p| p.particle_type == crate::particles::types::ParticleType::Tree && p.active).unwrap_or(false)
    };
    
    if is_tree {
        // For tree particles, we need to collect data first, then create branches
        // This avoids borrow conflicts
        
        // First, ensure particle is initialized (velocity, angle, etc.)
        {
            let particle = particle_list.get_particle_mut(particle_idx).unwrap();
            if particle.action_iterations == 0 && !particle.reinitialized {
                particle_init(particle, grid);
                particle.reinitialized = true;
            }
            // Also ensure velocity is set (might be 0 if not initialized)
            if particle.velocity == 0.0 && particle.x_velocity == 0.0 && particle.y_velocity == 0.0 {
                particle_init(particle, grid);
                particle.reinitialized = true;
            }
        }
        
        let (x, y, init_i, angle, velocity, size, generation, max_branches, branch_spacing, tree_type, branches, next_branch, iterations) = {
            let particle = particle_list.get_particle(particle_idx).unwrap();
            (
                particle.x, particle.y, particle.init_i, particle.angle, particle.velocity, particle.size,
                particle.tree_generation, particle.tree_max_branches, particle.tree_branch_spacing,
                particle.tree_type, particle.tree_branches, particle.tree_next_branch, particle.action_iterations
            )
        };
        
        // Now update the particle (move it, etc.)
        let should_remove = {
            let particle = particle_list.get_particle_mut(particle_idx).unwrap();
            // Store previous position for line drawing
            particle.prev_x = particle.x;
            particle.prev_y = particle.y;
            
            particle.action_iterations += 1;
            particle.x += particle.x_velocity;
            particle.y += particle.y_velocity;
            
            // Check if particle went off canvas
            if particle.off_canvas(grid.width as f32, grid.height as f32) {
                true
            } else {
                // Check wall collision
                let radius = particle.size / 2.0;
                let theta = particle.y_velocity.atan2(particle.x_velocity);
                let x_prime = particle.x + theta.cos() * radius;
                let y_prime = particle.y + theta.sin() * radius;
                let idx = (x_prime.round() as u32) + (y_prime.round() as u32) * grid.width;
                
                if idx < grid.elements.len() as u32 && grid.get_index(idx as usize) == crate::elements::Element::Wall {
                    true
                } else {
                    false
                }
            }
        };
        
        // Handle branch creation if needed (after releasing particle borrow)
        if let (Some(nb), Some(bs), Some(mb), Some(gen_val), Some(tt), Some(br)) = 
            (next_branch, branch_spacing, max_branches, generation, tree_type, branches) {
            let iter_val = iterations + 1;
            if iter_val >= nb && mb > 0 {
                // Create branches - we can now borrow particle_list
                let leaf_branch = br + 1 >= mb;
                let branch_angles = match tt {
                    0 => {
                        let branch_angle = std::f32::consts::PI / 8.0 + rand::thread_rng().gen_range(0.0..1.0) * std::f32::consts::PI / 4.0;
                        vec![angle + branch_angle, angle - branch_angle]
                    }
                    1 => {
                        let branch_angle = rand::thread_rng().gen_range(0.0..1.0) * std::f32::consts::PI / 16.0 + std::f32::consts::PI / 8.0;
                        vec![angle, angle + branch_angle, angle - branch_angle]
                    }
                    _ => vec![angle],
                };
                
                let spacing_factor = if tt == 0 { 0.9 } else { 0.6 };
                let new_branch_spacing = (bs as f32 * spacing_factor) as u32;
                
                for branch_angle in branch_angles {
                    if let Some(new_idx) = particle_list.add_active_particle(
                        crate::particles::types::ParticleType::Tree,
                        x, y, init_i,
                    ) {
                        if let Some(new_p) = particle_list.get_particle_mut(new_idx) {
                            new_p.tree_generation = Some(gen_val + 1);
                            new_p.tree_max_branches = Some(mb.saturating_sub(1));
                            new_p.tree_branch_spacing = Some(new_branch_spacing);
                            new_p.tree_next_branch = Some(new_branch_spacing);
                            new_p.angle = branch_angle;
                            new_p.set_velocity(velocity, branch_angle);
                            new_p.size = (size - 1.0).max(2.0);
                            new_p.tree_type = Some(tt);
                            new_p.tree_branches = Some(0);
                            if leaf_branch {
                                new_p.set_color(crate::elements::Element::Leaf);
                            }
                        }
                    }
                }
                
                // Update the original particle
                {
                    let particle = particle_list.get_particle_mut(particle_idx).unwrap();
                    let new_branches_count = br + 1;
                    particle.tree_branches = Some(new_branches_count);
                    
                    // Check if we've reached max branches (matches TypeScript: if (branches >= maxBranches))
                    if new_branches_count >= mb {
                        return true; // Remove particle - it's done growing
                    }
                    
                    let mut updated_spacing = bs;
                    if updated_spacing > 45 {
                        updated_spacing = (updated_spacing as f32 * 0.8) as u32;
                    }
                    let next_time = iter_val + (updated_spacing as f32 * (0.65 + rand::thread_rng().gen_range(0.0..1.0) * 0.35)) as u32;
                    particle.tree_next_branch = Some(next_time);
                    particle.tree_branch_spacing = Some(updated_spacing);
                }
            }
        }
        
        should_remove
    } else {
        // Non-tree particles - simple update
        let particle = particle_list.get_particle_mut(particle_idx);
        if let Some(particle) = particle {
            if !particle.active {
                false
            } else {
                particle_action(particle, None, particle_idx, grid)
            }
        } else {
            false
        }
    }
}

/// Update particles each frame
pub fn update_particles(
    mut particle_list: ResMut<ParticleList>,
    grid: Res<GameGrid>,
) {
    // Get active particle indices (clone to avoid borrow issues)
    let active_indices: Vec<usize> = particle_list.active_particles().to_vec();
    
    // Update each active particle
    for particle_idx in active_indices {
        // Initialize particle if needed (first frame)
        {
            let particle = particle_list.get_particle_mut(particle_idx);
            if let Some(particle) = particle {
                if particle.active && particle.action_iterations == 0 && !particle.reinitialized {
                    particle_init(particle, &grid);
                    particle.reinitialized = true; // Mark as initialized
                }
            }
        }
        
        // Update particle using helper function
        let should_remove = update_particle_safe(&mut *particle_list, particle_idx, &grid);
        
        if should_remove {
            particle_list.make_particle_inactive(particle_idx);
        }
    }
}

/// Render particles to particle texture
pub fn render_particles(
    particle_list: Res<ParticleList>,
    grid: Res<GameGrid>,
    mut images: ResMut<Assets<Image>>,
    mut particle_texture: ResMut<ParticleTexture>,
) {
    use crate::particles::render::render_particles_to_texture;
    render_particles_to_texture(particle_list, grid, images, particle_texture);
}

/// Composite particles onto main texture
pub fn composite_particles(
    grid: Res<GameGrid>,
    particle_list: Res<ParticleList>,
    mut images: ResMut<Assets<Image>>,
    mut particle_texture: ResMut<ParticleTexture>,
    render_texture: Res<RenderTexture>,
) {
    use crate::particles::render::composite_particles_to_main;
    composite_particles_to_main(grid, particle_list, images, particle_texture, render_texture);
}

/// Render the game grid to the texture
pub fn render_grid_to_texture(
    grid: Res<GameGrid>,
    rainbow_sand_times: Res<RainbowSandPlacementTimes>,
    mut images: ResMut<Assets<Image>>,
    mut sprite_query: Query<&mut Sprite, Without<Camera>>,
    mut render_texture: ResMut<RenderTexture>,
) {
    // Create pixel data from grid
    // Rgba8Unorm format: 4 u8 values per pixel (4 bytes per pixel)
    let mut pixel_data = Vec::with_capacity((grid.width * grid.height * 4) as usize);
    
    for (idx, element) in grid.elements.iter().enumerate() {
        let color = if *element == Element::RainbowSand {
            // RainbowSand: use placement time to determine color
            // The color is determined when placed and stays fixed
            // Get the placement time for this position, or use position-based hash as fallback
            let placement_time = rainbow_sand_times.0.get(&idx).copied();
            let (x, y) = grid.index_to_xy(idx);
            
            let placement_time = placement_time.unwrap_or_else(|| {
                // Fallback: if no placement time found, use position hash
                // This handles cases where sand moved and we lost the placement time
                (x.wrapping_mul(73856093)).wrapping_add(y.wrapping_mul(19349663)) as u32
            });
            
            // Use placement time to create color shift across full 360 degree hue range
            // Use modulo 256 to get full u8 range, which will be mapped to 0-360 degrees
            let shift = (placement_time % 256) as u8;
            
            element.to_encoded_color_with_shift(shift)
        } else {
            // Normal elements: no color shift
            element.to_encoded_color()
        };
        
        // Convert LinearRgba to u8 values (Rgba8Unorm format)
        pixel_data.push((color.red * 255.0).clamp(0.0, 255.0) as u8);
        pixel_data.push((color.green * 255.0).clamp(0.0, 255.0) as u8);
        pixel_data.push((color.blue * 255.0).clamp(0.0, 255.0) as u8);
        pixel_data.push((color.alpha * 255.0).clamp(0.0, 255.0) as u8);
    }

    // Create a new image from the pixel data each frame
    // This forces Bevy to recognize the change
    let mut new_image = Image::new_target_texture(grid.width, grid.height, TextureFormat::Rgba8Unorm);
    new_image.data = Some(pixel_data);
    new_image.asset_usage = RenderAssetUsages::RENDER_WORLD;
    new_image.texture_descriptor.usage = TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING;
    
    // Add the new image and update the sprite
    let new_handle = images.add(new_image);
    
    // Update the RenderTexture resource so composite_particles can use it
    render_texture.0 = new_handle.clone();
    
    // Update the sprite to use the new image handle
    // This forces Bevy to re-render with the new data
    for mut sprite in sprite_query.iter_mut() {
        sprite.image = new_handle.clone();
    }
}

/// Handle mouse scroll to adjust draw radius
pub fn handle_mouse_scroll(
    mut draw_radius: ResMut<DrawRadius>,
    mut scroll_evr: bevy::prelude::MessageReader<MouseWheel>,
    egui_contexts: Option<EguiContexts>,
) {
    // Don't process scroll if egui is consuming the input
    if let Some(mut contexts) = egui_contexts {
        if let Ok(ctx) = contexts.ctx_mut()
            && (ctx.wants_pointer_input() || ctx.is_pointer_over_area())
        {
            return;
        }
    }

    let mut total_scroll = 0.0;
    for ev in scroll_evr.read() {
        total_scroll += ev.y;
    }

    if total_scroll != 0.0 {
        // Adjust radius: scroll up increases, scroll down decreases
        // Clamp between 1.0 and 50.0
        draw_radius.0 = (draw_radius.0 + total_scroll * 0.5).clamp(1.0, 50.0);
    }
}

/// Draw circle outline to show where material will be placed
pub fn draw_circle_preview(
    mut gizmos: Gizmos,
    draw_radius: Res<DrawRadius>,
    windows: Query<&Window, With<PrimaryWindow>>,
    camera_query: Query<(&Camera, &GlobalTransform), With<Camera2d>>,
    egui_contexts: Option<EguiContexts>,
) {
    // Don't draw if egui is consuming the input
    if let Some(mut contexts) = egui_contexts {
        if let Ok(ctx) = contexts.ctx_mut()
            && (ctx.wants_pointer_input() || ctx.is_pointer_over_area())
        {
            return;
        }
    }

    let Ok(window) = windows.single() else {
        return;
    };

    let Some(cursor_position) = window.cursor_position() else {
        return;
    };

    let Ok((camera, camera_transform)) = camera_query.single() else {
        return;
    };

    // Convert screen coordinates to world coordinates
    let Ok(world_pos) = camera.viewport_to_world_2d(camera_transform, cursor_position) else {
        return;
    };

    // Draw circle outline at cursor position
    // Convert radius from texture space to world space
    let world_radius = draw_radius.0 * DISPLAY_FACTOR as f32;
    gizmos.circle_2d(world_pos, world_radius, Color::WHITE);
}

/// Handle mouse clicks for drawing (CPU version)
pub fn handle_mouse_clicks_cpu(
    mut grid: ResMut<GameGrid>,
    selected_element: Res<SelectedElement>,
    draw_radius: Res<DrawRadius>,
    overwrite_mode: Res<OverwriteMode>,
    mut rainbow_sand_counter: ResMut<RainbowSandPlacementCounter>,
    mut rainbow_sand_times: ResMut<RainbowSandPlacementTimes>,
    mouse_button_input: Res<ButtonInput<MouseButton>>,
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut line_state: ResMut<LineDrawingState>,
    windows: Query<&Window, With<PrimaryWindow>>,
    camera_query: Query<(&Camera, &GlobalTransform), With<Camera2d>>,
    egui_contexts: Option<EguiContexts>,
) {
    // Don't process clicks if egui is consuming the input
    if let Some(mut contexts) = egui_contexts {
        if let Ok(ctx) = contexts.ctx_mut()
            && (ctx.wants_pointer_input() || ctx.is_pointer_over_area())
        {
            return;
        }
    }

    let Ok(window) = windows.single() else {
        return;
    };

    let Some(cursor_position) = window.cursor_position() else {
        return;
    };

    let Ok((camera, camera_transform)) = camera_query.single() else {
        return;
    };

    // Convert screen coordinates to world coordinates
    let Ok(world_pos) = camera.viewport_to_world_2d(camera_transform, cursor_position) else {
        return;
    };

    // Convert world coordinates to grid coordinates
    let display_factor_f32 = DISPLAY_FACTOR as f32;
    let size_x_f32 = grid.width as f32;
    let size_y_f32 = grid.height as f32;
    let grid_x = ((world_pos.x / display_factor_f32) + size_x_f32 / 2.0)
        .clamp(0.0, size_x_f32 - 1.0) as u32;
    let normalized_y = (world_pos.y / display_factor_f32) + size_y_f32 / 2.0;
    let grid_y = (size_y_f32 - 1.0 - normalized_y).clamp(0.0, size_y_f32 - 1.0) as u32;

    if grid_x >= grid.width || grid_y >= grid.height {
        return;
    }

    // Check if shift is pressed
    let shift_pressed = keyboard_input.pressed(KeyCode::ShiftLeft) || keyboard_input.pressed(KeyCode::ShiftRight);
    line_state.shift_pressed = shift_pressed;
    
    // Draw circle of elements
    let radius = draw_radius.0;
    let radius_sq = radius * radius;

    if mouse_button_input.pressed(MouseButton::Left) {
        // Handle shift-key straight line drawing
        if shift_pressed {
            // Store start position on first click
            if line_state.start_x.is_none() {
                line_state.start_x = Some(grid_x);
                line_state.start_y = Some(grid_y);
            }
            
            // Draw line from start to current position
            if let (Some(start_x), Some(start_y)) = (line_state.start_x, line_state.start_y) {
                draw_line(
                    &mut grid,
                    start_x,
                    start_y,
                    grid_x,
                    grid_y,
                    radius,
                    selected_element.0,
                    overwrite_mode.0,
                    &mut rainbow_sand_counter,
                    &mut rainbow_sand_times,
                );
            }
        } else {
            // Normal circle drawing
            line_state.start_x = None;
            line_state.start_y = None;
            
            // Increment RainbowSand counter every few frames while placing
            // This ensures colors change at a moderate pace, creating visible gradients
            let current_placement_time = if selected_element.0 == Element::RainbowSand {
                // Increment counter every 3 frames while placing (faster than before)
                rainbow_sand_counter.frame_since_last_increment += 1;
                if rainbow_sand_counter.frame_since_last_increment >= 3 {
                    rainbow_sand_counter.counter = rainbow_sand_counter.counter.wrapping_add(1);
                    rainbow_sand_counter.frame_since_last_increment = 0;
                }
                rainbow_sand_counter.last_mouse_pressed = true;
                Some(rainbow_sand_counter.counter)
            } else {
                rainbow_sand_counter.last_mouse_pressed = true;
                None
            };
            
            // Add elements
            for dy in -(radius as i32)..=(radius as i32) {
                for dx in -(radius as i32)..=(radius as i32) {
                    let dist_sq = (dx * dx + dy * dy) as f32;
                    if dist_sq <= radius_sq {
                        let x = (grid_x as i32 + dx).max(0).min(grid.width as i32 - 1) as u32;
                        let y = (grid_y as i32 + dy).max(0).min(grid.height as i32 - 1) as u32;
                        
                        // Check overwrite mode: if disabled, only draw on empty spaces
                        if overwrite_mode.0 || grid.get(x, y) == Element::Background {
                            let idx = grid.xy_to_index(x, y);
                            grid.set(x, y, selected_element.0);
                            
                            // Store placement time for RainbowSand
                            if let Some(placement_time) = current_placement_time {
                                rainbow_sand_times.0.insert(idx, placement_time);
                            } else {
                                // Remove from placement times if not RainbowSand
                                rainbow_sand_times.0.remove(&idx);
                            }
                        }
                    }
                }
            }
        }
    } else if mouse_button_input.pressed(MouseButton::Right) {
        // Remove elements (set to background)
        for dy in -(radius as i32)..=(radius as i32) {
            for dx in -(radius as i32)..=(radius as i32) {
                let dist_sq = (dx * dx + dy * dy) as f32;
                if dist_sq <= radius_sq {
                    let x = (grid_x as i32 + dx).max(0).min(grid.width as i32 - 1) as u32;
                    let y = (grid_y as i32 + dy).max(0).min(grid.height as i32 - 1) as u32;
                    grid.set(x, y, Element::Background);
                }
            }
        }
    } else {
        // Reset mouse pressed state when button is released
        rainbow_sand_counter.last_mouse_pressed = false;
        rainbow_sand_counter.frame_since_last_increment = 0;
        // Reset line drawing state
        line_state.start_x = None;
        line_state.start_y = None;
    }
}

/// Draw a line between two points using Bresenham's line algorithm
fn draw_line(
    grid: &mut GameGrid,
    x0: u32,
    y0: u32,
    x1: u32,
    y1: u32,
    radius: f32,
    element: Element,
    overwrite: bool,
    rainbow_sand_counter: &mut RainbowSandPlacementCounter,
    rainbow_sand_times: &mut RainbowSandPlacementTimes,
) {
    let dx = (x1 as i32 - x0 as i32).abs();
    let dy = (y1 as i32 - y0 as i32).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx - dy;
    
    let mut x = x0 as i32;
    let mut y = y0 as i32;
    let radius_sq = radius * radius;
    
    // Get placement time for RainbowSand
    let current_placement_time = if element == Element::RainbowSand {
        rainbow_sand_counter.frame_since_last_increment += 1;
        if rainbow_sand_counter.frame_since_last_increment >= 3 {
            rainbow_sand_counter.counter = rainbow_sand_counter.counter.wrapping_add(1);
            rainbow_sand_counter.frame_since_last_increment = 0;
        }
        Some(rainbow_sand_counter.counter)
    } else {
        None
    };
    
    loop {
        // Draw circle at each point along the line
        for dy in -(radius as i32)..=(radius as i32) {
            for dx in -(radius as i32)..=(radius as i32) {
                let dist_sq = (dx * dx + dy * dy) as f32;
                if dist_sq <= radius_sq {
                    let px = (x + dx).max(0).min(grid.width as i32 - 1) as u32;
                    let py = (y + dy).max(0).min(grid.height as i32 - 1) as u32;
                    
                    if overwrite || grid.get(px, py) == Element::Background {
                        let idx = grid.xy_to_index(px, py);
                        grid.set(px, py, element);
                        
                        if let Some(placement_time) = current_placement_time {
                            rainbow_sand_times.0.insert(idx, placement_time);
                        } else {
                            rainbow_sand_times.0.remove(&idx);
                        }
                    }
                }
            }
        }
        
        if x == x1 as i32 && y == y1 as i32 {
            break;
        }
        
        let e2 = 2 * err;
        if e2 > -dy {
            err -= dy;
            x += sx;
        }
        if e2 < dx {
            err += dx;
            y += sy;
        }
    }
}

/// Handle window resize events - resize grid and clear it
pub fn handle_window_resize(
    mut resize_events: MessageReader<WindowResized>,
    mut grid: ResMut<GameGrid>,
    mut images: ResMut<Assets<Image>>,
    mut render_texture: ResMut<RenderTexture>,
    mut particle_texture: ResMut<ParticleTexture>,
    mut sprite_query: Query<&mut Sprite>,
    mut rainbow_sand_times: ResMut<RainbowSandPlacementTimes>,
) {
    for event in resize_events.read() {
        // Calculate new grid size based on window size and display factor
        let new_width = (event.width / DISPLAY_FACTOR as f32) as u32;
        let new_height = (event.height / DISPLAY_FACTOR as f32) as u32;
        
        // Only resize if the size actually changed
        if new_width != grid.width || new_height != grid.height {
            // Resize and clear the grid
            *grid = GameGrid::new(new_width, new_height);
            
            // Clear RainbowSand placement times
            rainbow_sand_times.0.clear();
            
            // Resize render texture
            if let Some(image) = images.get_mut(&render_texture.0) {
                let mut new_image = Image::new_target_texture(new_width, new_height, TextureFormat::Rgba8Unorm);
                new_image.asset_usage = RenderAssetUsages::RENDER_WORLD;
                new_image.texture_descriptor.usage = TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING;
                *image = new_image;
            }
            
            // Resize particle texture
            if let Some(image) = images.get_mut(&particle_texture.0) {
                let particle_pixel_data = vec![0u8; (new_width * new_height * 4) as usize];
                let mut new_particle_image = Image::new_target_texture(new_width, new_height, TextureFormat::Rgba8Unorm);
                new_particle_image.data = Some(particle_pixel_data);
                new_particle_image.asset_usage = RenderAssetUsages::RENDER_WORLD;
                new_particle_image.texture_descriptor.usage = TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING;
                *image = new_particle_image;
            }
            
            // Update sprite size
            for mut sprite in sprite_query.iter_mut() {
                sprite.custom_size = Some(bevy::math::UVec2::new(new_width, new_height).as_vec2());
            }
        }
    }
}
