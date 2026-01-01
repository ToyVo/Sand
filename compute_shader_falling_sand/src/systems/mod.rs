#![allow(
    clippy::needless_pass_by_value,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::similar_names
)]

use crate::plugins::{
    FallingSandImageBindGroups, FallingSandImages, FallingSandPipeline, FallingSandUniforms, NUM_SPIGOTS,
};
use crate::{DISPLAY_FACTOR, SHADER_ASSET_PATH, SIZE};
use bevy::{
    asset::RenderAssetUsages, render::render_resource::TextureUsages, window::PrimaryWindow,
};
use bevy::{
    prelude::*,
    render::{
        render_asset::RenderAssets,
        render_resource::{
            BindGroupEntries, BindGroupLayoutEntries, ComputePipelineDescriptor, PipelineCache,
            ShaderStages, StorageTextureAccess, TextureFormat, UniformBuffer,
            binding_types::{texture_storage_2d, uniform_buffer},
        },
        renderer::{RenderDevice, RenderQueue},
        texture::GpuImage,
    },
};
use bevy_egui::{EguiContexts, egui};
use std::borrow::Cow;
use bevy::render::extract_resource::ExtractResource;
use crate::elements::Element;

/// Resource to signal that the grid should be cleared
#[derive(Resource, Default, Clone, Copy, ExtractResource)]
pub struct ClearGrid(pub bool);

/// Resource to track simulation speed (0.0 = paused, 1.0 = normal, 2.0 = 2x speed)
#[derive(Resource, Clone, Copy, ExtractResource)]
pub struct SimulationSpeed(pub f32);

/// Resource to track accumulated simulation frames for speed control
#[derive(Resource, Default)]
pub struct SimulationFrameAccumulator(pub f32);

impl Default for SimulationSpeed {
    fn default() -> Self {
        Self(1.0) // Normal speed by default
    }
}

/// Resource to track whether to overwrite existing materials when drawing
#[derive(Resource, Clone, Copy)]
pub struct OverwriteMode(pub bool);

impl Default for OverwriteMode {
    fn default() -> Self {
        Self(true) // Overwrite by default
    }
}

/// Resource to track whether elements fall into the void or stop at edges
#[derive(Resource, Clone, Copy)]
pub struct FallIntoVoid(pub bool);

impl Default for FallIntoVoid {
    fn default() -> Self {
        Self(false) // Don't fall into void by default
    }
}

pub fn setup(mut commands: Commands, mut image_assets: ResMut<Assets<Image>>) {
    // Color textures (rgba32float)
    let mut image = Image::new_target_texture(SIZE.x, SIZE.y, TextureFormat::Rgba32Float);
    image.asset_usage = RenderAssetUsages::RENDER_WORLD;
    image.texture_descriptor.usage =
        TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;
    let texture_a_handle = image_assets.add(image.clone());
    let texture_b_handle = image_assets.add(image);
    
    // Element type textures (r32uint - stores element type ID: 0=background, 1=wall, 2=sand, 3=rainbow_sand)
    let mut element_type_image = Image::new_target_texture(SIZE.x, SIZE.y, TextureFormat::R32Uint);
    element_type_image.asset_usage = RenderAssetUsages::RENDER_WORLD;
    element_type_image.texture_descriptor.usage =
        TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;
    let element_type_a_handle = image_assets.add(element_type_image.clone());
    let element_type_b_handle = image_assets.add(element_type_image);

    commands.spawn((
        Sprite {
            image: texture_a_handle.clone(),
            custom_size: Some(SIZE.as_vec2()),
            ..default()
        },
        // DISPLAY_FACTOR is u32, cast to f32 for scale
        Transform::from_scale(Vec3::splat(DISPLAY_FACTOR as f32)),
    ));
    commands.spawn(Camera2d);

    commands.insert_resource(FallingSandImages {
        texture_a: texture_a_handle,
        texture_b: texture_b_handle,
        element_type_a: element_type_a_handle,
        element_type_b: element_type_b_handle,
    });

    commands.insert_resource(FallingSandUniforms {
        sand_color: LinearRgba::rgb(0.76, 0.70, 0.50), // Single sandy color for all regular sand
        size: SIZE,
        click_position: IVec2::new(-1, -1), // -1 means no click
        spigot_sizes: UVec4::new(3, 3, 3, 3).into(),
        spigot_elements: UVec4::new(Element::RainbowSand as u32, Element::RainbowSand as u32, Element::RainbowSand as u32, Element::RainbowSand as u32).into(),
        click_radius: 5.0,                  // Default radius
        click_action: 0,                    // 0 = no action
        selected_element: 0,                 // 0 = sand
        wall_color: LinearRgba::rgb(0.5, 0.5, 0.5), // Gray wall color
        color_shift_enabled: 1,             // Enabled by default
        sim_step: 0,                        // Start at step 0
        overwrite_mode: 1,                  // Overwrite by default
        fall_into_void: 0,                  // Don't fall into void by default
    });
    
    // Initialize simulation speed and clear grid resources
    commands.insert_resource(SimulationSpeed::default());
    commands.insert_resource(ClearGrid::default());
    commands.insert_resource(OverwriteMode::default());
    commands.insert_resource(FallIntoVoid::default());
    commands.insert_resource(Element::RainbowSand);
}

/// UI system for the egui controls window.
///
/// # Errors
/// Returns an error if the egui context cannot be accessed.
pub fn ui_system(
    mut contexts: EguiContexts,
    mut uniforms: ResMut<FallingSandUniforms>,
    mut clear_grid: ResMut<ClearGrid>,
    mut simulation_speed: ResMut<SimulationSpeed>,
    mut overwrite_mode: ResMut<OverwriteMode>,
    mut fall_into_void: ResMut<FallIntoVoid>,
    mut selected_element: ResMut<Element>,
) -> Result {
    egui::Window::new("Controls").show(contexts.ctx_mut()?, |ui| {
        // Element selection buttons
        ui.label("Selected Element:");
        ui.horizontal(|ui| {
            let sand_selected = *selected_element == Element::Sand;
            if ui.selectable_label(sand_selected, "Sand").clicked() {
                *selected_element = Element::Sand;
                uniforms.selected_element = Element::Sand.index();
            }
            
            let rainbow_selected = *selected_element == Element::RainbowSand;
            if ui.selectable_label(rainbow_selected, "Rainbow Sand").clicked() {
                *selected_element = Element::RainbowSand;
                uniforms.selected_element = Element::RainbowSand.index();
            }
            
            let wall_selected = *selected_element == Element::Wall;
            if ui.selectable_label(wall_selected, "Wall").clicked() {
                *selected_element = Element::Wall;
                uniforms.selected_element = Element::Wall.index();
            }
        });
        
        ui.separator();

        // Draw radius slider
        ui.horizontal(|ui| {
            ui.label("Draw Radius:");
            let mut radius = uniforms.click_radius;
            if ui.add(egui::Slider::new(&mut radius, 1.0..=50.0)).changed() {
                uniforms.click_radius = radius;
            }
        });

        ui.separator();

        // Fall into void toggle
        let mut fall_void = fall_into_void.0;
        if ui.checkbox(&mut fall_void, "Fall Into Void").changed() {
            fall_into_void.0 = fall_void;
            uniforms.fall_into_void = u32::from(fall_void);
        }
        ui.label("When enabled, elements fall off screen edges. When disabled, elements stop at edges.");

        ui.separator();

        // Overwrite mode toggle
        let mut overwrite = overwrite_mode.0;
        if ui.checkbox(&mut overwrite, "Overwrite").changed() {
            overwrite_mode.0 = overwrite;
            uniforms.overwrite_mode = u32::from(overwrite);
        }
        ui.label("When enabled, drawing overwrites existing materials. When disabled, only draws on empty spaces.");

        ui.separator();
        
        ui.label("Controls: Left Click = Place Selected Element, Right Click = Remove");

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

        ui.separator();

        // Clear button
        if ui.button("Clear Grid").clicked() {
            clear_grid.0 = true;
        }

        ui.separator();

        // Spigot controls for each of the 4 spigots
        ui.collapsing("Spigots", |ui| {
            let valid_elements = Element::spigot_valid_elements();
            let element_names = valid_elements.iter().map(|e| format!("{:?}", e)).collect::<Vec<String>>();
            for i in 0..NUM_SPIGOTS {
                ui.group(|ui| {
                    ui.label(format!("Spigot {}", i + 1));
                    let mut size = match i {
                        0 => uniforms.spigot_sizes.x as f32,
                        1 => uniforms.spigot_sizes.y as f32,
                        2 => uniforms.spigot_sizes.z as f32,
                        3 => uniforms.spigot_sizes.w as f32,
                        _ => unreachable!(),
                    };
                    if ui.add(egui::Slider::new(&mut size, 0.0..=6.0).step_by(1.0)).changed() {
                        match i {
                            0 => uniforms.spigot_sizes.x = size as u32,
                            1 => uniforms.spigot_sizes.y = size as u32,
                            2 => uniforms.spigot_sizes.z = size as u32,
                            3 => uniforms.spigot_sizes.w = size as u32,
                            _ => unreachable!(),
                        }
                    }
                    let current_size = match i {
                        0 => uniforms.spigot_sizes.x,
                        1 => uniforms.spigot_sizes.y,
                        2 => uniforms.spigot_sizes.z,
                        3 => uniforms.spigot_sizes.w,
                        _ => unreachable!(),
                    };
                    if current_size == 0 {
                        ui.label("(Size 0 = disabled)");
                    } else {
                        let current_element = Element::from_index(match i {
                            0 => uniforms.spigot_elements.x,
                            1 => uniforms.spigot_elements.y,
                            2 => uniforms.spigot_elements.z,
                            3 => uniforms.spigot_elements.w,
                            _ => unreachable!(),
                        });
                        let current_idx = valid_elements.iter().position(|&e| e == current_element).unwrap_or(0);
                        egui::ComboBox::from_id_salt(format!("spigot_{}_element", i))
                            .selected_text(&element_names[current_idx])
                            .show_ui(ui, |ui| {
                                for (idx, element) in valid_elements.iter().enumerate() {
                                    if ui.selectable_label(idx == current_idx, &element_names[idx]).clicked() {
                                        match i {
                                            0 => uniforms.spigot_elements.x = *element as u32,
                                            1 => uniforms.spigot_elements.y = *element as u32,
                                            2 => uniforms.spigot_elements.z = *element as u32,
                                            3 => uniforms.spigot_elements.w = *element as u32,
                                            _ => unreachable!(),
                                        }
                                    }
                                }
                            });
                    }
                });
            }
        });
    });
    Ok(())
}

// Switch texture to display every frame to show the one that was written to most recently.
pub fn switch_textures(images: Res<FallingSandImages>, mut sprite: Single<&mut Sprite>) {
    if sprite.image == images.texture_a {
        sprite.image = images.texture_b.clone();
    } else {
        sprite.image = images.texture_a.clone();
    }
}

// Handle mouse clicks and convert to texture coordinates
pub fn handle_mouse_clicks(
    mut uniforms: ResMut<FallingSandUniforms>,
    selected_element: Res<Element>,
    mouse_button_input: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    camera_query: Query<(&Camera, &GlobalTransform), With<Camera2d>>,
    egui_contexts: Option<EguiContexts>,
) {
    // Reset click action each frame (shader will check if it's valid)
    uniforms.click_action = 0;
    uniforms.click_position = IVec2::new(-1, -1);

    // Don't process clicks if egui is consuming the input
    if let Some(mut contexts) = egui_contexts {
        if let Ok(ctx) = contexts.ctx_mut()
            && (ctx.wants_pointer_input() || ctx.is_pointer_over_area())
        {
            return;
        }
    } else {
        // No egui context, continue processing
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

    // Convert world coordinates to texture coordinates
    // The sprite is centered at (0, 0) with size SIZE * DISPLAY_FACTOR
    // So world bounds are from -SIZE.x/2 * DISPLAY_FACTOR to +SIZE.x/2 * DISPLAY_FACTOR
    // Note: World Y increases upward, but texture Y increases downward, so we need to invert Y
    // Convert u32 to f32 for calculations
    {
        let display_factor_f32 = DISPLAY_FACTOR as f32;
        let size_x_f32 = SIZE.x as f32;
        let size_y_f32 = SIZE.y as f32;
        let texture_x = ((world_pos.x / display_factor_f32) + size_x_f32 / 2.0)
            .clamp(0.0, size_x_f32 - 1.0) as i32;
        let normalized_y = (world_pos.y / display_factor_f32) + size_y_f32 / 2.0;
        let texture_y = (size_y_f32 - 1.0 - normalized_y).clamp(0.0, size_y_f32 - 1.0) as i32;

        // Clamp to valid texture coordinates
        if texture_x >= 0
            && texture_x < i32::try_from(SIZE.x).unwrap_or(i32::MAX)
            && texture_y >= 0
            && texture_y < i32::try_from(SIZE.y).unwrap_or(i32::MAX)
        {
            uniforms.click_position = IVec2::new(texture_x, texture_y);

            // Set click action: 1 = add element (left click), 2 = remove element (right click)
            if mouse_button_input.pressed(MouseButton::Left) {
                uniforms.click_action = 1;
                // Update selected element in uniforms (use index() to map to shader constants)
                uniforms.selected_element = selected_element.index();
                // Rainbow sand colors are calculated in the shader based on sim_step/time
            } else if mouse_button_input.pressed(MouseButton::Right) {
                uniforms.click_action = 2;
            }
        }
    }
}

// Handle mouse scroll to adjust radius
pub fn handle_mouse_scroll(
    mut uniforms: ResMut<FallingSandUniforms>,
    mut scroll_evr: MessageReader<bevy::input::mouse::MouseWheel>,
    egui_contexts: Option<EguiContexts>,
) {
    // Don't process scroll if egui is consuming the input
    if let Some(mut contexts) = egui_contexts {
        if let Ok(ctx) = contexts.ctx_mut()
            && (ctx.wants_pointer_input() || ctx.is_pointer_over_area())
        {
            return;
        }
    } else {
        // No egui context, continue processing
    }

    let mut total_scroll = 0.0;
    for ev in scroll_evr.read() {
        total_scroll += ev.y;
    }

    if total_scroll != 0.0 {
        // Adjust radius: scroll up increases, scroll down decreases
        // Clamp between 1.0 and 50.0
        uniforms.click_radius = uniforms
            .click_radius
            .mul_add(total_scroll, 0.5)
            .clamp(1.0, 50.0);
    }
}

// Draw circle outline to show where sand will be placed/removed
pub fn draw_circle_preview(
    mut gizmos: Gizmos,
    uniforms: Res<FallingSandUniforms>,
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
    } else {
        // No egui context, continue processing
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
    // Convert u32 to f32 for calculation
    let world_radius = uniforms.click_radius * DISPLAY_FACTOR as f32;
    gizmos.circle_2d(world_pos, world_radius, Color::WHITE);
}

// Color shift is now handled entirely in the shader for rainbow sand
// Regular sand uses a single consistent color
pub fn shift_color_over_time(_uniforms: ResMut<FallingSandUniforms>, _time: Res<Time>) {
    // No longer needed - rainbow sand colors are calculated in the shader based on sim_step
}

// Increment simulation step each frame for alternating diagonal movement
// Respects simulation speed: 0.0 = paused (don't increment), > 0.0 = increment
pub fn increment_sim_step(
    mut uniforms: ResMut<FallingSandUniforms>,
    simulation_speed: Res<SimulationSpeed>,
) {
    if simulation_speed.0 > 0.0 {
        // For speeds > 1.0, we could increment multiple times, but that doesn't actually
        // make the simulation faster, just changes diagonal alternation pattern.
        // To actually speed up, we'd need to run the compute shader multiple times per frame.
        uniforms.sim_step = uniforms.sim_step.wrapping_add(1);
    }
}

// Reset clear_grid flag after the init shader has run
pub fn reset_clear_grid_flag(mut clear_grid: ResMut<ClearGrid>) {
    if clear_grid.0 {
        // The init shader runs in the render graph, so we reset the flag here
        // after it's been processed
        clear_grid.0 = false;
    }
}

// Sync overwrite_mode and fall_into_void resources to uniforms
pub fn sync_ui_settings_to_uniforms(
    mut uniforms: ResMut<FallingSandUniforms>,
    overwrite_mode: Res<OverwriteMode>,
    fall_into_void: Res<FallIntoVoid>,
) {
    // Only update if changed to avoid unnecessary writes
    if uniforms.overwrite_mode != u32::from(overwrite_mode.0) {
        uniforms.overwrite_mode = u32::from(overwrite_mode.0);
    }
    if uniforms.fall_into_void != u32::from(fall_into_void.0) {
        uniforms.fall_into_void = u32::from(fall_into_void.0);
    }
}

/// Prepares the bind groups for the falling sand compute shader.
///
/// # Panics
/// Panics if the GPU images for the falling sand textures are not found.
pub fn prepare_bind_group(
    mut commands: Commands,
    pipeline: Res<FallingSandPipeline>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    falling_sand_images: Res<FallingSandImages>,
    falling_sand_uniforms: Res<FallingSandUniforms>,
    render_device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    let view_a = gpu_images.get(&falling_sand_images.texture_a).unwrap();
    let view_b = gpu_images.get(&falling_sand_images.texture_b).unwrap();
    let element_type_view_a = gpu_images.get(&falling_sand_images.element_type_a).unwrap();
    let element_type_view_b = gpu_images.get(&falling_sand_images.element_type_b).unwrap();

    // Uniform buffer is used here to demonstrate how to set up a uniform in a compute shader
    // Alternatives such as storage buffers or push constants may be more suitable for your use case
    let mut uniform_buffer = UniformBuffer::from(*falling_sand_uniforms);
    uniform_buffer.write_buffer(&render_device, &queue);

    let bind_group_0 = render_device.create_bind_group(
        None,
        &pipeline.texture_bind_group_layout,
        &BindGroupEntries::sequential((
            &view_a.texture_view,
            &view_b.texture_view,
            &element_type_view_a.texture_view,
            &element_type_view_b.texture_view,
            &uniform_buffer,
        )),
    );
    let bind_group_1 = render_device.create_bind_group(
        None,
        &pipeline.texture_bind_group_layout,
        &BindGroupEntries::sequential((
            &view_b.texture_view,
            &view_a.texture_view,
            &element_type_view_b.texture_view,
            &element_type_view_a.texture_view,
            &uniform_buffer,
        )),
    );
    commands.insert_resource(FallingSandImageBindGroups([bind_group_0, bind_group_1]));
}

// Initialize the simulation frame accumulator and runs counter in the render world
pub fn init_simulation_frame_accumulator(mut commands: Commands) {
    commands.insert_resource(SimulationFrameAccumulator::default());
    // RunsThisFrame is defined in plugins, but we can't import it directly
    // Instead, we'll initialize it in the plugin's init system
    // Actually, let's just initialize it here using the full path
    commands.insert_resource(crate::plugins::RunsThisFrame::default());
}

pub fn init_falling_sand_pipeline(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    _render_queue: Res<RenderQueue>,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
) {
    let texture_bind_group_layout = render_device.create_bind_group_layout(
        "FallingSandImages",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::COMPUTE,
            (
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::ReadOnly),
                texture_storage_2d(TextureFormat::Rgba32Float, StorageTextureAccess::WriteOnly),
                texture_storage_2d(TextureFormat::R32Uint, StorageTextureAccess::ReadOnly),
                texture_storage_2d(TextureFormat::R32Uint, StorageTextureAccess::WriteOnly),
                uniform_buffer::<FallingSandUniforms>(false),
            ),
        ),
    );

    let shader = asset_server.load(SHADER_ASSET_PATH);
    let init_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        layout: vec![texture_bind_group_layout.clone()],
        shader: shader.clone(),
        entry_point: Some(Cow::from("init")),
        ..default()
    });
    let update_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        layout: vec![texture_bind_group_layout.clone()],
        shader,
        entry_point: Some(Cow::from("update")),
        ..default()
    });
    commands.insert_resource(FallingSandPipeline {
        texture_bind_group_layout,
        init_pipeline,
        update_pipeline,
    });
}
