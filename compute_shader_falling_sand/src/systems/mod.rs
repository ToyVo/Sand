#![allow(
    clippy::needless_pass_by_value,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::similar_names
)]

use crate::plugins::{
    FallingSandImageBindGroups, FallingSandImages, FallingSandPipeline, FallingSandUniforms,
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

pub fn setup(mut commands: Commands, mut image_assets: ResMut<Assets<Image>>) {
    let mut image = Image::new_target_texture(SIZE.x, SIZE.y, TextureFormat::Rgba32Float);
    image.asset_usage = RenderAssetUsages::RENDER_WORLD;
    image.texture_descriptor.usage =
        TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;
    let texture_a_handle = image_assets.add(image.clone());
    let texture_b_handle = image_assets.add(image);

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
    });

    commands.insert_resource(FallingSandUniforms {
        sand_color: LinearRgba::rgb(0.76, 0.70, 0.50), // Sandy color
        size: SIZE,
        click_position: IVec2::new(-1, -1), // -1 means no click
        continuous_spawn: 1,                // Enable continuous spawning by default (1 = true)
        click_radius: 5.0,                  // Default radius
        click_action: 0,                    // 0 = no action
        color_shift_enabled: 1,             // Enabled by default
    });
}

/// UI system for the egui controls window.
///
/// # Errors
/// Returns an error if the egui context cannot be accessed.
pub fn ui_system(mut contexts: EguiContexts, mut uniforms: ResMut<FallingSandUniforms>) -> Result {
    egui::Window::new("Controls").show(contexts.ctx_mut()?, |ui| {
        let mut continuous_spawn = uniforms.continuous_spawn == 1;
        if ui
            .checkbox(&mut continuous_spawn, "Continuous Spawn")
            .changed()
        {
            uniforms.continuous_spawn = u32::from(continuous_spawn);
        }

        ui.separator();

        ui.label("Sand Color:");
        // Convert LinearRgba to egui::Color32 (sRGB)
        // LinearRgba stores linear RGB values (0.0-1.0), convert to sRGB for display
        let linear = uniforms.sand_color;
        let mut color32 = egui::Color32::from_rgba_unmultiplied(
            (linear.red * 255.0).clamp(0.0, 255.0) as u8,
            (linear.green * 255.0).clamp(0.0, 255.0) as u8,
            (linear.blue * 255.0).clamp(0.0, 255.0) as u8,
            255,
        );

        if ui.color_edit_button_srgba(&mut color32).changed() {
            // Convert back from Color32 to LinearRgba
            let r = f32::from(color32.r()) / 255.0;
            let g = f32::from(color32.g()) / 255.0;
            let b = f32::from(color32.b()) / 255.0;
            uniforms.sand_color = LinearRgba::rgb(r, g, b);
        }

        ui.separator();

        let mut color_shift = uniforms.color_shift_enabled == 1;
        if ui
            .checkbox(&mut color_shift, "Shift Color Over Time")
            .changed()
        {
            uniforms.color_shift_enabled = u32::from(color_shift);
        }
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

            // Set click action: 1 = add sand (left click), 2 = remove sand (right click)
            if mouse_button_input.pressed(MouseButton::Left) {
                uniforms.click_action = 1;
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

// Shift sand color over time when enabled
pub fn shift_color_over_time(mut uniforms: ResMut<FallingSandUniforms>, time: Res<Time>) {
    if uniforms.color_shift_enabled == 0 {
        return;
    }

    // Use time to cycle through hues
    // We'll use a sine wave to smoothly cycle through colors
    let t = time.elapsed_secs() * 0.5; // Speed of color shift (0.5 cycles per second)

    // Convert RGB to HSV, shift hue, convert back
    // For simplicity, we'll cycle through RGB values using sine waves
    let two_pi = 2.0 * std::f32::consts::PI;
    let phase_offset = 2.0 * std::f32::consts::PI / 3.0;
    let r = (t * two_pi).sin().mul_add(0.3, 0.5);
    let g = (t.mul_add(two_pi, phase_offset)).sin().mul_add(0.3, 0.5);
    let b = (t.mul_add(two_pi, phase_offset * 2.0))
        .sin()
        .mul_add(0.3, 0.5);

    // Clamp values to [0.0, 1.0] and ensure they're bright enough
    let r = r.clamp(0.2, 1.0);
    let g = g.clamp(0.2, 1.0);
    let b = b.clamp(0.2, 1.0);

    uniforms.sand_color = LinearRgba::rgb(r, g, b);
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
            &uniform_buffer,
        )),
    );
    let bind_group_1 = render_device.create_bind_group(
        None,
        &pipeline.texture_bind_group_layout,
        &BindGroupEntries::sequential((
            &view_b.texture_view,
            &view_a.texture_view,
            &uniform_buffer,
        )),
    );
    commands.insert_resource(FallingSandImageBindGroups([bind_group_0, bind_group_1]));
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
