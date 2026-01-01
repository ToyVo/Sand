//! A compute shader that simulates falling sand.
//!
//! Compute shaders use the GPU for computing arbitrary information, that may be independent of what
//! is rendered to the screen.

use bevy::prelude::*;
use bevy_egui::{EguiPlugin, EguiPrimaryContextPass};
use compute_shader_falling_sand::{DISPLAY_FACTOR, SIZE, plugins, systems};

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::BLACK))
        .add_plugins((
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        resolution: (SIZE * DISPLAY_FACTOR).into(),
                        // uncomment for unthrottled FPS
                        // present_mode: bevy::window::PresentMode::AutoNoVsync,
                        ..default()
                    }),
                    ..default()
                })
                .set(ImagePlugin::default_nearest()),
            EguiPlugin::default(),
            plugins::FallingSandComputePlugin,
        ))
        .add_systems(Startup, systems::setup)
        .add_systems(EguiPrimaryContextPass, systems::ui_system)
        .add_systems(
            Update,
            (
                systems::increment_sim_step,
                systems::reset_clear_grid_flag,
                systems::sync_ui_settings_to_uniforms,
                systems::switch_textures,
                systems::handle_mouse_clicks,
                systems::handle_mouse_scroll,
                systems::draw_circle_preview,
                systems::shift_color_over_time,
            ),
        )
        .run();
}
