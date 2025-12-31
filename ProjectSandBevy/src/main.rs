//! A CPU-based falling sand simulation with full complex interactions.

use bevy::prelude::*;
use bevy_egui::{EguiPlugin, EguiPrimaryContextPass};
use ProjectSandBevy::{DISPLAY_FACTOR, SIZE, systems};

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::BLACK))
        .add_plugins((
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        resolution: (SIZE * DISPLAY_FACTOR).into(),
                        resizable: true,
                        // uncomment for unthrottled FPS
                        // present_mode: bevy::window::PresentMode::AutoNoVsync,
                        ..default()
                    }),
                    ..default()
                })
                .set(ImagePlugin::default_nearest()),
            EguiPlugin::default(),
        ))
        .add_systems(Startup, systems::setup)
        .add_systems(EguiPrimaryContextPass, systems::ui_system)
        .add_systems(
            Update,
            (
                systems::handle_window_resize,
                systems::handle_save_load,
                systems::update_game_simulation,
                systems::update_particles,
                systems::render_grid_to_texture,
                systems::render_particles,
                systems::composite_particles,
                systems::handle_mouse_clicks_cpu,
                systems::handle_mouse_scroll,
                systems::draw_circle_preview,
            )
                .chain(), // Ensure order: resize -> save/load -> update -> render grid -> render particles -> composite
        )
        .run();
}
