pub mod elements;
pub mod particles;
pub mod plugins;
pub mod simulation;
pub mod spigots;
pub mod systems;

pub const DISPLAY_FACTOR: u32 = 2;
pub const SIZE: bevy::math::UVec2 =
    bevy::math::UVec2::new(1280 / DISPLAY_FACTOR, 720 / DISPLAY_FACTOR);
pub const WORKGROUP_SIZE: u32 = 8;
