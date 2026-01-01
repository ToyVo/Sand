use bevy::{color::LinearRgba, ecs::resource::Resource};

#[derive(Debug, Resource, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[repr(u32)]
pub enum Element {
    Background = 0,
    Wall = 1,
    Sand = 2,
    RainbowSand = 3,
}

impl Default for Element {
    fn default() -> Self {
        Self::Background
    }
}

impl Element {
    /// Get the color for this element as LinearRgba
    pub fn color(&self) -> LinearRgba {
        match self {
            Element::Background => LinearRgba::rgb(0.0, 0.0, 0.0),
            Element::Wall => LinearRgba::rgb(0.5, 0.5, 0.5), // 127, 127, 127
            Element::Sand => LinearRgba::rgb(0.76, 0.70, 0.50), // 223, 193, 99
            Element::RainbowSand => LinearRgba::rgb(0.76, 0.70, 0.50), // Base color similar to sand, but will be shifted
        }
    }

    /// Get the element index (for shader encoding)
    pub fn index(&self) -> u32 {
        *self as u32
    }

    /// Convert from element index
    pub fn from_index(index: u32) -> Self {
        match index {
            0 => Element::Background,
            1 => Element::Wall,
            2 => Element::Sand,
            3 => Element::RainbowSand,
            _ => Element::Background,
        }
    }

    /// Get all elements that are valid for spigots (affected by gravity)
    pub fn spigot_valid_elements() -> Vec<Element> {
        vec![
            Element::Sand,
            Element::RainbowSand,
        ]
    }
}