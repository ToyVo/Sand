use bevy::prelude::*;

/// Element types in the simulation.
/// 
/// The order matters - it's used for indexing in the element actions array.
/// We use a similar color encoding scheme as the TypeScript version where
/// the lower 2 bits of R, G, B channels encode the element index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[repr(u8)]
pub enum Element {
    Background = 0,
    Wall = 1,
    Sand = 2,
    Water = 3,
    Fire = 4,
    Salt = 5,
    Oil = 6,
    Rock = 7,
    Ice = 8,
    Lava = 9,
    Steam = 10,
    SaltWater = 11,
    Plant = 12,
    Gunpowder = 13,
    Wax = 14,
    Concrete = 15,
    Nitro = 16,
    Napalm = 17,
    C4 = 18,
    Fuse = 19,
    Acid = 20,
    Cryo = 21,
    Methane = 22,
    Soil = 23,
    WetSoil = 24,
    Thermite = 25,
    Spout = 26,
    Well = 27,
    Torch = 28,
    Branch = 29,
    Leaf = 30,
    Pollen = 31,
    FallingWax = 32,
    ChilledIce = 33,
    Mystery = 34,
    ChargedNitro = 35,
    BurningThermite = 36,
    RainbowSand = 37,
    // More elements will be added here
}

impl Element {
    /// Get the color for this element as LinearRgba
    pub fn color(&self) -> LinearRgba {
        match self {
            Element::Background => LinearRgba::rgb(0.0, 0.0, 0.0),
            Element::Wall => LinearRgba::rgb(0.5, 0.5, 0.5), // 127, 127, 127
            Element::Sand => LinearRgba::rgb(0.76, 0.70, 0.50), // 223, 193, 99
            Element::Water => LinearRgba::rgb(0.0, 0.04, 1.0), // 0, 10, 255
            Element::Fire => LinearRgba::rgb(1.0, 0.0, 0.04), // 255, 0, 10
            Element::Salt => LinearRgba::rgb(0.99, 0.99, 0.99), // 253, 253, 253
            Element::Oil => LinearRgba::rgb(0.59, 0.24, 0.0), // 150, 60, 0
            Element::Rock => LinearRgba::rgb(0.27, 0.16, 0.03), // 68, 40, 8
            Element::Ice => LinearRgba::rgb(0.63, 0.91, 1.0), // 161, 232, 255
            Element::Lava => LinearRgba::rgb(0.96, 0.43, 0.16), // 245, 110, 40
            Element::Steam => LinearRgba::rgb(0.76, 0.84, 0.92), // 195, 214, 235
            Element::SaltWater => LinearRgba::rgb(0.50, 0.69, 1.0), // 127, 175, 255
            Element::Plant => LinearRgba::rgb(0.0, 0.86, 0.0), // 0, 220, 0
            Element::Gunpowder => LinearRgba::rgb(0.67, 0.67, 0.55), // 170, 170, 140
            Element::Wax => LinearRgba::rgb(0.94, 0.88, 0.83), // 239, 225, 211
            Element::Concrete => LinearRgba::rgb(0.71, 0.71, 0.71), // 180, 180, 180
            Element::Nitro => LinearRgba::rgb(0.0, 0.59, 0.10), // 0, 150, 26
            Element::Napalm => LinearRgba::rgb(0.86, 0.50, 0.27), // 220, 128, 70
            Element::C4 => LinearRgba::rgb(0.94, 0.90, 0.59), // 240, 230, 150
            Element::Fuse => LinearRgba::rgb(0.86, 0.69, 0.78), // 219, 175, 199
            Element::Acid => LinearRgba::rgb(0.62, 0.94, 0.16), // 157, 240, 40
            Element::Cryo => LinearRgba::rgb(0.0, 0.84, 1.0), // 0, 213, 255
            Element::Methane => LinearRgba::rgb(0.55, 0.55, 0.55), // 140, 140, 140
            Element::Soil => LinearRgba::rgb(0.47, 0.29, 0.13), // 120, 75, 33
            Element::WetSoil => LinearRgba::rgb(0.27, 0.14, 0.04), // 70, 35, 10
            Element::Thermite => LinearRgba::rgb(0.76, 0.55, 0.27), // 195, 140, 70
            Element::Spout => LinearRgba::rgb(0.46, 0.74, 0.99), // 117, 189, 252
            Element::Well => LinearRgba::rgb(0.51, 0.04, 0.11), // 131, 11, 28
            Element::Torch => LinearRgba::rgb(0.78, 0.02, 0.0), // 200, 5, 0
            Element::Branch => LinearRgba::rgb(0.65, 0.50, 0.39), // 166, 128, 100
            Element::Leaf => LinearRgba::rgb(0.32, 0.42, 0.18), // 82, 107, 45
            Element::Pollen => LinearRgba::rgb(0.90, 0.92, 0.43), // 230, 235, 110
            Element::FallingWax => LinearRgba::rgb(0.94, 0.88, 0.83), // 240, 225, 211
            Element::ChilledIce => LinearRgba::rgb(0.08, 0.60, 0.86), // 20, 153, 220
            Element::Mystery => LinearRgba::rgb(0.64, 0.91, 0.77), // 162, 232, 196
            Element::ChargedNitro => LinearRgba::rgb(0.96, 0.38, 0.31), // 245, 98, 78
            Element::BurningThermite => LinearRgba::rgb(1.0, 0.51, 0.51), // 255, 130, 130
            Element::RainbowSand => LinearRgba::rgb(0.76, 0.70, 0.50), // Base color similar to sand, but will be shifted
        }
    }

    /// Get the element index (for shader encoding)
    pub fn index(&self) -> u8 {
        *self as u8
    }

    /// Convert from element index
    pub fn from_index(index: u8) -> Self {
        match index {
            0 => Element::Background,
            1 => Element::Wall,
            2 => Element::Sand,
            3 => Element::Water,
            4 => Element::Fire,
            5 => Element::Salt,
            6 => Element::Oil,
            7 => Element::Rock,
            8 => Element::Ice,
            9 => Element::Lava,
            10 => Element::Steam,
            11 => Element::SaltWater,
            12 => Element::Plant,
            13 => Element::Gunpowder,
            14 => Element::Wax,
            15 => Element::Concrete,
            16 => Element::Nitro,
            17 => Element::Napalm,
            18 => Element::C4,
            19 => Element::Fuse,
            20 => Element::Acid,
            21 => Element::Cryo,
            22 => Element::Methane,
            23 => Element::Soil,
            24 => Element::WetSoil,
            25 => Element::Thermite,
            26 => Element::Spout,
            27 => Element::Well,
            28 => Element::Torch,
            29 => Element::Branch,
            30 => Element::Leaf,
            31 => Element::Pollen,
            32 => Element::FallingWax,
            33 => Element::ChilledIce,
            34 => Element::Mystery,
            35 => Element::ChargedNitro,
            36 => Element::BurningThermite,
            37 => Element::RainbowSand,
            _ => Element::Background,
        }
    }

    /// Encode element to color with index encoding in lower bits
    /// Similar to TypeScript version: uses lower 2 bits of R, G, B for index
    pub fn to_encoded_color(&self) -> LinearRgba {
        self.to_encoded_color_with_shift(0)
    }
    
    /// Encode color with optional color shift (for rainbow mode)
    /// The shift is added to the encoded index, creating a rainbow effect
    pub fn to_encoded_color_with_shift(&self, shift: u8) -> LinearRgba {
        // For RainbowSand, generate actual rainbow colors
        if matches!(self, Element::RainbowSand) {
            // Generate rainbow color based on shift (0-255 maps to full 0-360 degree hue range)
            // Use HSV to RGB conversion for smooth rainbow across full spectrum
            let hue = (shift as f32 / 255.0) * 360.0; // Map 0-255 to 0-360 degrees
            let saturation = 0.8; // High saturation for vibrant colors
            let value = 0.9; // Bright value
            
            // HSV to RGB conversion
            let c = value * saturation;
            let x = c * (1.0 - ((hue / 60.0) % 2.0 - 1.0).abs());
            let m = value - c;
            
            let (r, g, b) = if hue < 60.0 {
                (c, x, 0.0)
            } else if hue < 120.0 {
                (x, c, 0.0)
            } else if hue < 180.0 {
                (0.0, c, x)
            } else if hue < 240.0 {
                (0.0, x, c)
            } else if hue < 300.0 {
                (x, 0.0, c)
            } else {
                (c, 0.0, x)
            };
            
            let rgb_color = LinearRgba::rgb(r + m, g + m, b + m);
            
            // Still need to encode the element index in lower 2 bits for decoding
            let index = self.index();
            let r_idx = index & 0b11;
            let g_idx = (index >> 2) & 0b11;
            let b_idx = (index >> 4) & 0b11;
            
            let r = ((rgb_color.red * 255.0) as u8 & 0xFC) | r_idx;
            let g = ((rgb_color.green * 255.0) as u8 & 0xFC) | g_idx;
            let b = ((rgb_color.blue * 255.0) as u8 & 0xFC) | b_idx;
            
            LinearRgba::rgb(
                r as f32 / 255.0,
                g as f32 / 255.0,
                b as f32 / 255.0,
            )
        } else {
            // For other elements, use the original encoding with shift
            let base_color = self.color();
            let index = self.index();
            
            // Add shift to index (wraps at 64, which is 2^6)
            let shifted_index = (index as u8).wrapping_add(shift);
            
            // Encode shifted index in lower 2 bits: r_idx = shifted_index & 0b11, g_idx = (shifted_index >> 2) & 0b11, b_idx = (shifted_index >> 4) & 0b11
            let r_idx = shifted_index & 0b11;
            let g_idx = (shifted_index >> 2) & 0b11;
            let b_idx = (shifted_index >> 4) & 0b11;
            
            // Clear lower 2 bits and add shifted index
            let r = ((base_color.red * 255.0) as u8 & 0xFC) | r_idx;
            let g = ((base_color.green * 255.0) as u8 & 0xFC) | g_idx;
            let b = ((base_color.blue * 255.0) as u8 & 0xFC) | b_idx;
            
            LinearRgba::rgb(
                r as f32 / 255.0,
                g as f32 / 255.0,
                b as f32 / 255.0,
            )
        }
    }

    /// Decode element from encoded color
    pub fn from_encoded_color(color: LinearRgba) -> Self {
        let r = (color.red * 255.0) as u8;
        let g = (color.green * 255.0) as u8;
        let b = (color.blue * 255.0) as u8;
        
        // Extract index from lower 2 bits
        let index = (r & 0b11) | ((g & 0b11) << 2) | ((b & 0b11) << 4);
        
        Self::from_index(index)
    }

    /// Check if element is solid (doesn't fall)
    pub fn is_solid(&self) -> bool {
        matches!(self, Element::Wall)
    }

    /// Check if element is liquid (flows and spreads)
    pub fn is_liquid(&self) -> bool {
        matches!(self, Element::Water | Element::Oil | Element::SaltWater | Element::Nitro | Element::Napalm | Element::Acid)
    }

    /// Check if element is powder (falls like sand)
    pub fn is_powder(&self) -> bool {
        matches!(self, Element::Sand | Element::Salt | Element::Gunpowder | Element::Soil | Element::WetSoil | Element::Thermite | Element::Pollen | Element::Mystery | Element::ChargedNitro)
    }

    /// Check if element is empty/background
    pub fn is_empty(&self) -> bool {
        matches!(self, Element::Background)
    }

    /// Check if element is valid for spigots (anything affected by gravity)
    /// Excludes: Background, Wall, Fire, Ice, Steam, Plant, Wax, Fuse, C4, Cryo, Methane, Spout, Well, Torch, Branch, Leaf, FallingWax, ChilledIce, BurningThermite
    pub fn is_valid_for_spigot(&self) -> bool {
        !matches!(self, Element::Background | Element::Wall | Element::Fire | Element::Ice | Element::Steam | Element::Plant | Element::Wax | Element::Fuse | Element::C4 | Element::Cryo | Element::Methane | Element::Spout | Element::Well | Element::Torch | Element::Branch | Element::Leaf | Element::FallingWax | Element::ChilledIce | Element::BurningThermite)
    }

    /// Get all elements that are valid for spigots (affected by gravity)
    pub fn spigot_valid_elements() -> Vec<Element> {
        vec![
            Element::Sand,
            Element::RainbowSand,
            Element::Water,
            Element::Salt,
            Element::Oil,
            Element::Rock,
            Element::Lava,
            Element::SaltWater,
            Element::Gunpowder,
            Element::Concrete,
            Element::Nitro,
            Element::Napalm,
            Element::Acid,
            Element::Soil,
            Element::WetSoil,
            Element::Thermite,
            Element::Pollen,
            Element::Mystery,
            Element::ChargedNitro,
        ]
    }
}

impl Default for Element {
    fn default() -> Self {
        Element::Background
    }
}

