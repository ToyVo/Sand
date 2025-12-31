use crate::elements::Element;
use bevy::prelude::*;

pub const NUM_SPIGOTS: usize = 4;
pub const SPIGOT_HEIGHT: u32 = 10;
pub const DEFAULT_SPIGOT_SIZE: u32 = 5;

/// Resource storing spigot configuration
#[derive(Resource, Clone)]
pub struct Spigots {
    pub elements: [Element; NUM_SPIGOTS],
    pub sizes: [u32; NUM_SPIGOTS], // Size 0 = disabled, 1-6 = enabled with that size
}

impl Default for Spigots {
    fn default() -> Self {
        Self {
            elements: [
                Element::RainbowSand,
                Element::Water,
                Element::Salt,
                Element::Oil,
            ],
            sizes: [DEFAULT_SPIGOT_SIZE; NUM_SPIGOTS], // Default size (5) means enabled
        }
    }
}

impl Spigots {
    /// Get spigot positions evenly distributed across the given width
    pub fn get_spigot_positions(&self, width: u32) -> Vec<(u32, u32, u32)> {
        // Calculate spacing: evenly distribute spigots across the width
        // We want equal spacing between spigots and from edges
        let total_spigot_width: u32 = self.sizes.iter().sum::<u32>();
        let num_enabled = self.sizes.iter().filter(|&&s| s > 0).count() as u32;
        
        // If no spigots are enabled, return empty
        if num_enabled == 0 {
            return Vec::new();
        }
        
        // Calculate spacing: (total_width - sum_of_spigot_widths) / (num_spigots + 1)
        // This gives equal spacing on both sides and between spigots
        let available_width = width.saturating_sub(total_spigot_width);
        let spacing = if num_enabled > 1 {
            available_width / (num_enabled + 1)
        } else {
            // Single spigot: center it
            available_width / 2
        };
        
        // Start position: first spacing from left edge
        let start_x = spacing;
        
        let mut positions = Vec::new();
        let mut current_x = start_x;
        
        for i in 0..NUM_SPIGOTS {
            if self.sizes[i] > 0 {
                positions.push((current_x, self.sizes[i], i as u32));
                current_x += self.sizes[i] + spacing; // Move to next spigot with spacing
            }
        }
        
        positions
    }
}

