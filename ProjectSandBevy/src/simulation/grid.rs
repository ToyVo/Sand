use crate::elements::Element;
use crate::SIZE;
use bevy::prelude::*;

/// The game grid stores element data in a flat array
/// Index calculation: i = y * width + x
#[derive(Resource, serde::Serialize, serde::Deserialize)]
pub struct GameGrid {
    pub elements: Vec<Element>,
    pub width: u32,
    pub height: u32,
}

impl GameGrid {
    /// Clear the entire grid (set all elements to Background)
    pub fn clear(&mut self) {
        for element in &mut self.elements {
            *element = Element::Background;
        }
    }
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            elements: vec![Element::Background; (width * height) as usize],
            width,
            height,
        }
    }

    /// Get element at (x, y)
    pub fn get(&self, x: u32, y: u32) -> Element {
        if x >= self.width || y >= self.height {
            return Element::Background;
        }
        let idx = (y * self.width + x) as usize;
        self.elements[idx]
    }

    /// Set element at (x, y)
    pub fn set(&mut self, x: u32, y: u32, element: Element) {
        if x >= self.width || y >= self.height {
            return;
        }
        let idx = (y * self.width + x) as usize;
        self.elements[idx] = element;
    }

    /// Get element at index i
    pub fn get_index(&self, i: usize) -> Element {
        if i >= self.elements.len() {
            return Element::Background;
        }
        self.elements[i]
    }

    /// Set element at index i
    pub fn set_index(&mut self, i: usize, element: Element) {
        if i >= self.elements.len() {
            return;
        }
        self.elements[i] = element;
    }

    /// Convert index to (x, y)
    pub fn index_to_xy(&self, i: usize) -> (u32, u32) {
        let x = (i % self.width as usize) as u32;
        let y = (i / self.width as usize) as u32;
        (x, y)
    }

    /// Convert (x, y) to index
    pub fn xy_to_index(&self, x: u32, y: u32) -> usize {
        (y * self.width + x) as usize
    }

    /// Check if position is valid
    pub fn is_valid(&self, x: u32, y: u32) -> bool {
        x < self.width && y < self.height
    }

    /// Get max X index
    pub fn max_x(&self) -> u32 {
        self.width.saturating_sub(1)
    }

    /// Get max Y index
    pub fn max_y(&self) -> u32 {
        self.height.saturating_sub(1)
    }
}

impl Default for GameGrid {
    fn default() -> Self {
        Self::new(SIZE.x, SIZE.y)
    }
}

