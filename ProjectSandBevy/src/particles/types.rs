use bevy::prelude::*;
use crate::elements::Element;

/// Maximum number of particles in the system
/// Increased from 1000 to 2048 to handle tree generation better
pub const MAX_NUM_PARTICLES: usize = 2048;

/// Particle types (matching TypeScript indices)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ParticleType {
    Unknown = 0,
    Nitro = 1,
    Napalm = 2,
    C4 = 3,
    Lava = 4,
    Magic1 = 5,  // multi-pronged star
    Magic2 = 6,  // spiral
    Methane = 7,
    Tree = 8,
    ChargedNitro = 9,
    Nuke = 10,
}

impl ParticleType {
    pub fn from_index(index: u8) -> Self {
        match index {
            0 => ParticleType::Unknown,
            1 => ParticleType::Nitro,
            2 => ParticleType::Napalm,
            3 => ParticleType::C4,
            4 => ParticleType::Lava,
            5 => ParticleType::Magic1,
            6 => ParticleType::Magic2,
            7 => ParticleType::Methane,
            8 => ParticleType::Tree,
            9 => ParticleType::ChargedNitro,
            10 => ParticleType::Nuke,
            _ => ParticleType::Unknown,
        }
    }

    pub fn index(&self) -> u8 {
        *self as u8
    }
}

/// A single particle in the simulation
/// Particles are separate from grid elements and move independently
#[derive(Component, Clone)]
pub struct Particle {
    pub particle_type: ParticleType,
    pub init_x: f32,
    pub init_y: f32,
    pub x: f32,
    pub y: f32,
    pub prev_x: f32,  // Previous x position (for line drawing)
    pub prev_y: f32,  // Previous y position (for line drawing)
    pub init_i: usize,  // Initial grid index
    pub color: Element,  // Element color to use for rendering
    pub velocity: f32,
    pub angle: f32,
    pub x_velocity: f32,
    pub y_velocity: f32,
    pub size: f32,
    pub action_iterations: u32,
    pub active: bool,
    pub reinitialized: bool,
    
    // Type-specific data (stored as Option to avoid boxing)
    pub max_iterations: Option<u32>,  // For particles with fixed lifetimes
    pub min_y: Option<f32>,  // For charged nitro (wall collision)
    pub magic_2_max_radius: Option<f32>,  // For magic2 spiral
    pub magic_2_theta: Option<f32>,
    pub magic_2_speed: Option<f32>,
    pub magic_2_radius_spacing: Option<f32>,
    pub magic_2_radius: Option<f32>,
    pub y_acceleration: Option<f32>,  // For lava particles
    pub init_y_velocity: Option<f32>,  // For lava particles
    // Tree particle data
    pub tree_generation: Option<u32>,  // Generation number
    pub tree_branch_spacing: Option<u32>,  // Spacing between branches
    pub tree_max_branches: Option<u32>,  // Maximum branches to create
    pub tree_next_branch: Option<u32>,  // Iteration when next branch should be created
    pub tree_branches: Option<u32>,  // Number of branches created so far
    pub tree_type: Option<u8>,  // Tree type (0 = Tree0, 1 = Tree2, etc.)
}

impl Default for Particle {
    fn default() -> Self {
        Self {
            particle_type: ParticleType::Unknown,
            init_x: -1.0,
            init_y: -1.0,
            x: -1.0,
            y: -1.0,
            prev_x: -1.0,
            prev_y: -1.0,
            init_i: 0,
            color: Element::Fire,
            velocity: 0.0,
            angle: 0.0,
            x_velocity: 0.0,
            y_velocity: 0.0,
            size: 0.0,
            action_iterations: 0,
            active: false,
            reinitialized: false,
            max_iterations: None,
            min_y: None,
            magic_2_max_radius: None,
            magic_2_theta: None,
            magic_2_speed: None,
            magic_2_radius_spacing: None,
            magic_2_radius: None,
            y_acceleration: None,
            init_y_velocity: None,
            tree_generation: None,
            tree_branch_spacing: None,
            tree_max_branches: None,
            tree_next_branch: None,
            tree_branches: None,
            tree_type: None,
        }
    }
}

impl Particle {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set particle color (element to use for rendering)
    pub fn set_color(&mut self, color: Element) {
        self.color = color;
    }

    /// Set velocity from magnitude and angle
    pub fn set_velocity(&mut self, velocity: f32, angle: f32) {
        self.velocity = velocity;
        self.angle = angle;
        self.x_velocity = velocity * angle.cos();
        self.y_velocity = velocity * angle.sin();
    }

    /// Check if particle is off canvas
    pub fn off_canvas(&self, max_x: f32, max_y: f32) -> bool {
        self.x < 0.0 || self.x > max_x || self.y < 0.0 || self.y > max_y
    }

    /// Reset particle to inactive state
    pub fn reset(&mut self) {
        self.particle_type = ParticleType::Unknown;
        self.init_x = -1.0;
        self.init_y = -1.0;
        self.x = -1.0;
        self.y = -1.0;
        self.prev_x = -1.0;
        self.prev_y = -1.0;
        self.init_i = 0;
        self.color = Element::Fire;
        self.velocity = 0.0;
        self.angle = 0.0;
        self.x_velocity = 0.0;
        self.y_velocity = 0.0;
        self.size = 0.0;
        self.action_iterations = 0;
        self.active = false;
        self.reinitialized = false;
        self.max_iterations = None;
        self.min_y = None;
        self.magic_2_max_radius = None;
        self.magic_2_theta = None;
        self.magic_2_speed = None;
        self.magic_2_radius_spacing = None;
        self.magic_2_radius = None;
        self.y_acceleration = None;
        self.init_y_velocity = None;
        self.tree_generation = None;
        self.tree_branch_spacing = None;
        self.tree_max_branches = None;
        self.tree_next_branch = None;
        self.tree_branches = None;
        self.tree_type = None;
    }
}

/// Paintable particle colors - colors that can be copied from particle canvas to main canvas
/// These match element colors that particles can represent
pub const PAINTABLE_PARTICLE_COLORS: &[Element] = &[
    Element::Fire,
    Element::Wall,
    Element::Rock,
    Element::Lava,
    Element::Plant,
    Element::Spout,
    Element::Well,
    Element::Wax,
    Element::Ice,
    Element::Branch,
    Element::Leaf,
    Element::Leaf,
];

/// Magic colors for magic particles (random color selection)
pub const MAGIC_COLORS: &[Element] = &[
    Element::Wall,
    Element::Plant,
    Element::Spout,
    Element::Well,
    Element::Wax,
    Element::Ice,
    Element::Branch,
    Element::Leaf,
];

