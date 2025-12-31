use bevy::prelude::*;
use crate::particles::types::{Particle, ParticleType, MAX_NUM_PARTICLES};

/// Resource to manage all particles in the system
/// Uses a pool of pre-allocated particles to avoid allocation overhead
#[derive(Resource)]
pub struct ParticleList {
    /// All particles (pre-allocated pool)
    pub particles: Vec<Particle>,
    /// Indices of active particles
    pub active_indices: Vec<usize>,
    /// Indices of inactive particles (available for reuse)
    pub inactive_indices: Vec<usize>,
    /// Count of each particle type
    pub particle_counts: [u32; 11], // 11 particle types (0-10)
}

impl Default for ParticleList {
    fn default() -> Self {
        let mut particles = Vec::with_capacity(MAX_NUM_PARTICLES);
        let mut inactive_indices = Vec::with_capacity(MAX_NUM_PARTICLES);
        
        // Pre-allocate all particles
        for i in 0..MAX_NUM_PARTICLES {
            particles.push(Particle::new());
            inactive_indices.push(i);
        }
        
        Self {
            particles,
            active_indices: Vec::new(),
            inactive_indices,
            particle_counts: [0; 11],
        }
    }
}

impl ParticleList {
    /// Add an active particle at the given position
    /// Returns Some(particle_index) if successful, None if no particles available
    pub fn add_active_particle(
        &mut self,
        particle_type: ParticleType,
        x: f32,
        y: f32,
        grid_i: usize,
    ) -> Option<usize> {
        // Check if we have inactive particles available
        if self.inactive_indices.is_empty() {
            return None;
        }
        
        // Get an inactive particle
        let particle_idx = self.inactive_indices.pop().unwrap();
        let particle = &mut self.particles[particle_idx];
        
        // Initialize particle
        particle.reset();
        particle.particle_type = particle_type;
        particle.init_x = x;
        particle.init_y = y;
        particle.x = x;
        particle.y = y;
        particle.prev_x = x;
        particle.prev_y = y;
        particle.init_i = grid_i;
        particle.active = true;
        particle.action_iterations = 0;
        particle.reinitialized = false;
        
        // Move to active list
        self.active_indices.push(particle_idx);
        self.particle_counts[particle_type.index() as usize] += 1;
        
        Some(particle_idx)
    }
    
    /// Make a particle inactive (return it to the pool)
    pub fn make_particle_inactive(&mut self, particle_idx: usize) {
        let particle = &mut self.particles[particle_idx];
        if !particle.active {
            return; // Already inactive
        }
        
        let particle_type = particle.particle_type;
        particle.active = false;
        self.particle_counts[particle_type.index() as usize] -= 1;
        
        // Remove from active list
        if let Some(pos) = self.active_indices.iter().position(|&i| i == particle_idx) {
            self.active_indices.remove(pos);
        }
        
        // Add to inactive list
        self.inactive_indices.push(particle_idx);
        
        // Reset particle
        particle.reset();
    }
    
    /// Check if a particle type is currently active
    pub fn particle_active(&self, particle_type: ParticleType) -> bool {
        self.particle_counts[particle_type.index() as usize] > 0
    }
    
    /// Get count of active particles of a type
    pub fn particle_count(&self, particle_type: ParticleType) -> u32 {
        self.particle_counts[particle_type.index() as usize]
    }
    
    /// Reinitialize a particle to a new type
    pub fn reinitialize_particle(&mut self, particle_idx: usize, new_type: ParticleType) {
        let particle = &mut self.particles[particle_idx];
        if !particle.active {
            return;
        }
        
        let old_type = particle.particle_type;
        self.particle_counts[old_type.index() as usize] -= 1;
        self.particle_counts[new_type.index() as usize] += 1;
        
        particle.particle_type = new_type;
        particle.reinitialized = true;
        particle.action_iterations = 0;
    }
    
    /// Get all active particle indices
    pub fn active_particles(&self) -> &[usize] {
        &self.active_indices
    }
    
    /// Get mutable access to a particle
    pub fn get_particle_mut(&mut self, idx: usize) -> Option<&mut Particle> {
        self.particles.get_mut(idx)
    }
    
    /// Get read-only access to a particle
    pub fn get_particle(&self, idx: usize) -> Option<&Particle> {
        self.particles.get(idx)
    }
}

