use crate::{SIZE, WORKGROUP_SIZE};
use bevy::{
    prelude::*,
    render::{
        extract_resource::ExtractResource,
        render_graph::{self, RenderLabel},
        render_resource::{CachedPipelineState, ComputePassDescriptor, PipelineCache, BindGroup, BindGroupLayout, CachedComputePipelineId, ShaderType},
        renderer::RenderContext,
    },
    shader::PipelineCacheError,
};

pub struct FallingSandComputePlugin;

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct FallingSandLabel;

#[derive(Resource, Clone, ExtractResource)]
pub struct FallingSandImages {
    pub texture_a: Handle<Image>,
    pub texture_b: Handle<Image>,
}

#[derive(Resource, Clone, Copy, ExtractResource, ShaderType)]
pub struct FallingSandUniforms {
    pub draw_color: LinearRgba, // Color for drawing (encoded element)
    pub size: UVec2,
    pub click_position: IVec2,
    pub click_radius: f32,     // Radius of the circle for placing/removing sand
    pub click_action: u32,     // 0 = no action, 1 = add element, 2 = remove element
    pub color_shift_enabled: u32, // 1 = true, 0 = false
    pub selected_element: u32, // Element type index for drawing
    pub fall_into_void: u32,   // 1 = true, 0 = false - elements disappear at bottom edge
    pub overwrite_mode: u32,   // 1 = true, 0 = false - overwrite existing elements when clicking
    // Spigot data (4 spigots)
    pub spigot_0_x: u32,
    pub spigot_0_width: u32,
    pub spigot_0_color: LinearRgba,
    pub spigot_0_enabled: u32,
    pub spigot_1_x: u32,
    pub spigot_1_width: u32,
    pub spigot_1_color: LinearRgba,
    pub spigot_1_enabled: u32,
    pub spigot_2_x: u32,
    pub spigot_2_width: u32,
    pub spigot_2_color: LinearRgba,
    pub spigot_2_enabled: u32,
    pub spigot_3_x: u32,
    pub spigot_3_width: u32,
    pub spigot_3_color: LinearRgba,
    pub spigot_3_enabled: u32,
}

#[derive(Resource)]
pub struct FallingSandPipeline {
    pub texture_bind_group_layout: BindGroupLayout,
    pub init_pipeline: CachedComputePipelineId,
    pub update_pipeline: CachedComputePipelineId,
}

#[derive(Resource)]
pub struct FallingSandImageBindGroups(pub [BindGroup; 2]);

pub struct FallingSandNode {
    pub state: FallingSandState,
}

pub enum FallingSandState {
    Loading,
    Init,
    Update(usize),
}

impl Plugin for FallingSandComputePlugin {
    fn build(&self, _app: &mut App) {
        // Compute shader plugin disabled - using CPU simulation now
        // The plugin is kept for type definitions but the implementation is disabled
    }
}

impl Default for FallingSandNode {
    fn default() -> Self {
        Self {
            state: FallingSandState::Loading,
        }
    }
}

impl render_graph::Node for FallingSandNode {
    fn update(&mut self, world: &mut World) {
        let pipeline = world.resource::<FallingSandPipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();

        // if the corresponding pipeline has loaded, transition to the next stage
        match self.state {
            FallingSandState::Loading => {
                match pipeline_cache.get_compute_pipeline_state(pipeline.init_pipeline) {
                    CachedPipelineState::Ok(_) => {
                        self.state = FallingSandState::Init;
                    }
                    // If the shader hasn't loaded yet, just wait.
                    CachedPipelineState::Err(PipelineCacheError::ShaderNotLoaded(_)) => {}
                    CachedPipelineState::Err(err) => {
                        panic!("Initializing compute shader:\n{err}")
                    }
                    _ => {}
                }
            }
            FallingSandState::Init => {
                if let CachedPipelineState::Ok(_) =
                    pipeline_cache.get_compute_pipeline_state(pipeline.update_pipeline)
                {
                    self.state = FallingSandState::Update(1);
                }
            }
            FallingSandState::Update(0) => {
                self.state = FallingSandState::Update(1);
            }
            FallingSandState::Update(1) => {
                self.state = FallingSandState::Update(0);
            }
            FallingSandState::Update(_) => unreachable!(),
        }
    }

    fn run(
        &self,
        _graph: &mut render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), render_graph::NodeRunError> {
        let bind_groups = &world.resource::<FallingSandImageBindGroups>().0;
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = world.resource::<FallingSandPipeline>();

        let mut pass = render_context
            .command_encoder()
            .begin_compute_pass(&ComputePassDescriptor::default());

        // select the pipeline based on the current state
        match self.state {
            FallingSandState::Loading => {}
            FallingSandState::Init => {
                let init_pipeline = pipeline_cache
                    .get_compute_pipeline(pipeline.init_pipeline)
                    .unwrap();
                pass.set_bind_group(0, &bind_groups[0], &[]);
                pass.set_pipeline(init_pipeline);
                pass.dispatch_workgroups(SIZE.x / WORKGROUP_SIZE, SIZE.y / WORKGROUP_SIZE, 1);
            }
            FallingSandState::Update(index) => {
                let update_pipeline = pipeline_cache
                    .get_compute_pipeline(pipeline.update_pipeline)
                    .unwrap();
                pass.set_bind_group(0, &bind_groups[index], &[]);
                pass.set_pipeline(update_pipeline);
                pass.dispatch_workgroups(SIZE.x / WORKGROUP_SIZE, SIZE.y / WORKGROUP_SIZE, 1);
            }
        }

        Ok(())
    }
}
