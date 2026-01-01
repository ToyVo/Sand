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
