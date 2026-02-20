use crate::rwkv::state::LayerState;
use crate::rwkv::tensor::Vec1D;

#[derive(Debug, Clone)]
pub struct RwkvBlock {
    pub dim: usize,
}

impl RwkvBlock {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    pub fn forward_step(&self, x: &Vec1D, state: &mut LayerState) -> Vec1D {
        let _ = self.dim;
        state.prev_x_time = x.clone();
        state.prev_x_channel = x.clone();
        x.clone()
    }
}
