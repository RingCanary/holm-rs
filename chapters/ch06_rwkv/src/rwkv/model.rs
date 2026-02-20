use crate::rwkv::block::RwkvBlock;
use crate::rwkv::state::LayerState;
use crate::rwkv::tensor::Vec1D;

#[derive(Debug, Clone)]
pub struct TinyRwkv {
    pub dim: usize,
    pub blocks: Vec<RwkvBlock>,
    pub states: Vec<LayerState>,
}

impl TinyRwkv {
    pub fn new(n_layers: usize, dim: usize) -> Self {
        let blocks = (0..n_layers).map(|_| RwkvBlock::new(dim)).collect();
        let states = (0..n_layers).map(|_| LayerState::zeros(dim)).collect();
        Self {
            dim,
            blocks,
            states,
        }
    }

    pub fn step(&mut self, x: &Vec1D) -> Vec1D {
        let mut h = x.clone();
        for (block, state) in self.blocks.iter().zip(self.states.iter_mut()) {
            h = block.forward_step(&h, state);
        }
        h
    }
}
