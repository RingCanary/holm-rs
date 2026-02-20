use crate::rwkv::tensor::Vec1D;

#[derive(Debug, Clone)]
pub struct LayerState {
    pub prev_x_time: Vec1D,
    pub num: Vec1D,
    pub den: Vec1D,
    pub prev_x_channel: Vec1D,
}

impl LayerState {
    pub fn zeros(dim: usize) -> Self {
        Self {
            prev_x_time: Vec1D::zeros(dim),
            num: Vec1D::zeros(dim),
            den: Vec1D::zeros(dim),
            prev_x_channel: Vec1D::zeros(dim),
        }
    }
}
