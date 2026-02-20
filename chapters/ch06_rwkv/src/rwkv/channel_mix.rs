use crate::rwkv::tensor::Vec1D;

pub fn squared_relu(x: &Vec1D) -> Vec1D {
    let out = x
        .data
        .iter()
        .map(|v| if *v > 0.0 { v * v } else { 0.0 })
        .collect();
    Vec1D { data: out }
}
