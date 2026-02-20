use crate::rwkv::tensor::Vec1D;

pub fn interpolate(x: &Vec1D, prev_x: &Vec1D, mix: &Vec1D) -> Vec1D {
    let mut out = vec![0.0; x.len()];
    for (i, v) in out.iter_mut().enumerate() {
        *v = x.data[i] * mix.data[i] + prev_x.data[i] * (1.0 - mix.data[i]);
    }
    Vec1D { data: out }
}
