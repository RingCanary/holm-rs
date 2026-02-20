#[derive(Debug, Clone)]
pub struct Vec1D {
    pub data: Vec<f32>,
}

impl Vec1D {
    pub fn zeros(n: usize) -> Self {
        Self { data: vec![0.0; n] }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }
}
