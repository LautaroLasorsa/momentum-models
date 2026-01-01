use candle_core::Device;

#[derive(Clone)]
pub struct Config {
    pub model_levels: Vec<usize>,
    pub data_levels: Vec<usize>,
    pub dim: usize,
    pub alpha: f64,
    pub dev: Device,
    pub samples: usize,
    pub warm_up: usize,
    pub model_seed: u64,
    pub data_seed: u64,
    pub lr: f64,
    pub n_threads: usize,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            model_levels: (1usize..20).collect(),
            data_levels: (1usize..20).collect(),
            dim: 10,
            alpha: 0.95,
            dev: Device::cuda_if_available(0).unwrap_or(Device::Cpu),
            samples: 30000,
            warm_up: 3000,
            model_seed: 42,
            data_seed: 123,
            lr: 0.01,
            n_threads: 8,
        }
    }
}
