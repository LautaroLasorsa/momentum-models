use candle_core::{Device, Result, Tensor};
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, Normal};

pub struct InputProducer {
    dev: Device,
    ws: Vec<Tensor>,
    alfa: f64,
    rng: StdRng,
    dim: usize,
}

impl InputProducer {
    pub fn new(dev: Device, dim: usize, levels: usize, alfa: f64, seed: u64) -> Result<Self> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut ws = Vec::with_capacity(levels);
        for _ in 0..levels {
            ws.push(Self::randn(&mut rng, dim, 0.0, 1.0, &dev)?);
        }
        Ok(Self {
            dev,
            ws,
            alfa,
            rng,
            dim,
        })
    }

    fn randn(rng: &mut StdRng, dim: usize, mean: f64, std: f64, dev: &Device) -> Result<Tensor> {
        let normal = Normal::new(mean, std).unwrap();
        let data: Vec<f64> = (0..dim).map(|_| normal.sample(rng)).collect();
        Tensor::from_vec(data, dim, dev)
    }

    fn randn_scalar(rng: &mut StdRng, mean: f64, std: f64, dev: &Device) -> Result<Tensor> {
        let normal = Normal::new(mean, std).unwrap();
        let val = normal.sample(rng);
        Tensor::new(val, dev)
    }

    pub fn next(&mut self) -> Result<(Tensor, Tensor)> {
        let last_idx = self.ws.len() - 1;

        let dw = Self::randn(&mut self.rng, self.dim, 0.0, 1.0, &self.dev)?;
        let scaled_last = self.ws[last_idx].affine(self.alfa, 0.0)?;
        let scaled_dw = dw.affine(1.0 - self.alfa, 0.0)?;
        self.ws[last_idx] = scaled_last.add(&scaled_dw)?;

        for i in (0..last_idx).rev() {
            let scaled_current = self.ws[i].affine(self.alfa, 0.0)?;
            let scaled_next = self.ws[i + 1].affine(1.0 - self.alfa, 0.0)?;
            self.ws[i] = scaled_current.add(&scaled_next)?;
        }

        let x = Self::randn(&mut self.rng, self.dim, 0.0, 1.0, &self.dev)?;
        let noise = Self::randn_scalar(&mut self.rng, 0.0, 0.1, &self.dev)?;
        let dot = x.mul(&self.ws[0])?.sum_all()?;
        let y = dot.add(&noise)?.reshape(1)?;

        Ok((x, y))
    }
}
