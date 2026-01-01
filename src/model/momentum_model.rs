use candle_core::{DType, Device, Result, Tensor, Var};
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, Normal};

pub struct MomentumModel {
    dev: Device,
    ws: Var,
    ms: Vec<Tensor>,
    alfa: f64,
    lr: f64,
}

impl MomentumModel {
    pub fn new(
        dev: Device,
        dim: usize,
        levels: usize,
        alfa: f64,
        lr: f64,
        seed: u64,
    ) -> Result<Self> {
        let mut rng = StdRng::seed_from_u64(seed);
        let ws = Var::from_tensor(&Self::randn(&mut rng, dim + 1, 0.0, 2.0, &dev)?)?;

        let mut ms = Vec::with_capacity(levels);
        for _ in 0..levels {
            ms.push(Tensor::zeros(dim + 1, DType::F64, &dev)?);
        }

        Ok(Self {
            dev,
            ws,
            ms,
            alfa,
            lr,
        })
    }

    fn randn(rng: &mut StdRng, dim: usize, mean: f64, std: f64, dev: &Device) -> Result<Tensor> {
        let normal = Normal::new(mean, std).unwrap();
        let data: Vec<f64> = (0..dim).map(|_| normal.sample(rng)).collect();
        Tensor::from_vec(data, dim, dev)
    }

    pub fn step(&mut self, x: &Tensor, y: Option<&Tensor>) -> Result<(Tensor, Option<Tensor>)> {
        let one = Tensor::ones(1, DType::F64, &self.dev)?;
        let x_aug = Tensor::cat(&[x, &one], 0)?;
        let dot = x_aug.mul(self.ws.as_tensor())?.sum_all()?;
        let yp = dot.reshape(1)?;

        let loss = if let Some(y) = y {
            let diff = yp.sub(y)?;
            let loss = diff.sqr()?.mean_all()?;

            let grads = loss.backward()?;
            if let Some(grad) = grads.get(self.ws.as_tensor()) {
                let last = self.ms.len() - 1;
                self.ms[last] = grad.clone();
                for i in (0..last).rev() {
                    let scaled_next = self.ms[i + 1].affine(1.0 - self.alfa, 0.0)?;
                    self.ms[i] = self.ms[i].affine(self.alfa, 0.0)?.add(&scaled_next)?;
                }
                // ws[0] = ws[0] - (1 - alfa) * ws[1]  o  ws[0] - scaled_grad si levels=1
                let new_w = self
                    .ws
                    .as_tensor()
                    .sub(&(self.ms[0].affine(self.lr, 0.0)?))?;

                self.ws.set(&new_w)?;
            }

            Some(loss)
        } else {
            None
        };

        Ok((yp, loss))
    }
}
