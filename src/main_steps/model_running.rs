use super::config_step::Config;
use crate::{input, model};
use candle_core::{self, Device, Tensor};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::sync::{Arc, Mutex};
use std::thread;

pub struct RunConfiguration<'a> {
    pub ml: usize,
    pub dl: usize,
    pub dev: &'a Device,
    pub dim: usize,
    pub alpha: f64,
    pub model_seed: u64,
    pub data_seed: u64,
    pub lr: f64,
    pub samples: usize,
    pub warm_up: usize,
    pub pb: Option<&'a ProgressBar>,
}

pub fn run_case(rc: RunConfiguration) -> Result<f64, candle_core::Error> {
    if let Some(pb) = rc.pb {
        pb.set_message(format!("ML={}, DL={}", rc.ml, rc.dl));
        pb.set_position(0);
    }

    let mut modeli = model::MomentumModel::new(
        rc.dev.clone(),
        rc.dim,
        rc.ml,
        rc.alpha,
        rc.lr,
        rc.model_seed,
    )?;
    let mut inputg =
        input::InputProducer::new(rc.dev.clone(), rc.dim, rc.dl, rc.alpha, rc.data_seed)?;
    let mut losses: Vec<Tensor> = Vec::with_capacity(rc.samples - rc.warm_up);

    for i in 0..rc.samples {
        let (x, y) = inputg.next()?;
        let (_yp, loss) = modeli.step(&x, Some(&y))?;
        if let Some(pb) = rc.pb {
            pb.inc(1);
        }
        if i >= rc.warm_up {
            losses.push(loss.unwrap());
        }
    }
    let avg_loss = Tensor::stack(&losses, 0)?.mean_all()?.to_scalar::<f64>()?;

    if let Some(pb) = rc.pb {
        pb.finish_with_message(format!("ML={}, DL={} done: {:2.2}", rc.ml, rc.dl, avg_loss));
    }
    Ok(avg_loss)
}

pub fn run_step(conf: &Config) -> Result<Vec<Vec<f64>>, candle_core::Error> {
    // Configurar thread pool
    let conf = conf.clone();
    rayon::ThreadPoolBuilder::new()
        .num_threads(conf.n_threads)
        .build_global()
        .unwrap();

    // Crear todas las combinaciones (ml, dl)
    let tasks: Vec<(usize, usize)> = conf
        .model_levels
        .iter()
        .flat_map(|&ml| conf.data_levels.iter().map(move |&dl| (ml, dl)))
        .collect();

    // Preparar progress bars (una por tarea)
    let mp = MultiProgress::new();
    let style = ProgressStyle::default_bar()
        .template("{msg:>15} [{bar:40.green/red}] {pos}/{len} ({eta})")
        .unwrap()
        .progress_chars("█▓░");

    let progress_bars: Vec<ProgressBar> = tasks
        .iter()
        .map(|(ml, dl)| {
            let pb = mp.add(ProgressBar::new(conf.samples as u64));
            pb.set_style(style.clone());
            pb.set_message(format!("ML={:2.}, DL={:2.}", ml, dl));
            pb
        })
        .collect();

    // Almacenar resultados
    let results: Arc<Mutex<Vec<(usize, usize, f64)>>> = Arc::new(Mutex::new(Vec::new()));

    // Ejecutar en paralelo en un hilo separado para que MultiProgress pueda renderizar
    let handle = {
        let results = Arc::clone(&results);
        thread::spawn(move || {
            tasks.par_iter().enumerate().for_each(|(idx, &(ml, dl))| {
                let pb = &progress_bars[idx];
                let local_dev = conf.dev.clone();

                let result = run_case(RunConfiguration {
                    ml,
                    dl,
                    dev: &local_dev,
                    dim: conf.dim,
                    alpha: conf.alpha,
                    model_seed: conf.model_seed,
                    data_seed: conf.data_seed,
                    lr: conf.lr,
                    samples: conf.samples,
                    warm_up: conf.warm_up,
                    pb: Some(pb),
                });

                if let Ok(avg_loss) = result {
                    results.lock().unwrap().push((ml, dl, avg_loss));
                }
            });
        })
    };

    // Esperar a que terminen todos los workers
    handle.join().unwrap();
    // Convertir resultados a matriz
    let results_vec = results.lock().unwrap();
    let mut matrix: Vec<Vec<f64>> =
        vec![vec![f64::NAN; conf.data_levels.len()]; conf.model_levels.len()];

    for &(ml, dl, loss) in results_vec.iter() {
        let row = conf.model_levels.iter().position(|&x| x == ml).unwrap();
        let col = conf.data_levels.iter().position(|&x| x == dl).unwrap();
        matrix[row][col] = loss;
    }
    Ok(matrix)
}
