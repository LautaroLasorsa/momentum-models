mod input;
mod main_steps;
mod model;
mod printing;

#[allow(unused_imports)]
use main_steps::{config_step, meta_analysis, model_running};
use printing::to_console::print_table;

fn main() {
    let config = config_step::Config::default();
    let matrix = model_running::run_step(&config).unwrap();
    println!("\n=== Results Table ===");
    print_table(&matrix, &config.model_levels, &config.data_levels);
    let _ = meta_analysis::linear_model(&config, &matrix, false);
    let _ = meta_analysis::linear_model(&config, &matrix, true);
}
