use textplots::{Chart, Plot, Shape};

#[allow(dead_code)]
pub fn plot_losses(losses: &[f64], window_size: usize) {
    if losses.is_empty() {
        return;
    }

    let raw_points: Vec<(f32, f32)> = losses
        .iter()
        .enumerate()
        .map(|(i, &l)| (i as f32, l as f32))
        .collect();

    let smoothed: Vec<f64> = losses
        .windows(window_size)
        .map(|w| w.iter().sum::<f64>() / w.len() as f64)
        .collect();

    let smoothed_points: Vec<(f32, f32)> = smoothed
        .iter()
        .enumerate()
        .map(|(i, &l)| ((i + window_size / 2) as f32, l as f32))
        .collect();

    let max_x = losses.len() as f32;

    println!("\nLoss (raw):");
    Chart::new(120, 30, 0.0, max_x)
        .lineplot(&Shape::Lines(&raw_points))
        .display();

    println!("\nLoss (smoothed, window={}):", window_size);
    Chart::new(120, 30, 0.0, max_x)
        .lineplot(&Shape::Lines(&smoothed_points))
        .display();
}

pub fn print_table(data: &[Vec<f64>], row_headers: &[usize], col_headers: &[usize]) {
    let col_width = 12;

    // Header row
    print!("{:>col_width$}", "");
    for col in col_headers {
        print!("{:>col_width$}", col);
    }
    println!();

    // Data rows
    for (row_idx, row) in data.iter().enumerate() {
        print!("{:>col_width$}", row_headers[row_idx]);
        for val in row {
            print!("{:>col_width$.6}", val);
        }
        println!();
    }
}
