use super::config_step::Config;
use linregress::{FormulaRegressionBuilder, RegressionDataBuilder};
use statrs::distribution::{ContinuousCDF, StudentsT};

pub fn pivot(
    cols: &Vec<f64>,
    rows: &Vec<f64>,
    matrix: &Vec<Vec<f64>>,
) -> Result<Vec<Vec<f64>>, String> {
    if rows.len() != matrix.len() {
        return Err("Rows and matrix must have the same length".to_string());
    }
    let mut result: Vec<Vec<f64>> = Vec::new();
    for ri in 0..rows.len() {
        if cols.len() != matrix[ri].len() {
            return Err("Columns and row must have the same length".to_string());
        }
        for ci in 0..cols.len() {
            result.push(vec![rows[ri], cols[ci], matrix[ri][ci]]);
        }
    }
    Ok(result)
}

fn significance_stars(p: f64) -> &'static str {
    if p < 0.001 {
        "***"
    } else if p < 0.01 {
        "**"
    } else if p < 0.05 {
        "*"
    } else if p < 0.1 {
        "."
    } else {
        ""
    }
}

pub fn linear_model(conf: &Config, results: &Vec<Vec<f64>>, log: bool) -> Result<(), String> {
    let observations = pivot(
        &conf.model_levels.iter().map(|&x| x as f64).collect(),
        &conf.data_levels.iter().map(|&x| x as f64).collect(),
        &results,
    )?;

    let model_levels: Vec<f64> = observations.iter().map(|row| row[0]).collect();
    let data_levels: Vec<f64> = observations.iter().map(|row| row[1]).collect();
    let losses: Vec<f64> = observations
        .iter()
        .map(|row| if log { row[2].ln() } else { row[2] })
        .collect();

    let loss_label = if log { "log(loss)" } else { "loss" };
    let n = observations.len();

    let data = RegressionDataBuilder::new()
        .build_from(vec![
            (loss_label, losses),
            ("model_level", model_levels),
            ("data_level", data_levels),
        ])
        .map_err(|e| e.to_string())?;

    let model = FormulaRegressionBuilder::new()
        .data(&data)
        .formula(format!("{loss_label} ~ model_level + data_level"))
        .fit()
        .map_err(|e| e.to_string())?;

    // Calcular t-value para 95% CI
    let k = 3; // intercept + 2 predictores
    let df = (n - k) as f64;
    let t_dist = StudentsT::new(0.0, 1.0, df).map_err(|e| e.to_string())?;
    let t_value = t_dist.inverse_cdf(0.975);

    // Recolectar datos
    let params: Vec<f64> = model.parameters().to_vec();
    let p_values: Vec<f64> = model.p_values().to_vec();
    let se_pairs: Vec<f64> = model.iter_se_pairs().map(|(_, se)| se).collect();

    // iter_se_pairs no incluye intercept, pero parameters/p_values sí
    // Calcular SE del intercept desde p-value y parámetro: t = param/se, p = 2*(1-cdf(|t|))
    let intercept_se = if p_values[0] > 0.0 && p_values[0] < 1.0 {
        let t_stat = t_dist.inverse_cdf(1.0 - p_values[0] / 2.0);
        (params[0] / t_stat).abs()
    } else {
        0.0
    };

    let mut se = vec![intercept_se];
    se.extend(se_pairs);

    let names = vec!["(Intercept)", "model_level", "data_level"];

    // Imprimir resultados
    println!(
        "\n=== Linear Regression: {} ~ model_level + data_level ===",
        loss_label
    );
    println!("n = {}, R² = {:.4}\n", n, model.rsquared());
    println!(
        "{:<15} {:>12} {:>12} {:>12} {:>12} {:>8}",
        "Parameter", "Estimate", "Std.Err", "CI Lower", "CI Upper", "Pr(>|t|)"
    );
    println!("{}", "-".repeat(75));

    for i in 0..params.len() {
        let ci_lower = params[i] - t_value * se[i];
        let ci_upper = params[i] + t_value * se[i];
        let stars = significance_stars(p_values[i]);

        println!(
            "{:<15} {:>12.4} {:>12.4} {:>12.4} {:>12.4} {:>8.4} {}",
            names[i], params[i], se[i], ci_lower, ci_upper, p_values[i], stars
        );
    }

    println!("{}", "-".repeat(75));
    println!("Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1\n");

    Ok(())
}
