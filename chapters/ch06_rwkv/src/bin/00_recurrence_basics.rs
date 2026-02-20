//! Lesson 00: Recurrence Basics
//!
//! Demonstrates the core WKV-style recurrence used in RWKV:
//!   num_t = num_{t-1} * exp(-decay) + k_t * v_t
//!   den_t = den_{t-1} * exp(-decay) + k_t
//!   y_t = num_t / den_t
//!
//! This is a scalar (1D) version for educational clarity.

/// Single WKV recurrence step (scalar version).
///
/// # Arguments
/// * `num` - Previous numerator state
/// * `den` - Previous denominator state  
/// * `k` - Current key value
/// * `v` - Current value
/// * `decay` - Time decay factor (e^-decay applied)
/// * `bonus` - Additive bonus to numerator (helps gradient flow)
///
/// # Returns
/// (new_num, new_den, y) where y is the output at this step
fn wkv_step(num: f64, den: f64, k: f64, v: f64, decay: f64, bonus: f64) -> (f64, f64, f64) {
    let decay_factor = (-decay).exp();
    let new_num = num * decay_factor + k * v + bonus;
    let new_den = den * decay_factor + k;
    let y = new_num / new_den;
    (new_num, new_den, y)
}

fn main() {
    println!("Lesson 00: Recurrence Basics");
    println!("Target: understand running state update y_t = f(x_t, state_{{t-1}}).\n");

    // Parameters
    let decay = 1.0; // Controls how fast previous state fades
    let bonus = 0.0; // No bonus for this demo

    // Initial state
    let mut num = 0.0;
    let mut den = 0.0;

    // Hardcoded sequence of (k, v) pairs - 5 steps
    let sequence: [(f64, f64); 5] = [
        (1.0, 2.0), // Step 0
        (0.5, 3.0), // Step 1
        (2.0, 1.0), // Step 2
        (1.5, 4.0), // Step 3
        (0.8, 2.5), // Step 4
    ];

    println!("WKV Recurrence Demo (scalar, decay={decay}, bonus={bonus})");
    println!("{}", "-".repeat(60));
    println!(
        "{:>5} | {:>6} {:>6} | {:>10} {:>10} | {:>8}",
        "t", "k", "v", "num", "den", "y"
    );
    println!("{}", "-".repeat(60));

    let mut outputs: Vec<f64> = Vec::new();

    for (t, (k, v)) in sequence.iter().enumerate() {
        let (new_num, new_den, y) = wkv_step(num, den, *k, *v, decay, bonus);

        println!(
            "{:>5} | {:>6.2} {:>6.2} | {:>10.4} {:>10.4} | {:>8.4}",
            t, k, v, new_num, new_den, y
        );

        num = new_num;
        den = new_den;
        outputs.push(y);
    }

    println!("{}", "-".repeat(60));
    println!();

    // Sanity checks with tolerance
    let eps = 1e-6;

    // Check 1: Output at t=0 should equal v[0] (since initial state is zero)
    let y0_expected = 2.0; // v[0] when k[0]=1.0: num = 0 + 1*2 = 2, den = 0 + 1 = 1, y = 2/1 = 2
    assert!(
        (outputs[0] - y0_expected).abs() < eps,
        "Sanity check 1 failed: y[0]={} != expected {}",
        outputs[0],
        y0_expected
    );
    println!(
        "✓ Sanity check 1: y[0] = {} (expected {})",
        outputs[0], y0_expected
    );

    // Check 2: Verify recurrence manually for t=1
    // After t=0: num=2.0, den=1.0
    // At t=1: decay_factor = e^-1 ≈ 0.3679
    //         new_num = 2.0 * 0.3679 + 0.5 * 3.0 = 0.7358 + 1.5 = 2.2358
    //         new_den = 1.0 * 0.3679 + 0.5 = 0.8679
    //         y = 2.2358 / 0.8679 ≈ 2.576
    let decay_factor = (-decay).exp();
    let manual_y1 = (2.0 * decay_factor + 0.5 * 3.0) / (1.0 * decay_factor + 0.5);
    assert!(
        (outputs[1] - manual_y1).abs() < eps,
        "Sanity check 2 failed: y[1]={} != expected {}",
        outputs[1],
        manual_y1
    );
    println!(
        "✓ Sanity check 2: y[1] = {:.4} (manually computed: {:.4})",
        outputs[1], manual_y1
    );

    // Check 3: y should always be bounded between min(v) and max(v) for this setup
    // (This is a property of the WKV attention-like averaging)
    let v_min = sequence
        .iter()
        .map(|(_, v)| *v)
        .fold(f64::INFINITY, f64::min);
    let v_max = sequence
        .iter()
        .map(|(_, v)| *v)
        .fold(f64::NEG_INFINITY, f64::max);
    let y_min = outputs.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let y_max = outputs.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    assert!(
        y_min >= v_min - eps && y_max <= v_max + eps,
        "Sanity check 3 failed: outputs [{}, {}] exceed v range [{}, {}]",
        y_min,
        y_max,
        v_min,
        v_max
    );
    println!(
        "✓ Sanity check 3: all outputs within value range [{}, {}]",
        v_min, v_max
    );

    println!("\nAll sanity checks passed!");
    println!("\nKey insight: WKV maintains a weighted average of past values,");
    println!("where weights come from keys and decay controls memory length.");
}
