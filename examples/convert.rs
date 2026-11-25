use std::{env, error::Error};

use flint::Image;

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = env::args_os().skip(1);
    let input = args.next().ok_or("usage: convert INPUT OUTPUT")?;
    let output = args.next().ok_or("usage: convert INPUT OUTPUT")?;
    if args.next().is_some() {
        return Err("usage: convert INPUT OUTPUT".into());
    }
    let image = Image::open(input)?;
    let plan = image.explain()?;
    let stats = image.save_with_stats(output)?;
    println!("backend={:?}", stats.backend);
    println!("peak_managed_bytes={}", stats.peak_managed_bytes);
    println!("estimated_managed_bytes={}", plan.managed_bytes);
    Ok(())
}
