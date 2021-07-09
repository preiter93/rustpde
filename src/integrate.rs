//! Integrate traits and examples
pub mod conv_term;
pub mod diffusion;
pub mod navier;
pub use conv_term::conv_term;
pub use navier::Navier2D;

const MAX_TIMESTEP: usize = 100000;

/// Integrate trait, step forward in time, and write results
pub trait Integrate {
    /// Update solution
    fn update(&mut self);
    /// Receive current time
    fn get_time(&self) -> f64;
    /// Get timestep
    fn get_dt(&self) -> f64;
    /// Write results (can be used as callback)
    fn write(&mut self);
}

/// Integrade pde, that implements the Integrate trait.
///
/// Specify save_intervall to force writing an output.
///
/// Stop Criteria:
/// 1. Timestep limit
/// 2. Time limit
pub fn integrate<T: Integrate>(mut pde: T, max_time: f64, save_intervall: Option<f64>) {
    let mut timestep: usize = 0;
    let eps_dt = pde.get_dt() * 1e-4;
    loop {
        // Update
        pde.update();
        timestep += 1;

        // Save
        if let Some(dt_save) = &save_intervall {
            if (pde.get_time() % dt_save) < pde.get_dt() / 2.
                || (pde.get_time() % dt_save) > dt_save - pde.get_dt() / 2.
            {
                //println!("Save at time: {:4.3}", pde.get_time());
                pde.write();
            }
        }

        // Break
        if pde.get_time() + eps_dt >= max_time {
            println!("time limit reached: {:?}", pde.get_time());
            break;
        }
        if timestep >= MAX_TIMESTEP {
            println!("timestep limit reached: {:?}", timestep);
            break;
        }
    }
}
