//! Implementations of volumetric weight averages
use super::Field;
use crate::Real;
use crate::Spaced;
use ndarray::prelude::*;

impl<S: Spaced<f64, 2>> Field<S, Real, 2> {
    /// Return volumetric weighted average along axis
    /// # Example
    ///```
    /// use ndarray::{array, Axis};
    /// use rustpde::{chebyshev, Field2, Space2};
    /// let (nx, ny) = (64, 66);
    /// let mut field = Field2::new(Space2::new([chebyshev(nx), chebyshev(ny)]));
    /// let (nx, ny) = (6, 5);
    /// let mut field = Field2::new(Space2::new([chebyshev(nx), chebyshev(ny)]));
    /// for mut lane in field.v.lanes_mut(Axis(1)) {
    ///     for (i, vi) in lane.iter_mut().enumerate() {
    ///         *vi = i as f64;
    ///     }
    /// }
    /// assert!(field.average_axis(0) == array![0.0, 1.0, 2.0, 3.0, 4.0]);
    ///```
    pub fn average_axis(&self, axis: usize) -> Array1<f64> {
        let mut weighted_avg = Array2::<f64>::zeros(self.v.raw_dim());
        let length = self.x[axis][self.x[axis].len() - 1] - self.x[axis][0];
        ndarray::Zip::from(self.v.lanes(Axis(axis)))
            .and(weighted_avg.lanes_mut(Axis(axis)))
            .for_each(|ref v, mut s| {
                s.assign(&(v * &self.dx[axis]));
                s /= length.abs();
            });
        weighted_avg.sum_axis(Axis(axis))
    }

    /// Return volumetric weighted average
    /// # Example
    ///```
    /// use ndarray::{array, Axis};
    /// use rustpde::{chebyshev, Field2, Space2};
    /// let (nx, ny) = (64, 66);
    /// let mut field = Field2::new(Space2::new([chebyshev(nx), chebyshev(ny)]));
    /// let (nx, ny) = (6, 5);
    /// let mut field = Field2::new(Space2::new([chebyshev(nx), chebyshev(ny)]));
    /// for mut lane in field.v.lanes_mut(Axis(1)) {
    ///     for (i, vi) in lane.iter_mut().enumerate() {
    ///         *vi = i as f64;
    ///     }
    /// }
    /// assert!(field.average() == 2.);
    ///```
    pub fn average(&self) -> f64 {
        let mut avg_x = Array1::<f64>::zeros(self.dx[1].raw_dim());
        let length = self.x[1][self.x[1].len() - 1] - self.x[1][0];
        avg_x.assign(&(self.average_axis(0) * &self.dx[1]));
        avg_x /= length.abs();
        let avg = avg_x.sum_axis(Axis(0));
        avg[[]]
    }
}
