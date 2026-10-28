//! Scores each sample from its log-normalized expression: a weighted sum over a signature's genes.

pub struct Scorer {
    /// one weight per gene; zero for genes outside the signature
    pub weights: Vec<f64>,
}

impl Scorer {
    pub fn new(weights: Vec<f64>) -> Self {
        Scorer { weights }
    }

    /// The sample's score: its expression weighted by the signature, divided by the total weight.
    pub fn score(&self, expression: &[f64]) -> f64 {
        let total: f64 = self.weights.iter().map(|w| w.abs()).sum();
        let sum: f64 = expression.iter().zip(&self.weights).map(|(x, w)| x * w).sum();
        sum / total
    }
}
