use std::io::{BufReader, BufWriter};
use std::error::Error;
use std::fs::File;
use std::fs;
use std::path::{Path, PathBuf};
use std::iter::zip;
use ndarray::prelude::*;
use rand::thread_rng;
use rand::distributions::Distribution;
use statrs::distribution::{Beta, Normal};
use serde_json;
use serde::{Serialize, Deserialize};
use std::process;
use bio_anno_rs::BEDGraphData;
use ordered_float::OrderedFloat;
use std::collections::VecDeque;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}


/// Simple struct to hold command line arguments
#[derive(Deserialize, Debug)]
pub struct Config {
    pub data_file: PathBuf,
    pub sample_num: usize,
    pub particle_num: usize,
    pub beta_num: usize,
    pub mu: Vec<f64>,
    pub sd: Vec<f64>,
}


trait Optimizer {
    fn log_lik(&self) -> f64;
    fn run_objective(&self) -> f64;
}


/// defines a particle
struct Particle {
    eps: f64,
    theta: Vec<f64>,
    yhat: Vec<f64>,
    // other stuff for NS weights calculation
    //x: f64,
    w: f64,
    i: usize,
}


impl Particle {
    fn new(theta: Vec<f64>) -> Particle {
        let eps = f64::NEG_INFINITY;
        let yhat: Vec<f64> = Vec::new();
        let w = 0.0;
        let i = 0;
        Particle{ eps, theta, yhat, w, i }
    }

    fn run(&mut self) {
    }

    fn update_log_lik(&mut self) {
    }
}


/// contains the sets of live and dead particles
/// could contain bayesian evidence, err, etc.
struct Particles {
    live: VecDeque<Particle>,
    dead: Vec<Particle>,
}


impl Particles {
    fn new(
            particle_num: usize,
            beta_num: usize,
            mu: &Vec<f64>,
            sd: &Vec<f64>,
            rng: &mut rand::rngs::ThreadRng,
    ) -> Result<Particles, Box<dyn Error>> {

        let mut live: VecDeque<Particle> = VecDeque::new();
        let mut priors: Vec<Vec<f64>> = Vec::new();

        // priors is a vec of vec<f64>
        // outer vec is particles, each inner vec
        // is a given particle's theta
        for _ in 0..particle_num {
            priors.push(Vec::new());
        }

        for (i,(mu_i, sd_i)) in zip(mu, sd).enumerate() {
            let beta_i: Vec<f64> = Normal::new(*mu_i, *sd_i)?
                .sample_iter(&mut *rng)
                .take(particle_num)
                .collect();
            // push each sample from the prior into each particle's
            // inner vec for the given prior
            for j in 0..particle_num {
                priors[j].push(beta_i[j]);
            }
        }

        // now that we have the priors samples, intantiate particles
        //  with their theta vecs, -Inf likelihood, and 0.0 weights
        for i in 0..particle_num {
            let theta = priors[i].to_vec();
            let particle = Particle::new(theta);
            live.push_back(particle);
        }

        // sort particles by likelihood
        live.make_contiguous().sort_unstable_by_key(|x| OrderedFloat(x.eps));
        let dead: Vec<Particle> = Vec::new();
        Ok(Particles{live, dead})
    }

    fn len(&self) -> usize {
        self.live.len()
    }

    fn sample_to_live(&mut self) {
    }

    fn add_to_live(&mut self, new_particle: Particle) -> Result<(), Box<dyn Error>> {
        let pos = self.live
            .binary_search_by_key(
                &OrderedFloat(new_particle.eps), |a| OrderedFloat(a.eps)
            ).unwrap_or_else(|e| e);
        self.live.insert(pos, new_particle);
        Ok(())
    }

    fn move_worst_to_dead(&mut self) {
        let worst = self.live.pop_front().unwrap();
        self.dead.push(worst);
    }

    fn update_worst(&mut self, eps: f64, w: f64, iter: usize) {
        self.live[0].i = iter;
        self.live[0].eps = eps;
        self.live[0].w = w;
    }
}


pub fn run(config: &Config) -> Result<(), Box<dyn Error>> {

    // read in observed y vals
    let y: Vec<f64> = fs::read_to_string(&config.data_file)?
        .split(' ')
        .map(|x| x.parse().unwrap())
        .collect();

    let mut rng = thread_rng();

    // set up live particles
    // each particle should only have loglik, beta vec, weight. Weights
    // should initialize to 0.0 and loglik to -Inf
    let particles = Particles::new(
        config.particle_num,
        config.beta_num,
        &config.mu,
        &config.sd,
        &mut rng,
    )?;

    let dist = Beta::new(1.0, particles.len() as f64)?;

    // sample new live particle with higher likelihood than current lowest in live set
    // use gaussian proc as described by Khammash?
    // use splines?

    // get vectors for weights and log-likelihoods
    //let mut w: Vec<f64> = Vec::new();
    //let mut l: Vec<f64> = Vec::new();

    let x_i = 1.0;
    // replace definite sample num with some convergence criterion
    //let mut converged = false;

    //while !converged {
    for i in 0..config.sample_num {

        // I'll use notations from Mikelson and Khammash, 2020
        // sample from Beta distribution to get the relative allocation of remaining
        // volume to this likelihood
        let t: f64 = dist.sample(&mut rng);

        let x_im = x_i;
        let x_i = t * x_im;
        let w_i = x_im - x_i;

        // simulate system

        //println!("Calculating log-likelihood.");
        //let l_i = 0.0; //signals.log_lik(&y)?;
        //println!("Log likelihood: {:?}", log_lik);
        particles.update_worst(w_i, l_i);
        particles.move_worst_to_dead();
        particles.sample_to_live();

    }

    Ok(())
}
