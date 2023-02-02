use std::error::Error;
use std::fs;
use std::path::PathBuf;
use std::iter::zip;
use rand::thread_rng;
use rand::distributions::Distribution;
use statrs::distribution::{Beta, Normal};
use ordered_float::OrderedFloat;
use serde::Deserialize;
use std::collections::VecDeque;

use rand::seq::SliceRandom;
use rand::Rng;
use rv::data::Partition;
use rv::dist::{Crp, Gaussian, NormalInvGamma};
use rv::misc::ln_pflip;
use rv::traits::*;
use rv::ConjugateModel;
use std::sync::Arc;



#[cfg(test)]
mod tests {
    use super::*;

    fn set_up_test_particles() -> Particles {
        let mut live: VecDeque<Particle> = VecDeque::new();
        let dead: Vec<Particle> = Vec::new();
        let mut eps = 0.0;
        let mut w = 0.1;
        let mut i = 0;
        for i in 0..3 {
            let theta = vec![i as f64; 2];
            let yhat = vec![(i+1) as f64; 2];
            let part = Particle::new_with_all(
                eps,
                theta,
                yhat,
                w,
                i
            );
            live.push_back(part);
            eps += 1.0;
            w *= 0.5;
        }
        Particles::new_with_particles(live, dead)
    }

    #[test]
    fn test_move_worst() {
        let mut particles = set_up_test_particles();
        assert_eq!(particles.live[0].eps, 0.0);
        assert_eq!(particles.live.len(), 3);
        assert_eq!(particles.dead.len(), 0);

        particles.move_worst_to_dead();
        assert_eq!(particles.live[0].eps, 1.0);
        assert_eq!(particles.dead[0].eps, 0.0);
        assert_eq!(particles.live.len(), 2);
        assert_eq!(particles.dead.len(), 1);

        particles.move_worst_to_dead();
        assert_eq!(particles.live[0].eps, 2.0);
        assert_eq!(particles.dead[1].eps, 1.0);
        assert_eq!(particles.live.len(), 1);
        assert_eq!(particles.dead.len(), 2);
    }

    #[test]
    fn test_update_worst() {
        let mut particles = set_up_test_particles();
        particles.update_worst(5.0, 7);
        assert_eq!(particles.live[0].i, 7);
        assert_eq!(particles.live[0].w, 5.0);
    }

    #[test]
    fn test_add_to_live() {
        let mut particles = set_up_test_particles();
        println!("{:?}", particles);
        let part = Particle::new_with_all(
            0.5,
            vec![-1.0; 2],
            vec![-1.0; 2],
            0.000001,
            20,
        );
        particles.add_to_live(part).unwrap();
        assert_eq!(particles.live.len(), 4);
        assert_eq!(particles.dead.len(), 0);
        assert_eq!(particles.live[1].eps, 0.5);
        assert_eq!(particles.live[1].w, 0.000001);

        let part = Particle::new_with_all(
            0.4,
            vec![-1.0; 2],
            vec![-1.0; 2],
            0.111,
            20,
        );
        particles.add_to_live(part).unwrap();
        assert_eq!(particles.live.len(), 5);
        assert_eq!(particles.live[1].eps, 0.4);
        assert_eq!(particles.live[1].w, 0.111);
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
///
/// Fields:
/// eps: the likelihood of this particle
/// theta: the particle's parameter vectors
/// yhat: the y-values implied by the particle's parameters
/// w: the weight
/// i: the iteraction at which this particle was allocated to the dead set
#[derive(Debug)]
struct Particle {
    eps: f64,
    theta: Vec<f64>,
    yhat: Vec<f64>,
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

    fn new_with_all(
            eps: f64,
            theta: Vec<f64>,
            yhat: Vec<f64>,
            w: f64,
            i: usize,
    ) -> Particle {
        Particle{ eps, theta, yhat, w, i }
    }

    fn run(&mut self) {
    }

    fn update_log_lik(&mut self) {
    }
}


/// contains the sets of live and dead particles
/// could contain bayesian evidence, err, etc.
#[derive(Debug)]
struct Particles {
    live: VecDeque<Particle>,
    dead: Vec<Particle>,
}


impl Particles {
    fn new(
            particle_num: usize,
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

        for (mu_i, sd_i) in zip(mu, sd) {
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

    fn new_with_particles(
            live: VecDeque<Particle>,
            dead: Vec<Particle>,
    ) -> Particles {
        Particles{ live, dead }
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

    fn update_worst(&mut self, w: f64, iter: usize) {
        self.live[0].i = iter;
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
    let mut particles = Particles::new(
        config.particle_num,
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
        particles.update_worst(w_i, i);
        particles.move_worst_to_dead();
        particles.sample_to_live();

    }

    Ok(())
}

// Copied from https://gitlab.com/baxe/rv/-/blob/master/examples/dpgmm.rs on 2023-02-02
// Dirichlet Process Mixture Model
// -------------------------------
//
// In this example, we're going to build a Dirichlet Process Mixture Model
// (DPMM). In a typical mixture model, we assume we know the number of
// copmonents and learn the parameters for each component that best fit the
// data. For example, we might use a 2-component model to fit to bi-modal data.
// The DPMM uses a probabilistic process -- the Diriclet Process -- to describe
// how data are assigned to components, and does inference on the parameters of
// that process as well as the component parameters. The DPMM weighs simplicity
// (prefer fewer componets) with explanation.
//
// Below, we implement the collapsed Gibbs algorithm for sampling from s DPMM.
// The code is generic to any type of mixture as long as it has a conjugate
// prior.
//
// References
// ----------
//
// Neal, R. M. (2000). Markov chain sampling methods for Dirichlet process
//     mixture models. Journal of computational and graphical statistics, 9(2),
//     249-265.
//
// Rasmussen, C. E. (1999, December). The infinite Gaussian mixture model. In
//     NIPS (Vol. 12, pp. 554-560).

// Infinite mixture (CRP) model
//
// This code is general to any type of mixture as long as it has a conjugate
// prior
struct Dpmm<X, Fx, Pr>
where
    Fx: Rv<X> + HasSuffStat<X>,
    Pr: ConjugatePrior<X, Fx>,
{
    // The data
    xs: Vec<X>,
    // Keeps track of the data IDs as they're removed and replaced
    ixs: Vec<usize>,
    // The prior on the partition of data
    crp: Crp,
    // The current partition
    partition: Partition,
    // The Prior on each of the components.
    prior: Arc<Pr>,
    // A vector of component models with conjugate priors
    components: Vec<ConjugateModel<X, Fx, Pr>>,
}

impl<X, Fx, Pr> Dpmm<X, Fx, Pr>
where
    Fx: Rv<X> + HasSuffStat<X>,
    Pr: ConjugatePrior<X, Fx>,
{
    // Draws a Dpmm from the prior
    fn new<R: Rng>(xs: Vec<X>, prior: Pr, alpha: f64, rng: &mut R) -> Self {
        let n = xs.len();

        // Partition prior
        let crp = Crp::new(alpha, n).expect("Invalid params");

        // Initial partition drawn from the prior
        let partition = crp.draw(rng);

        // Put the prior in a reference counter
        let prior_arc = Arc::new(prior);

        // Create an empty component for each partition. Drawing component
        // models is used as a template; The parameters don't matter because we
        // marginalize them away through the magic of conjugate priors.
        let mut components: Vec<ConjugateModel<X, Fx, Pr>> = (0..partition.k())
            .map(|_| {
                ConjugateModel::new(&prior_arc.draw(rng), prior_arc.clone())
            })
            .collect();

        // Given the data to their respective components by having them observe
        // their data.
        xs.iter()
            .zip(partition.z().iter())
            .for_each(|(xi, &zi)| components[zi].observe(xi));

        Dpmm {
            xs,
            ixs: (0..n).collect(),
            crp,
            partition,
            prior: prior_arc,
            components,
        }
    }

    // Number of data
    fn n(&self) -> usize {
        self.xs.len()
    }

    /// Remove and return the datum at index `ix`. Return the datum and its
    /// index.
    fn remove(&mut self, pos: usize) -> (X, usize) {
        let x = self.xs.remove(pos);
        let ix = self.ixs.remove(pos);
        let zi = self.partition.z()[pos];

        let is_singleton = self.partition.counts()[zi] == 1;
        self.partition.remove(pos).expect("could not remove");

        // If x was in a component by itself, remove that component; otherwise
        // have that component forget it.
        if is_singleton {
            let _cj = self.components.remove(zi);
        } else {
            self.components[zi].forget(&x);
            assert!(self.components[zi].n() > 0);
        }

        (x, ix)
    }

    // For a datum `x` with index `ix`, assigns `x` to a partition
    // probabilistically according to the DPGMM. The datum is appended to the
    // end of `xs` and the assignment, `z`.
    fn insert<R: Rng>(&mut self, x: X, ix: usize, rng: &mut R) {
        let mut ln_weights: Vec<f64> = self
            .partition
            .counts()
            .iter()
            .zip(self.components.iter())
            .map(|(&w, cj)| (w as f64).ln() + cj.ln_pp(&x)) // nk * p(xi|xk)
            .collect();

        let mut ctmp: ConjugateModel<X, Fx, Pr> =
            ConjugateModel::new(&self.prior.draw(rng), self.prior.clone());

        // probability of being in a new category -- α * p(xi)
        ln_weights.push(self.crp.alpha().ln() + ctmp.ln_pp(&x));

        // Draws a new assignment in proportion with the weights
        let zi = ln_pflip(&ln_weights, 1, false, rng)[0];

        // Here is where we re-insert the data back into xs, ixs, and the
        // partition.
        if zi == self.partition.k() {
            // If we've created a singleton, we must push a new component
            ctmp.observe(&x);
            self.components.push(ctmp);
        }

        // Push x, ix, and zi to the end of the list
        self.components[zi].observe(&x);
        self.xs.push(x);
        self.ixs.push(ix);
        self.partition.append(zi).expect("Could not append");
    }

    // reassigns a the datum at the position `pos`
    fn step<R: Rng>(&mut self, pos: usize, rng: &mut R) {
        let (x, ix) = self.remove(pos);
        self.insert(x, ix, rng);
    }

    // Reassigns each datum in random order
    fn scan<R: Rng>(&mut self, rng: &mut R) {
        let mut positions: Vec<usize> = (0..self.n()).collect();
        positions.shuffle(rng);
        positions.iter().for_each(|&pos| self.step(pos, rng));
    }

    // Run the DPGMM for `iters` iterations
    fn run<R: Rng>(&mut self, iters: usize, rng: &mut R) {
        (0..iters).for_each(|_| self.scan(rng));
        self.sort() // restore data/assignment order
    }

    // The data get shuffled as a result of the removal/insertion process, so we
    // need to re-sort the data by their indices to ensure the data and the
    // assignment are in the same order they were when they were passed in
    fn sort(&mut self) {
        // This will at most do n swaps, but I feel like there's probably some
        // really obvious way to do better. Oh well... I'm an ML guy, not an
        // algorithms guy.
        for i in 0..self.n() {
            while self.ixs[i] != i {
                let j = self.ixs[i];
                self.ixs.swap(i, j);
                self.partition.z_mut().swap(i, j);
                self.xs.swap(i, j);
            }
        }
    }
}

fn main() {
    let mut rng = rand::thread_rng();

    // Generate 100 data from two Gaussians. The Gaussians are far enough apart
    // that the DPGMM should separate them.
    let mut xs: Vec<f64> =
        Gaussian::new(-3.0, 1.0).unwrap().sample(50, &mut rng);
    let mut ys: Vec<f64> =
        Gaussian::new(3.0, 1.0).unwrap().sample(50, &mut rng);
    xs.append(&mut ys);

    // Parameters are more or less arbitrary. The only thing we need to worry
    // about is scale.
    let prior = NormalInvGamma::new(0.0, 1.0, 1.0, 1.0).unwrap();

    // Draw a DPGMM from the prior
    let mut dpgmm = Dpmm::new(xs, prior, 1.0, &mut rng);

    // .. and run it
    dpgmm.run(200, &mut rng);

    // there should be two categories, the first half belong to one category,
    // and the second half belong to the other. Something like
    // [0, 0, 0, 0, ...,0, 1, ..., 1, 1, 1, 1] -- subject to some noise,
    // because we don't actually know how many components there are.
    let mut zs_a = dpgmm.partition.z().clone();
    let zs_b = zs_a.split_off(50);
    println!("{:?}", zs_a);
    println!("{:?}", zs_b);
}
