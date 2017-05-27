/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include "Eigen/Dense"

#include "particle_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	std::random_device rd;
    std::mt19937 gen(rd());
	std::normal_distribution<double> dist_x(x, std[0]);
	std::normal_distribution<double> dist_y(y, std[1]);
	std::normal_distribution<double> dist_theta(theta, std[2]);
	
	num_particles = 2;
	for(int i = 0; i < num_particles; i++){
		Particle p;
		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_x(gen);
		p.theta = dist_theta(gen);
		p.weight = 1;
		particles.push_back(p);
	}

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	double c = velocity/yaw_rate;
	double c2 = delta_t * yaw_rate;
	double c3 = delta_t * velocity;

	std::default_random_engine gen;
	std::normal_distribution<double> noise_x(0, std_pos[0]);
	std::normal_distribution<double> noise_y(0, std_pos[1]);
	std::normal_distribution<double> noise_theta(0, std_pos[2]);
	// going straight
	if(yaw_rate < 0.0001){
		for (auto p : particles){
			p.x = p.x + c3 * cos(p.theta) + noise_x(gen);
			p.y = p.y + c3 * sin(p.theta) + noise_y(gen);
			p.theta = p.theta + noise_theta(gen); 
		} 
	// turning
	}else{
		for (auto p : particles){
			p.x = p.x + c * (sin(p.theta + c2) - sin(p.theta)) + noise_x(gen);
			p.y = p.y + c * (cos(p.theta) - cos(p.theta + c2)) + noise_y(gen);
			p.theta = p.theta + c2 + noise_theta(gen);
		} 
	}
	

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for(int i = 0; i < observations.size(); i++){
		int id_min = -1;
		double min_dist = 999999.0;
		LandmarkObs obv = observations[i];
		for(int j = 0; j < predicted.size(); j++){
			LandmarkObs prd = predicted[j];
			double d = dist(obv.x,obv.y,prd.x,prd.y);
			if(d < min_dist){
				id_min = prd.id;
			}
		}

		if(id_min == -1){
			std::cout << "Error finding asscociation" << std::endl;
			observations[i].id = 0;
		}else{
			observations[i].id = id_min;
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html
	/*
	This is my workflow for updating weights (repeated for each particle):

1. Make list of all landmarks within sensor range of particle, call this `predicted_lm`
2. Convert all observations from local to global frame, call this `transformed_obs`
3. Perform `dataAssociation`. This will put the index of the `predicted_lm` nearest to each `transformed_obs` in the `id` field of the `transformed_obs` element.
4. Loop through all the `transformed_obs`. Use the saved index in the `id` to find the associated landmark and compute the gaussian. 
5. Multiply all the gaussian values together to get total probability of particle (the weight). (edited)
	*/
	

	for(int i = 0; i < num_particles; i++){
		auto p = particles[i];
		std::vector<LandmarkObs> predicted_lm;
		for (auto map_lm : map_landmarks.landmark_list){
			double d = dist(map_lm.x_f,map_lm.y_f,p.x,p.y);
			if(d <= sensor_range){
				LandmarkObs tmp;
				tmp.x = map_lm.x_f;
				tmp.y = map_lm.y_f;
				tmp.id = map_lm.id_i;
				predicted_lm.push_back(tmp);
			}
		}

		if(predicted_lm.size() != 0){
			std::vector<LandmarkObs> transformed_obs;
			for(auto obs : observations){
				LandmarkObs t_obs;

				//rotation
				t_obs.x = obs.x * cos(p.theta) - obs.y * sin(p.theta);
				t_obs.y = obs.x * sin(p.theta) + obs.y * cos(p.theta);

				//translation
				t_obs.x = obs.x + p.x;
				t_obs.y = obs.y + p.y;

				transformed_obs.push_back(t_obs);
			}

			dataAssociation(predicted_lm, transformed_obs);
			
			for(auto obs : transformed_obs){
				auto lm_it = find_if(begin(map_landmarks.landmark_list), end(map_landmarks.landmark_list), [=] (Map::single_landmark_s const& f) { 
					return (f.id_i == obs.id); 
				});

				bool found = (lm_it != end(map_landmarks.landmark_list));
				if(found){
					double pi = 3.14159265359;
					/*VectorXd x_d = VectorXd(2);
					x_d << obs.x, obs.y;
					VectorXd m_d = VectorXd(2);
					m_d << lm_it->x_f, lm_it->y_f;
					MatrixXd cov = MatrixXd(2,2);
					cov << pow(std_landmark[0],2), 0,
						0, pow(std_landmark[1],2);
					
					VectorXd tmp = (x_d - m_d);
					double c1 = tmp.transpose() * cov.inverse() * tmp;*/
					//Guassian PDF
					double x = obs.x;
					double y = obs.y;
					double m_x = lm_it->x_f;
					double m_y =  lm_it->y_f;
					double sig_x = std_landmark[0];
					double sig_y = std_landmark[1];
					double c3 = exp(-((x-m_x)*(x-m_x)/(2.0*sig_x*sig_x) + (y-m_y)*(y-m_y)/(2.0*sig_y*sig_y))) / (2.0*pi*sig_x*sig_y);
					//p.weight = p.weight * exp( -0.5 * c1) / sqrt( (2 * pi * cov).determinant());
					p.weight = p.weight * c3;
				}else{
					std::cout << "Map id not found!!!!" << std::endl;
				}

			}

		}
		particles[i] = p;
	}
}



void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	double sum = 0;
	int N = particles.size();
	std::vector<double> w;
	std::vector<Particle> resampled;
	double max_weight = 0;
	for( auto p : particles){
		w.push_back(p.weight);
		if(p.weight > max_weight){
			max_weight = p.weight;
		}
	}

	std::random_device rd;
    std::mt19937 gen_1(rd());
	std::uniform_int_distribution<int> dist_int(0,N);
    std::mt19937 gen_2(rd());
	std::uniform_real_distribution<double> dist_real(0, 2 * max_weight);

	double beta = 0;
	int index = dist_int(gen_1);
	for(auto p : particles){
		beta = beta + dist_real(gen_2);
		while(w[index] < beta){
			beta = beta - w[index];
			index = index + 1;
			index = index % N;
		}
		resampled.push_back(particles[index]);
	}
	
	
	
    /*std::discrete_distribution<int> d(w.begin(),w.end());

	for( auto p : particles){
		resampled.push_back(particles[d(gen)]);
	}*/
	for(int i = 0; i < num_particles; i++){
		particles[i] = resampled[i];	
	}
	
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
