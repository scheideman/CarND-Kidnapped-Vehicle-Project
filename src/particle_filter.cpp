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
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

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
	
	num_particles = 200;
	for(int i = 0; i < num_particles; i++){
		Particle p;
		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_x(gen);
		p.theta = dist_theta(gen);
		p.weight = 1.0;
		particles.push_back(p);
	}
	is_initialized = true;

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
	if(fabs(yaw_rate) < 0.00001){
		for (int i = 0; i < num_particles; i++){
			auto p = particles[i];
			p.x = p.x + c3 * cos(p.theta) + noise_x(gen);
			p.y = p.y + c3 * sin(p.theta) + noise_y(gen);
			p.theta = p.theta + noise_theta(gen); 
			particles[i] = p;
		} 
	// turning
	}else{
		for (int i = 0; i < num_particles; i++){
			auto p = particles[i];
			p.x = p.x + c * (sin(p.theta + c2) - sin(p.theta)) + noise_x(gen);
			p.y = p.y + c * (cos(p.theta) - cos(p.theta + c2)) + noise_y(gen);
			p.theta = p.theta + c2 + noise_theta(gen);
			particles[i] = p;
		} 
	}
	

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for(unsigned int i = 0; i < observations.size(); i++){
		int id_min = -1;
		double min_dist = 999999.0;
		LandmarkObs obv = observations[i];
		for(unsigned int j = 0; j < predicted.size(); j++){
			LandmarkObs prd = predicted[j];
			double d = dist(obv.x,obv.y,prd.x,prd.y);
			if(d < min_dist){
				min_dist = d;
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
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html	

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

				//rotation + translation
				t_obs.x = obs.x * cos(p.theta) - obs.y * sin(p.theta) + p.x;
				t_obs.y = obs.x * sin(p.theta) + obs.y * cos(p.theta) + p.y;
				t_obs.id = obs.id;

				transformed_obs.push_back(t_obs);
			}

			dataAssociation(predicted_lm, transformed_obs);
			particles[i].weight = 1.0;
			
			for(auto obs : transformed_obs){
				auto lm_it = find_if(begin(predicted_lm), end(predicted_lm), [=] (LandmarkObs const& f) { 
					return (f.id == obs.id); 
				});

				bool found = (lm_it != end(predicted_lm));
				if(found){
					double pi = 3.14159265359;
					//Guassian PDF
					double x = obs.x;
					double y = obs.y;
					double m_x = lm_it->x;
					double m_y =  lm_it->y;
					double sig_x = std_landmark[0];
					double sig_y = std_landmark[1];
					double guassian_pdf = ( 1.0/(2.0*pi*sig_x*sig_y)) * exp( -( pow(m_x-x,2)/(2*pow(sig_x, 2)) + (pow(m_y-y,2)/(2*pow(sig_y, 2))) ) );
					particles[i].weight *= guassian_pdf;
				}else{
					std::cout << "Map id not found!!!!" << std::endl;
				}

			}

		}
		
	}
}



void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
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
	std::uniform_int_distribution<int> dist_int(0,num_particles-1);
    std::mt19937 gen_2(rd());
	std::uniform_real_distribution<double> dist_real(0, 2.0 * max_weight);

	double beta = 0.0;
	int index = dist_int(gen_1);
	for(auto p : particles){
		beta = beta + dist_real(gen_2);
		while(w[index] < beta){
			beta = beta - w[index];
			index = (index + 1) % num_particles;
			
		}
		resampled.push_back(particles[index]);
	}
	particles = resampled;
    
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
