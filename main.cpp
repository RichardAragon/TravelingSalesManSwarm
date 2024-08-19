#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <limits>
#include <random>

using namespace std;

const int NUM_CITIES = 20;
const int NUM_PARTICLES = 500;
const int MAX_ITERATIONS = 2000;
const float INITIAL_INERTIA_WEIGHT = 0.9;
const float FINAL_INERTIA_WEIGHT = 0.4;
const float COGNITIVE_COMPONENT = 1.49445;
const float SOCIAL_COMPONENT = 1.49445;
const float MUTATION_RATE = 0.1;
const float GAUSSIAN_STDDEV = 0.1;  // Standard deviation for Gaussian perturbations
const int PRUNE_PERCENTAGE = 10;    // Percentage of particles to prune

struct City {
    int x, y;
};

struct Particle {
    vector<int> position;
    vector<int> best_position;
    float best_cost;
    float cost;
};

vector<City> cities(NUM_CITIES);
vector<Particle> swarm(NUM_PARTICLES);
vector<int> global_best_position;
float global_best_cost = numeric_limits<float>::max();
mt19937 rng(time(0));  // Random number generator

// Function to generate random cities
void generateCities() {
    for (auto& city : cities) {
        city.x = rng() % 100;
        city.y = rng() % 100;
    }
}

// Function to calculate the cost of a given route
float calculateCost(const vector<int>& route) {
    float total_cost = 0;
    for (int i = 0; i < route.size() - 1; i++) {
        int cityA = route[i];
        int cityB = route[i + 1];
        float dx = cities[cityA].x - cities[cityB].x;
        float dy = cities[cityA].y - cities[cityB].y;
        total_cost += sqrt(dx * dx + dy * dy);
    }
    // Return to the starting city
    int startCity = route[0];
    int endCity = route.back();
    total_cost += sqrt(pow(cities[startCity].x - cities[endCity].x, 2) +
                       pow(cities[startCity].y - cities[endCity].y, 2));
    return total_cost;
}

// Function to initialize particles
void initializeParticles() {
    for (auto& particle : swarm) {
        particle.position.clear();
        for (int i = 0; i < NUM_CITIES; i++) {
            particle.position.push_back(i);
        }
        shuffle(particle.position.begin(), particle.position.end(), rng);
        particle.best_position = particle.position;
        particle.cost = calculateCost(particle.position);
        particle.best_cost = particle.cost;
        
        if (particle.cost < global_best_cost) {
            global_best_cost = particle.cost;
            global_best_position = particle.position;
        }
    }
}

// Function to apply mutation and Gaussian perturbation
void applyMutationAndGaussian(vector<int>& position) {
    // Mutation
    if (uniform_real_distribution<>(0, 1)(rng) < MUTATION_RATE) {
        int index1 = rng() % NUM_CITIES;
        int index2 = rng() % NUM_CITIES;
        swap(position[index1], position[index2]);
    }
    
    // Gaussian perturbation (swap based on Gaussian probability)
    normal_distribution<float> gauss_dist(0.0, GAUSSIAN_STDDEV);
    for (int i = 0; i < NUM_CITIES; i++) {
        if (uniform_real_distribution<>(0, 1)(rng) < fabs(gauss_dist(rng))) {
            int index2 = rng() % NUM_CITIES;
            swap(position[i], position[index2]);
        }
    }
}

// Function to prune the worst-performing particles
void pruneParticles() {
    // Sort swarm by cost in ascending order
    sort(swarm.begin(), swarm.end(), [](const Particle& a, const Particle& b) {
        return a.cost < b.cost;
    });

    // Remove the worst 10% of particles
    int prune_count = NUM_PARTICLES * PRUNE_PERCENTAGE / 100;
    for (int i = 0; i < prune_count; i++) {
        // Reinitialize worst particles to new random positions
        for (int j = 0; j < NUM_CITIES; j++) {
            swarm[NUM_PARTICLES - 1 - i].position[j] = j;
        }
        shuffle(swarm[NUM_PARTICLES - 1 - i].position.begin(), swarm[NUM_PARTICLES - 1 - i].position.end(), rng);
        swarm[NUM_PARTICLES - 1 - i].cost = calculateCost(swarm[NUM_PARTICLES - 1 - i].position);
        swarm[NUM_PARTICLES - 1 - i].best_position = swarm[NUM_PARTICLES - 1 - i].position;
        swarm[NUM_PARTICLES - 1 - i].best_cost = swarm[NUM_PARTICLES - 1 - i].cost;
    }
}

// Function to update the particles
void updateParticles(int iteration) {
    float inertia_weight = INITIAL_INERTIA_WEIGHT - ((INITIAL_INERTIA_WEIGHT - FINAL_INERTIA_WEIGHT) * iteration / MAX_ITERATIONS);
    
    for (auto& particle : swarm) {
        // Shuffle starting positions to encourage exploration
        shuffle(particle.position.begin(), particle.position.end(), rng);

        // Update based on personal best and global best
        for (int i = 0; i < NUM_CITIES; i++) {
            if (uniform_real_distribution<>(0, 1)(rng) < COGNITIVE_COMPONENT) {
                swap(particle.position[i], particle.position[particle.best_position[i]]);
            }
            if (uniform_real_distribution<>(0, 1)(rng) < SOCIAL_COMPONENT) {
                swap(particle.position[i], particle.position[global_best_position[i]]);
            }
        }
        
        // Apply mutation and Gaussian perturbation
        applyMutationAndGaussian(particle.position);
        
        particle.cost = calculateCost(particle.position);
        
        // Aggressively reward the particle if it finds a better solution
        if (particle.cost < particle.best_cost) {
            particle.best_cost = particle.cost;
            particle.best_position = particle.position;
        }
        
        // Update global best if needed
        if (particle.cost < global_best_cost) {
            global_best_cost = particle.cost;
            global_best_position = particle.position;
        }
    }
    
    // Prune the worst-performing particles
    pruneParticles();
}

// Function to print the best route found
void printBestRoute() {
    cout << "Best Route: ";
    for (int city : global_best_position) {
        cout << city << " -> ";
    }
    cout << global_best_position[0] << endl;
    cout << "Best Cost: " << global_best_cost << endl;
}

int main() {
    generateCities();
    initializeParticles();
    
    for (int iteration = 0; iteration < MAX_ITERATIONS; iteration++) {
        updateParticles(iteration);
    }
    
    printBestRoute();
    
    return 0;
}
