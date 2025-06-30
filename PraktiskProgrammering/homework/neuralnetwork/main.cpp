#include"../includes/matrix.h"
#include"../includes/minimization.h"
#include<iostream>
#include<fstream>
#include<functional>
#include<cmath>


using vec = pp::vector;
using mat = pp::matrix;




class ANN{
    public:
        vec weights;
        vec as;
        vec bs;

        ANN(int n) : weights(n), as(n), bs(n) {
            // Initialize weights, as, and bs with random values
            for (int i = 0; i < n; ++i) {
                weights[i] = 0.1 * ((double)rand() / RAND_MAX - 0.5); // small random weights
                as[i] = -1.0 + 2.0 * i / (n - 1.0); // spread across input domain
                bs[i] = 0.4; // fixed receptive width
            }
        };

        double forward(double input) {
            double output = 0.0; // Initialize output
            vec hidden(weights.size()); // Initialize hidden layer output vector
            for (int i = 0; i < weights.size(); ++i) {
                hidden[i] = (input - as[i])/bs[i]; // Apply parameter scaling
                hidden[i] = activation_function(hidden[i]); // Apply activation function
                output += hidden[i] * weights[i]; // Scale by weights
            }
            return output; // Return the output of the ANN
            };
        
        double activation_function(double x) {
            return x*std::exp(-x*x); // Gaussian activation function
        };

        void set_weights(const vec& new_weights) {
            if (new_weights.size() != weights.size()) {
                throw std::invalid_argument("New weights size must match the ANN size.");
            }
            weights = new_weights; // Set new weights
        };
        double training_forward(double input, vec weights, vec as, vec bs) {
            double output = 0.0; // Initialize output
            vec hidden(weights.size()); // Initialize hidden layer output vector
            for (int i = 0; i < weights.size(); ++i) {
                hidden[i] = (input - as[i])/bs[i]; // Apply parameter scaling
                hidden[i] = activation_function(hidden[i]); // Apply activation function
                output += hidden[i] * weights[i]; // Scale by weights
            }
            return output; // Return the output of the ANN
        };
        void train(vec& x, vec& y){

            // Use a minimization algorithm to optimize weights, as, and bs
            vec train_params(weights.size() + as.size() + bs.size());
            for (int i = 0; i < weights.size(); ++i) {
                train_params[i] = weights[i]; // Initialize weights
                train_params[i + weights.size()] = as[i]; // Initialize as
                train_params[i + 2 * weights.size()] = bs[i]; // Initialize bs
            }
            auto a_gradient = [&](vec& params){ // Compute the analytic gradient of the cost function
                vec grad(params.size());
                for (int i = 0; i < x.size(); ++i) {
                    vec t_weights(weights.size()), t_as(as.size()), t_bs(bs.size());
                    for (int j = 0; j < weights.size(); ++j) {
                        t_weights[j] = params[j];
                        t_as[j]      = params[j + weights.size()];
                        t_bs[j]      = params[j + 2 * weights.size()];
                    }

                    double xi = x[i], yi = y[i];
                    double output = training_forward(xi, t_weights, t_as, t_bs);
                    double error = output - yi;

                    for (int j = 0; j < weights.size(); ++j) {
                        double z = (xi - t_as[j]) / t_bs[j];
                        double act = z * std::exp(-z * z);
                        double dact_dz = std::exp(-z * z) * (1 - 2 * z * z);

                        grad[j] += 2 * error * act;
                        grad[j + weights.size()] += 2 * error * t_weights[j] * dact_dz * (-1.0 / t_bs[j]);
                        grad[j + 2 * weights.size()] += 2 * error * t_weights[j] * dact_dz * ((xi - t_as[j]) / (t_bs[j] * t_bs[j]));
                    }
                }
                return grad / x.size(); // Return the average gradient
            };

            
            //Now, minimize the cost function using a minimization algorithm
            double learning_rate = 0.001; // Learning rate for the optimization
            int max_iterations = 10000; // Maximum number of iterations for the optimization
            for (int iter = 0; iter < max_iterations; ++iter) {
                if (iter % 100 == 0) {
                    double cost = 0;
                    for (int i = 0; i < x.size(); ++i) {
                        vec t_weights(weights.size()), t_as(as.size()), t_bs(bs.size());
                        for (int j = 0; j < weights.size(); ++j) {
                            t_weights[j] = train_params[j];
                            t_as[j]      = train_params[j + weights.size()];
                            t_bs[j]      = train_params[j + 2 * weights.size()];
                        }
                        double pred = training_forward(x[i], t_weights, t_as, t_bs);
                        cost += std::pow(pred - y[i], 2);
                    }
                    std::cout << "Iter " << iter << ", Cost: " << cost / x.size() << "\n";
                }
                vec grad = a_gradient(train_params); // Compute the analytic gradient
                if (grad.norm() < 1e-6) { // Check for convergence
                    std::cout << "Converged after " << iter << " iterations.\n";
                    break;
                }
                for (int i = 0; i < train_params.size(); ++i) {
                    train_params[i] -= learning_rate * grad[i]; // Update parameters
                }
            }
            // Update weights, as, and bs with the optimized parameters
            for (int i = 0; i < weights.size(); ++i) {
                weights[i] = train_params[i]; // Update weights
                as[i] = train_params[i + weights.size()]; // Update as
                bs[i] = train_params[i + 2 * weights.size()]; // Update bs
         }
     };
     double dydx(double input){
        //This function computes the derivative of the ANN output with respect to the input
        double derivative = 0.0; // Initialize derivative
        for (int i = 0; i < weights.size(); ++i) {
            double z = (input - as[i]) / bs[i]; // Compute the normalized input
            double dact_dz = std::exp(-z * z) * (1 - 2 * z * z); // Derivative of the activation function
            derivative += weights[i] * dact_dz / bs[i]; // Scale by weights and normalize by bs
     }
        return derivative; // Return the derivative
    }

    double dy2dx2(double input){
        //This function computes the second derivative of the ANN output with respect to the input
        double second_derivative = 0.0; // Initialize second derivative
        for (int i = 0; i < weights.size(); ++i) {
            double z = (input - as[i]) / bs[i]; // Compute the normalized input
            double ddact_dz2 = 2*z* std::exp(-z * z) * (2 * z * z - 3); // Second derivative of the activation function
            second_derivative += weights[i] * (ddact_dz2 / (bs[i] * bs[i]));
        }
        return second_derivative; // Return the second derivative
     };

     double anti_derivative(double input){
        //This function computes the anti-derivative of the ANN output with respect to the input
        double integral = 0.0; // Initialize integral
        for (int i = 0; i < weights.size(); ++i) {
            double z = (input - as[i]) / bs[i]; // Compute the normalized input
            double act = activation_function(z); // Apply the activation function
            integral += 0.5 * weights[i]* bs[i] * act/z;
        }
    return integral; // Return the integral scaled by dx
     };
};


vec f_test(vec& x){
    vec y(x.size());
    vec first_term(x.size());
    vec second_term(x.size());
    for (int i = 0; i < x.size(); ++i) {
        first_term[i] = cos(5*x[i] - 1.0); // cos(5x - 1)
        second_term[i] = exp(-x[i]*x[i]); // exp(-x^2)
        y[i] = first_term[i] * second_term[i]; // f(x) =
    }
    return y;
}

int main(){

    ANN ann(32); // Create an ANN with 8 neurons
    std::cout << "Testing ANN forward pass with random input:\n";
    double input = static_cast<double>(rand()) / RAND_MAX; // Random input
    double output = ann.forward(input); // Forward pass
    std::cout << "Input: " << input << ", Output: " << output << "\n"; // Print input and output
    for (int i = 0; i < ann.weights.size(); ++i) {
        std::cout << "Weight[" << i << "]: " << ann.weights[i] << ", a: " << ann.as[i] << ", b: " << ann.bs[i] << "\n"; // Print weights and parameters
    };

    // Generate training data
    int n_points = 15; // Number of training points
    double x_min = -1.0, x_max = 1.0; // Range of input values
    vec x_train(n_points), y_train(n_points);
    std::ofstream train_file("data/training_data.txt");
    for (int i = 0; i < n_points; ++i) {
        x_train[i] = x_min + (x_max - x_min) * static_cast<double>(i) / (n_points - 1); // Generate evenly spaced input values
        y_train[i] = f_test(x_train)[i]; // Compute corresponding output values using the test function
        train_file << x_train[i] << " " << y_train[i] << "\n"; // Write to file
    }
    train_file.close(); // Close the training data file
    //Generate training data file for full function:
    n_points = 100;
    vec x_full(n_points), y_full(n_points);
    std::ofstream full_train_file("data/output_exact.txt");
    for (int i = 0; i < n_points; ++i) {
        x_full[i] = x_min + (x_max - x_min) * static_cast<double>(i) / (n_points - 1); // Generate evenly spaced input values
        y_full[i] = f_test(x_full)[i]; // Compute corresponding output values using the test function
        full_train_file << x_full[i] << " " << y_full[i] << "\n"; // Write to file
    }
    full_train_file.close(); // Close the full training data file




    //Generate output points:
    int n_output_points = 100; // Number of output points
    double output_min = -1.5, output_max = 1.5; // Range of output values
    vec x_output(n_output_points), y_output(n_output_points);

    std::ofstream output_file("data/ann_output_untrained.txt");
    for (int i = 0; i < n_output_points; ++i) {
        x_output[i] = output_min + (output_max - output_min) * static_cast<double>(i) / (n_output_points - 1); // Generate evenly spaced output values
        y_output[i] = ann.forward(x_output[i]); // Compute corresponding output values using the untrained ANN
        output_file << x_output[i] << " " << y_output[i] << "\n"; // Write to file
    }
    output_file.close();

    std::cout << "Training ANN with generated data...\n";
    ann.train(x_train, y_train); // Train the ANN with the generated data
    std::cout << "Training complete.\n";

    vec y_deriv(n_output_points);
    vec y_2nd_deriv(n_output_points);
    vec y_integral(n_output_points);
    // Now, generate output points again after training
    output_file.open("data/ann_output_trained.txt");
    for (int i = 0; i < n_output_points; ++i) {
        y_output[i] = ann.forward(x_output[i]); // Compute corresponding output values using the trained ANN
        y_deriv[i] = ann.dydx(x_output[i]); // Compute the first derivative
        y_2nd_deriv[i] = ann.dy2dx2(x_output[i]);
        y_integral[i] = ann.anti_derivative(x_output[i]); // Compute the anti-derivative
        output_file << x_output[i] << " " << y_output[i] << "\t" << y_deriv[i] << "\t" << y_2nd_deriv[i] << "\t" << y_integral[i] << "\n"; // Write to file
    }
    output_file.close();



    return 0;
}