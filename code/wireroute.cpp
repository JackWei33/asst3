/**
 * Parallel VLSI Wire Routing via OpenMP
 * Name 1(andrew_id 1), Name 2(andrew_id 2)
 */

#include "wireroute.h"

#include <algorithm>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <string>
#include <vector>
#include <random>
#include <cstdlib>
#include <climits>

#include <unistd.h>
#include <omp.h>

void print_stats(const std::vector<std::vector<int>>& occupancy) {
    int max_occupancy = 0;
    long long total_cost = 0;

    for (const auto& row : occupancy) {
        for (const int count : row) {
            max_occupancy = std::max(max_occupancy, count);
            total_cost += count * count;
        }
    }

    std::cout << "Max occupancy: " << max_occupancy << '\n';
    std::cout << "Total cost: " << total_cost << '\n';
}

void write_output(const std::vector<Wire>& wires, const int num_wires, const std::vector<std::vector<int>>& occupancy, const int dim_x, const int dim_y, const int num_threads, std::string input_filename) {
    if (std::size(input_filename) >= 4 && input_filename.substr(std::size(input_filename) - 4) == ".txt") {
        input_filename.resize(std::size(input_filename) - 4);
    }

    const std::string occupancy_filename = input_filename + "_occupancy_" + std::to_string(num_threads) + ".txt";
    const std::string wires_filename = input_filename + "_wires_" + std::to_string(num_threads) + ".txt";

    std::ofstream out_occupancy(occupancy_filename, std::fstream::out);
    if (!out_occupancy) {
        std::cerr << "Unable to open file: " << occupancy_filename << '\n';
        exit(EXIT_FAILURE);
    }

    out_occupancy << dim_x << ' ' << dim_y << '\n';
    for (const auto& row : occupancy) {
        for (const int count : row) {
            out_occupancy << count << ' ';
        }
        out_occupancy << '\n';
    }

    out_occupancy.close();

    std::ofstream out_wires(wires_filename, std::fstream:: out);
    if (!out_wires) {
        std::cerr << "Unable to open file: " << wires_filename << '\n';
        exit(EXIT_FAILURE);
    }

    out_wires << dim_x << ' ' << dim_y << '\n' << num_wires << '\n';

    for (const auto& [start_x, start_y, end_x, end_y, bend1_x, bend1_y] : wires) {
        out_wires << start_x << ' ' << start_y << ' ' << bend1_x << ' ' << bend1_y << ' ';

        if (start_y == bend1_y) {
        // first bend was horizontal

            if (end_x != bend1_x) {
                // two bends

                out_wires << bend1_x << ' ' << end_y << ' ';
            }
        } else if (start_x == bend1_x) {
            // first bend was vertical

            if(end_y != bend1_y) {
                // two bends

                out_wires << end_x << ' ' << bend1_y << ' ';
            }
        }
        out_wires << end_x << ' ' << end_y << '\n';
    }

    out_wires.close();
}


void set_random_path(Wire &wire) {
    if ((wire.start_x == wire.end_x) || (wire.start_y == wire.end_y)) {
        wire.bend1_x = wire.start_x;
        wire.bend1_y = wire.start_y;
        return;
    }

    int delta_x = wire.end_x - wire.start_x;
    int delta_y = wire.end_y - wire.start_y;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(1, std::abs(delta_x) + std::abs(delta_y));
    int random_number = distrib(gen);

    if (random_number <= std::abs(delta_x)) {
        if (wire.end_x > wire.start_x) {
            wire.bend1_x = wire.start_x + random_number;
        }
        else {
            wire.bend1_x = wire.start_x - random_number;
        }
        wire.bend1_y = wire.start_y;
    }
    else {
        random_number -= std::abs(delta_x);
        if (wire.end_y > wire.start_y) {
            wire.bend1_y = wire.start_y + random_number;
        }
        else {
            wire.bend1_y = wire.start_y - random_number;
        }
        wire.bend1_x = wire.start_x;
    }
}

// Fills in line in occupancy. Inclusive of start and exclusive of end.
void fill_in_occupancy(std::vector<std::vector<int>>& occupancy, int start_x, int start_y, int end_x, int end_y, int add_or_sub) {
    if ((start_x != end_x) && (start_y != end_y)) {
        printf("Error at fill_in_occupancy\n");
        return;
    }

    if (start_x != end_x) {
        int curr_x = start_x;
        while (curr_x != end_x) {
            occupancy[start_y][curr_x] += add_or_sub;
            if (end_x > curr_x) {
                curr_x += 1;
            }
            else {
                curr_x -= 1;
            }
        }
    }
    else {
        int curr_y = start_y;
        while (curr_y != end_y) {
            occupancy[curr_y][start_x] += add_or_sub;
            if (end_y > curr_y) {
                curr_y += 1;
            }
            else {
                curr_y -= 1;
            }
        }
    }
}

// Returns the cost of a straight line (inclusive of start and exclusive of end)
int cost_of_route(std::vector<std::vector<int>>& occupancy, int start_x, int start_y, int end_x, int end_y) {
    if ((start_x != end_x) && (start_y != end_y)) {
        printf("Error at cost_of_route\n");
        return 0;
    }
    int cost = 0;

    if (start_x != end_x) {
        int curr_x = start_x;
        while (curr_x != end_x) {
            cost += std::pow(occupancy[start_y][curr_x] + 1, 2);
            if (end_x > curr_x) {
                curr_x += 1;
            }
            else {
                curr_x -= 1;
            }
        }
    }
    else {
        int curr_y = start_y;
        while (curr_y != end_y) {
            cost += std::pow(occupancy[curr_y][start_x] + 1, 2);
            if (end_y > curr_y) {
                curr_y += 1;
            }
            else {
                curr_y -= 1;
            }
        }
    }
    return cost;
}

// Returns a vector mapping all coordinates a wire goes to from start coords to end coords
std::vector<std::pair<int, int>> find_bends(int start_x, int start_y, int end_x, int end_y, int bend1_x, int bend1_y) {
    std::vector<std::pair<int, int>> vec;
    vec.push_back(std::make_pair(start_x, start_y));

    if ((start_x == end_x) || (start_y == end_y)) {
        vec.push_back(std::make_pair(end_x, end_y));
        return vec;
    }

    vec.push_back(std::make_pair(bend1_x, bend1_y));

    if (start_y == bend1_y && end_x != bend1_x) {
        vec.push_back(std::make_pair(bend1_x, end_y));
    }
    else if (start_x == bend1_x && end_y != bend1_y) {
        vec.push_back(std::make_pair(end_x, bend1_y));
    }

    vec.push_back(std::make_pair(end_x, end_y));
    return vec;
}


void solve_within_wire(std::vector<Wire>& wires, std::vector<std::vector<int>>& occupancy, int num_threads, int SA_iters, double SA_prob) {
    // Initalize random wiring paths
    for (Wire& wire : wires) {
        set_random_path(wire);
        std::vector<std::pair<int, int>> bends = find_bends(wire.start_x, wire.start_y, wire.end_x, wire.end_y, wire.bend1_x, wire.bend1_y);
        for (long unsigned int i = 0; i < bends.size() - 1; i++) {
            fill_in_occupancy(occupancy, bends[i].first, bends[i].second, bends[i+1].first, bends[i+1].second, 1);
        }
        occupancy[bends[bends.size() - 1].second][bends[bends.size() - 1].first] += 1;
    }

    for (int i = 0; i < SA_iters; i++) {
        for (Wire& wire : wires) {
            // Remove current path
            std::vector<std::pair<int, int>> bends = find_bends(wire.start_x, wire.start_y, wire.end_x, wire.end_y, wire.bend1_x, wire.bend1_y);
            for (long unsigned int i = 0; i < bends.size() - 1; i++) {
                fill_in_occupancy(occupancy, bends[i].first, bends[i].second, bends[i+1].first, bends[i+1].second, -1);
            }
            occupancy[bends[bends.size() - 1].second][bends[bends.size() - 1].first] += -1;

            // Find best new path
            int delta_x = wire.end_x - wire.start_x;
            int delta_y = wire.end_y - wire.start_y;
            
            int min_new_bend1_x = 0;
            int min_new_bend1_y = 0;
            int min_cost = INT_MAX;

            for (int j = 1; j < std::abs(delta_x) + std::abs(delta_y) + 1; j++) {
                int random_number = j;
                int new_bend1_x, new_bend1_y;
                if (random_number <= std::abs(delta_x)) {
                    if (wire.end_x > wire.start_x) {
                        new_bend1_x = wire.start_x + random_number;
                    }
                    else {
                        new_bend1_x = wire.start_x - random_number;
                    }
                    new_bend1_y = wire.start_y;
                }
                else {
                    random_number -= std::abs(delta_x);
                    if (wire.end_y > wire.start_y) {
                        new_bend1_y = wire.start_y + random_number;
                    }
                    else {
                        new_bend1_y = wire.start_y - random_number;
                    }
                    new_bend1_x = wire.start_x;
                }
                std::vector<std::pair<int, int>> bends = find_bends(wire.start_x, wire.start_y, wire.end_x, wire.end_y, new_bend1_x, new_bend1_y);
                int cost = 0;
                for (long unsigned int i = 0; i < bends.size() - 1; i++) {
                    cost += cost_of_route(occupancy, bends[i].first, bends[i].second, bends[i+1].first, bends[i+1].second);
                }
                cost += std::pow(occupancy[bends[bends.size() - 1].second][bends[bends.size() - 1].first] + 1, 2);

                if (cost < min_cost) {
                    min_cost = cost;
                    min_new_bend1_x = new_bend1_x;
                    min_new_bend1_y = new_bend1_y;
                }
            }
            
            // Add new path to occupancy
            wire.bend1_x = min_new_bend1_x;
            wire.bend1_y = min_new_bend1_y;

            bends = find_bends(wire.start_x, wire.start_y, wire.end_x, wire.end_y, wire.bend1_x, wire.bend1_y);
            for (long unsigned int i = 0; i < bends.size() - 1; i++) {
                fill_in_occupancy(occupancy, bends[i].first, bends[i].second, bends[i+1].first, bends[i+1].second, 1);
            }
            occupancy[bends[bends.size() - 1].second][bends[bends.size() - 1].first] += 1;
        }
    }
}

int main(int argc, char *argv[]) {
    const auto init_start = std::chrono::steady_clock::now();

    std::string input_filename;
    int num_threads = 0;
    double SA_prob = 0.1;
    int SA_iters = 5;
    char parallel_mode = '\0';
    int batch_size = 1;

    int opt;
    while ((opt = getopt(argc, argv, "f:n:p:i:m:b:")) != -1) {
        switch (opt) {
            case 'f':
                input_filename = optarg;
                break;
            case 'n':
                num_threads = atoi(optarg);
                break;
            case 'p':
                SA_prob = atof(optarg);
                break;
            case 'i':
                SA_iters = atoi(optarg);
                break;
            case 'm':
                parallel_mode = *optarg;
                break;
            case 'b':
                batch_size = atoi(optarg);
                break;
            default:
                std::cerr << "Usage: " << argv[0] << " -f input_filename -n num_threads [-p SA_prob] [-i SA_iters] -m parallel_mode -b batch_size\n";
                exit(EXIT_FAILURE);
        }
    }

    // Check if required options are provided
    if (empty(input_filename) || num_threads <= 0 || SA_iters <= 0 || (parallel_mode != 'A' && parallel_mode != 'W') || batch_size <= 0) {
        std::cerr << "Usage: " << argv[0] << " -f input_filename -n num_threads [-p SA_prob] [-i SA_iters] -m parallel_mode -b batch_size\n";
        exit(EXIT_FAILURE);
    }

    std::cout << "Number of threads: " << num_threads << '\n';
    std::cout << "Simulated annealing probability parameter: " << SA_prob << '\n';
    std::cout << "Simulated annealing iterations: " << SA_iters << '\n';
    std::cout << "Input file: " << input_filename << '\n';
    std::cout << "Parallel mode: " << parallel_mode << '\n';
    std::cout << "Batch size: " << batch_size << '\n';

    std::ifstream fin(input_filename);

    if (!fin) {
        std::cerr << "Unable to open file: " << input_filename << ".\n";
        exit(EXIT_FAILURE);
    }

    int dim_x, dim_y;
    int num_wires;

    /* Read the grid dimension and wire information from file */
    fin >> dim_x >> dim_y >> num_wires;

    std::vector<Wire> wires(num_wires);
    std::vector occupancy(dim_y, std::vector<int>(dim_x));

    for (auto& wire : wires) {
        fin >> wire.start_x >> wire.start_y >> wire.end_x >> wire.end_y;
        wire.bend1_x = wire.start_x;
        wire.bend1_y = wire.start_y;
    }

    /* Initialize any additional data structures needed in the algorithm */

    const double init_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - init_start).count();
    std::cout << "Initialization time (sec): " << std::fixed << std::setprecision(10) << init_time << '\n';

    const auto compute_start = std::chrono::steady_clock::now();

    /** 
     * Implement the wire routing algorithm here
     * Feel free to structure the algorithm into different functions
     * Don't use global variables.
     * Use OpenMP to parallelize the algorithm. 
     */
    solve_within_wire(wires, occupancy, num_threads, SA_iters, SA_prob);

    const double compute_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - compute_start).count();
    std::cout << "Computation time (sec): " << compute_time << '\n';

    /* Write wires and occupancy matrix to files */

    print_stats(occupancy);
    write_output(wires, num_wires, occupancy, dim_x, dim_y, num_threads, input_filename);
}

validate_wire_t Wire::to_validate_format(void) const {
    throw std::logic_error("to_validate_format not implemented.");
    // validate_wire_t wire;
    // std::vector<std::pair<int, int>> bends = find_bends(start_x, start_y, end_x, end_y, bend1_x, bend1_y);

    // wire.num_pts = bends.size();
    // for (long unsigned int i = 0; i < bends.size(); i++) {
    //     wire.p[i].x = 0;
    //     wire.p[i].y = 0;
    // }
    // return wire;
}
