//things to include
#include "utils.h"
#include <math.h>
#include <tuple>
using namespace std;


//global vars
long n;
double h_2; //h^2
int iter; //will store number of iterations
double residual; // will store residual

void update(double* u){

}


//calculate the residual given a vector u
double calc_residual(double* u){
}

std::tuple<int, double> laplace(double* u, int res_factor,int max_iter) {	
}

int main(int argc, char** argv){
	N = read_option<long>("-n", argc, argv);
	double* u = (double*) malloc(N *N * sizeof(double)); // create vector of lenght n
}