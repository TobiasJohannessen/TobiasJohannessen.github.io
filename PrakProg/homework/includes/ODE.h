#include <functional>
#include"../includes/matrix.h"
#include<vector>

using vec = pp::vector;

std::tuple<vec, vec> rkstep12(
	std::function<vec(double, const vec&, const vec&)>,
	double,
	const vec&,
	const vec&
);

std::tuple<std::vector<double>, std::vector<vec>> driver(
    std::function<vec(double, const vec&)>,
	double,
    double,
	const vec&,
	double dx,
    double acc,
    double eps
);


std::tuple<vec, vec, vec> slope_field(
    std::function<vec(double, const vec&)>,
    double,
    double,
    double,
    double dx
);
