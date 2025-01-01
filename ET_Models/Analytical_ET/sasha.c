#include <math.h>

double f(int n, double args[n]){
    double intexp, intpre;
    intpre = 1.0 / (((4.0 * args[4] * args[0]) + 1.0) * sqrt(args[0]));
    intexp = (-pow(args[3],2) / (4 * args[0])) - ((pow(args[2],2) + pow((args[1] - args[0]),2)) / ((4.0 * args[4] * args[0]) + 1.0));
    return intpre * exp(intexp);
}
