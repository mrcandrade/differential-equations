#include <stdio.h>
#include <math.h>

int main(){
    double C = 1e-2;
    double R = 320;
    double q0 = 1;
    double tau = R*C;
    double dt = 1e-4;

    FILE *arq = fopen("corrente.txt", "w");

    for(double t = 0; t <= 10; t += 0.5){
        double q  = q0 * exp(-t/tau);
        double qd = q0 * exp(-(t+dt)/tau);
        double ql = q0 * exp(-(t-dt)/tau);

        double id  = (qd - q)/dt;
        double ic  = (qd - ql)/(2*dt);

        fprintf(arq, "%lf %lf %lf\n", t, id, ic);
    }

    fclose(arq);
    return 0;
}
