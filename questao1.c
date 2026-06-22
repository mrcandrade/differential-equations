#include <stdio.h>
#include <math.h>

int main() {
    FILE *arq;
    double x, F;
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    int n = 0;

    arq = fopen("C:\\Users\\Maxter\\Desktop\\prova2fiscomp\\force.txt", "r");


    while(fscanf(arq, "%lf %lf", &x, &F) == 2){
        double lx = log(x);
        double ly = log(F);
        sx += lx;
        sy += ly;
        sxx += lx * lx;
        sxy += lx * ly;
        n++;
    }
    fclose(arq);

    double a = (n*sxy - sx*sy) / (n*sxx - sx*sx); 
    double b = (sy - a*sx)/n;                     

    printf("k = %lf\n", k);
    printf("alpha = %lf\n", a);

    return 0;
}
