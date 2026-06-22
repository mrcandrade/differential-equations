#include <stdio.h>
#include <math.h>

double trapezio(double k, double a, double xi, double xf, int N){
    double h = (xf - xi)/N;
    double soma = 0.0;

    for(int i = 1; i < N; i++){
        double x = xi + i*h;
        soma += k * pow(x, a);
    }

    soma = h * ( (k*pow(xi,a))/2 + soma + (k*pow(xf,a))/2 );
    return soma;
}

double simpson(double k, double a, double xi, double xf, int N){
    if(N % 2 == 1) N++;

    double h = (xf - xi)/N;
    double soma1 = 0, soma2 = 0;

    for(int i = 1; i < N; i++){
        double x = xi + i*h;
        if(i % 2 == 0) soma2 += k*pow(x,a);
        else soma1 += k*pow(x,a);
    }

    return (h/3.0)*(k*pow(xi,a) + 4*soma1 + 2*soma2 + k*pow(xf,a));
}

int main(){
    double k = 0.6580;
    /*
     * questao2.c
     * Calcula o trabalho W = integral_{xi}^{xf} F(x) dx
     * onde F(x) = k * x^a (valores de k e a do enunciado).
     * Implementa o metodo do trapezio e o metodo de Simpson.
     */

    #include <stdio.h>
    #include <stdlib.h>
    #include <math.h>

    /* Metodo do trapezio: integra f(x) = k * x^a de xi a xf com N subintervalos */
    double trapezio(double k, double a, double xi, double xf, int N){
        double h = (xf - xi) / N;
        double soma = 0.0;

        for(int i = 1; i < N; ++i){
            double x = xi + i * h;
            soma += k * pow(x, a);
        }

        soma = h * ( (k * pow(xi, a)) / 2.0 + soma + (k * pow(xf, a)) / 2.0 );
        return soma;
    }

    /* Metodo de Simpson: exige N par; se N impar, incrementa N */
    double simpson(double k, double a, double xi, double xf, int N){
        if(N % 2 == 1) {
            N += 1; /* torna N par */
        }

        double h = (xf - xi) / N;
        double soma_odd = 0.0;  /* soma dos termos com coeficiente 4 */
        double soma_even = 0.0; /* soma dos termos com coeficiente 2 */

        for(int i = 1; i < N; ++i){
            double x = xi + i * h;
            if(i % 2 == 0) soma_even += k * pow(x, a);
            else soma_odd += k * pow(x, a);
        }

        return (h / 3.0) * ( k * pow(xi, a) + 4.0 * soma_odd + 2.0 * soma_even + k * pow(xf, a) );
    }

    int main(void){
        /* parametros do enunciado (substitua pelos seus valores se diferente) */
        const double k = 0.6580;
        const double a = 2.0;
        const double xi = 0.0;
        const double xf = 5.0;
        const int N = 1000; /* pedido no enunciado para o trapézio */

        double W_trap = trapezio(k, a, xi, xf, N);
        double W_simp = simpson(k, a, xi, xf, N);

        printf("Resultados para xi=%.2f, xf=%.2f, N=%d:\n", xi, xf, N);
        printf(" - Trapezio: %.10f\n", W_trap);
        printf(" - Simpson : %.10f\n", W_simp);

        /* grava opcional em arquivo dentro de output/ */
        system("mkdir output >nul 2>nul");
        FILE *fout = fopen("output/questao2_results.txt", "w");
        if(fout){
            fprintf(fout, "# xi xf N k a\n");
            fprintf(fout, "%.6f %.6f %d %.6f %.6f\n", xi, xf, N, k, a);
            fprintf(fout, "# Metodo Resultado\n");
            fprintf(fout, "Trapezio %.10f\n", W_trap);
            fprintf(fout, "Simpson  %.10f\n", W_simp);
            fclose(fout);
            printf("Resultados gravados em output/questao2_results.txt\n");
        } else {
            perror("Aviso: nao foi possivel abrir output/questao2_results.txt");
        }

        return 0;
    }
