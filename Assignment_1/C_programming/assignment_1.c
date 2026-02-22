#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <lapacke.h>

void generate_matrix(double* A, int m, int n)
{
    for(int j = 0; j < n; j++)
        for(int i = 0; i < m; i++)
            A[i + j*m] = (double)rand() / RAND_MAX;
}

void TSQR(double* A_local, int m_local, int n, double* R_final, MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    /* Local QR */
    double* tau = malloc(n * sizeof(double));

    /* Householder QR */
    LAPACKE_dgeqrf(LAPACK_COL_MAJOR, m_local, n, A_local, m_local, tau);

    /* Extract R */
    double* R_local = malloc(n*n*sizeof(double));
    for(int j = 0; j < n; j++)
        for(int i = 0; i < n; i++)
            R_local[i + j*n] =
                (i <= j) ? A_local[i + j*m_local] : 0.0;

    free(tau);

    /* Reduction tree */
    int step = 1;

    while(step < size)
    {
        if(rank % (2*step) == 0)
        {
            if(rank + step < size)
            {
                double* R_recv = malloc(n*n*sizeof(double));

                MPI_Recv(R_recv, n*n, MPI_DOUBLE, rank + step, 0, comm, MPI_STATUS_IGNORE);

                int m2 = 2*n;
                double* R_stack = malloc(m2*n*sizeof(double));

                for(int j = 0; j < n; j++)
                {
                    for(int i = 0; i < n; i++)
                        R_stack[i + j*m2] = R_local[i + j*n];

                    for(int i = 0; i < n; i++)
                        R_stack[i+n + j*m2] = R_recv[i + j*n];
                }

                /* QR of stacked matrix */
                double* tau2 = malloc(n*sizeof(double));

                LAPACKE_dgeqrf(LAPACK_COL_MAJOR, m2, n, R_stack, m2, tau2);

                for(int j = 0; j < n; j++)
                    for(int i = 0; i < n; i++)
                        R_local[i + j*n] =
                            (i <= j) ? R_stack[i + j*m2] : 0.0;

                free(R_recv);
                free(R_stack);
                free(tau2);
            }
        }
        else
        {
            MPI_Send(R_local, n*n, MPI_DOUBLE, rank - step, 0, comm);
            break;
        }

        step *= 2;
    }

    if(rank == 0)
        for(int i = 0; i < n*n; i++)
            R_final[i] = R_local[i];

    free(R_local);
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if(size != 4)
    {
        if(rank == 0)
            printf("Run with 4 processes.\n");
        MPI_Finalize();
        return 0;
    }

    int m = 16;
    int n = 4;

    int m_local = m / size;

    double* A_local = malloc(m_local*n*sizeof(double));
    double* A_full = NULL;

    if(rank == 0)
    {
        A_full = malloc(m*n*sizeof(double));
        generate_matrix(A_full, m, n);
    }

    MPI_Scatter(A_full, m_local*n, MPI_DOUBLE, A_local, m_local*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double* R = NULL;
    if(rank == 0)
        R = malloc(n*n*sizeof(double));

    TSQR(A_local, m_local, n, R, MPI_COMM_WORLD);

    /* check A^T A = R^T R */
    if(rank == 0)
    {
        double err = 0.0;

        for(int i = 0; i < n; i++)
            for(int j = 0; j < n; j++)
            {
                double AtA = 0.0;
                double RtR = 0.0;

                for(int k = 0; k < m; k++)
                    AtA += A_full[k + i*m] * A_full[k + j*m];

                for(int k = 0; k < n; k++)
                    RtR += R[k + i*n] * R[k + j*n];

                err += (AtA - RtR)*(AtA - RtR);
            }

        printf("||A^T A - R^T R||_F = %e\n", sqrt(err));

        free(A_full);
        free(R);
    }

    free(A_local);

    MPI_Finalize();
    return 0;
}
