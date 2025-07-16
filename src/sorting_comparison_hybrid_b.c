#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mpi.h>
#include <omp.h>

void swap(int *a, int *b)
{
    int t = *a;
    *a = *b;
    *b = t;
}

// Sequential BubbleSort
void seqBubbleSort(int arr[], int n)
{
    for (int i = 0; i < n - 1; i++)
    {
        int swapped = 0;
        for (int j = 0; j < n - i - 1; j++)
        {
            if (arr[j] > arr[j + 1])
            {
                swap(&arr[j], &arr[j + 1]);
                swapped = 1;
            }
        }
        if (!swapped)
            break;
    }
}

// OpenMP parallel BubbleSort
void hybridBubbleSort(int *arr, int n)
{
    for (int i = 0; i < n - 1; i++)
    {
        int swapped = 0;
#pragma omp parallel for shared(arr) reduction(| : swapped)
        for (int j = 0; j < n - i - 1; j++)
        {
            if (arr[j] > arr[j + 1])
            {
#pragma omp critical
                swap(&arr[j], &arr[j + 1]);
                swapped = 1;
            }
        }
        if (!swapped)
            break;
    }
}

// MPI + OpenMP Hybrid Sort wrapper
void mpiHybridSort(int *data, int n, int rank, int size, void (*sortFunc)(int *, int))
{
    int *counts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int));
    int rem = n % size;
    int sum = 0;

    for (int i = 0; i < size; i++)
    {
        counts[i] = n / size + (i < rem ? 1 : 0);
        displs[i] = sum;
        sum += counts[i];
    }

    int local_n = counts[rank];
    int *local = malloc(local_n * sizeof(int));

    MPI_Scatterv(data, counts, displs, MPI_INT, local, local_n, MPI_INT, 0, MPI_COMM_WORLD);

    sortFunc(local, local_n);

    int *gathered = NULL;
    if (rank == 0)
        gathered = malloc(n * sizeof(int));

    MPI_Gatherv(local, local_n, MPI_INT, gathered, counts, displs, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0 && gathered != NULL)
    {
        seqBubbleSort(gathered, n); // Final full array sort
        memcpy(data, gathered, n * sizeof(int));
        free(gathered);
    }

    free(local);
    free(counts);
    free(displs);
}

// Accuracy checker
int calculate_accuracy(int *ref, int *test, int n)
{
    int correct = 0;
    for (int i = 0; i < n; i++)
    {
        if (ref[i] == test[i])
            correct++;
    }
    return (int)(((double)correct / n) * 100.0);
}

int main(int argc, char **argv)
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n;
    if (rank == 0)
    {
        printf("=== Sequential vs Hybrid(MPI+OpenMP) BubbleSort ===\n");
        printf("Enter array size: ");
        fflush(stdout);
        scanf("%d", &n);
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int *original = malloc(n * sizeof(int));
    int *seqBubble = malloc(n * sizeof(int));
    int *hybridBubble = malloc(n * sizeof(int));

    if (rank == 0)
    {
        srand(time(NULL));
        for (int i = 0; i < n; i++)
            original[i] = rand() % 100000;

        memcpy(seqBubble, original, n * sizeof(int));
        memcpy(hybridBubble, original, n * sizeof(int));
    }

    double seqBubbleTime = 0.0, hybridBubbleTime = 0.0;

    if (rank == 0)
    {
        double start = MPI_Wtime();
        seqBubbleSort(seqBubble, n);
        seqBubbleTime = MPI_Wtime() - start;
    }

    MPI_Bcast(hybridBubble, n, MPI_INT, 0, MPI_COMM_WORLD);
    double start = MPI_Wtime();
    mpiHybridSort(hybridBubble, n, rank, size, hybridBubbleSort);
    hybridBubbleTime = MPI_Wtime() - start;

    if (rank == 0)
    {
        printf("\n=== Results (Array size: %d) ===\n", n);
        printf("--------------------------------------------------\n");
        printf("Algorithm           | Sequential | Hybrid (MPI+OMP)\n");
        printf("--------------------------------------------------\n");
        printf("BubbleSort          | %9.6fs | %9.6fs\n", seqBubbleTime, hybridBubbleTime);
        printf("--------------------------------------------------\n");

        // Speed comparison print
        if (seqBubbleTime < hybridBubbleTime)
            printf("BubbleSort: Sequential is faster (%.6fs vs %.6fs)\n", seqBubbleTime, hybridBubbleTime);
        else if (hybridBubbleTime < seqBubbleTime)
            printf("BubbleSort: Hybrid (MPI+OMP) is faster (%.6fs vs %.6fs)\n", hybridBubbleTime, seqBubbleTime);
        else
            printf("BubbleSort: Both have same performance (%.6fs)\n", seqBubbleTime);

        printf("\nAccuracy of Hybrid BubbleSort: %d%%\n", calculate_accuracy(seqBubble, hybridBubble, n));

        printf("\nFirst 100 Sorted Elements (BubbleSort - Sequential): \n");
        for (int i = 0; i < 100 && i < n; i++)
            printf("%d ", seqBubble[i]);
        printf("\n");
    }

    free(original);
    free(seqBubble);
    free(hybridBubble);

    MPI_Finalize();
    return 0;
}
