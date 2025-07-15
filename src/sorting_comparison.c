#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <string.h>

void swap(int *a, int *b)
{
    int t = *a;
    *a = *b;
    *b = t;
}

void seqQuickSort(int arr[], int low, int high)
{
    if (low < high)
    {
        int pivot = arr[high];
        int i = low - 1;
        for (int j = low; j <= high - 1; j++)
        {
            if (arr[j] < pivot)
            {
                i++;
                swap(&arr[i], &arr[j]);
            }
        }
        swap(&arr[i + 1], &arr[high]);
        int pi = i + 1;
        seqQuickSort(arr, low, pi - 1);
        seqQuickSort(arr, pi + 1, high);
    }
}

void parQuickSort(int arr[], int low, int high)
{
    const int THRESHOLD = 500;
    if (low < high)
    {
        if (high - low < THRESHOLD)
        {
            seqQuickSort(arr, low, high);
            return;
        }

        int pivot = arr[high];
        int i = low - 1;
        for (int j = low; j <= high - 1; j++)
        {
            if (arr[j] < pivot)
            {
                i++;
                swap(&arr[i], &arr[j]);
            }
        }
        swap(&arr[i + 1], &arr[high]);
        int pi = i + 1;

#pragma omp task shared(arr)
        parQuickSort(arr, low, pi - 1);
#pragma omp task shared(arr)
        parQuickSort(arr, pi + 1, high);
#pragma omp taskwait
    }
}

void mpiQuickSort(int *data, int n, int rank, int size)
{
    int *counts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int));
    int rem = n % size;
    int sum = 0;

    for (int i = 0; i < size; i++)
    {
        counts[i] = n / size;
        if (rem > 0)
        {
            counts[i]++;
            rem--;
        }
        displs[i] = sum;
        sum += counts[i];
    }

    int local_n = counts[rank];
    int *local = malloc(local_n * sizeof(int));

    MPI_Scatterv(data, counts, displs, MPI_INT, local, local_n, MPI_INT, 0, MPI_COMM_WORLD);

    seqQuickSort(local, 0, local_n - 1);

    int *gathered = NULL;
    if (rank == 0)
        gathered = malloc(n * sizeof(int));

    MPI_Gatherv(local, local_n, MPI_INT, gathered, counts, displs, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        seqQuickSort(gathered, 0, n - 1);
        memcpy(data, gathered, n * sizeof(int));
        free(gathered);
    }

    free(local);
    free(counts);
    free(displs);
}

void hybridQuickSort(int *data, int n, int rank, int size)
{
    int *counts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int));
    int rem = n % size;
    int sum = 0;

    for (int i = 0; i < size; i++)
    {
        counts[i] = n / size;
        if (rem > 0)
        {
            counts[i]++;
            rem--;
        }
        displs[i] = sum;
        sum += counts[i];
    }

    int local_n = counts[rank];
    int *local = malloc(local_n * sizeof(int));

    MPI_Scatterv(data, counts, displs, MPI_INT, local, local_n, MPI_INT, 0, MPI_COMM_WORLD);

#pragma omp parallel
    {
#pragma omp single
        parQuickSort(local, 0, local_n - 1);
    }

    int *gathered = NULL;
    if (rank == 0)
        gathered = malloc(n * sizeof(int));

    MPI_Gatherv(local, local_n, MPI_INT, gathered, counts, displs, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        seqQuickSort(gathered, 0, n - 1);
        memcpy(data, gathered, n * sizeof(int));
        free(gathered);
    }

    free(local);
    free(counts);
    free(displs);
}

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

void parBubbleSort(int arr[], int n)
{
    for (int phase = 0; phase < n; phase++)
    {
#pragma omp parallel for
        for (int i = phase % 2; i < n - 1; i += 2)
        {
            if (arr[i] > arr[i + 1])
            {
                swap(&arr[i], &arr[i + 1]);
            }
        }
    }
}

void mpiBubbleSort(int *data, int n, int rank, int size)
{
    int *counts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int));
    int rem = n % size;
    int sum = 0;

    for (int i = 0; i < size; i++)
    {
        counts[i] = n / size;
        if (rem > 0)
        {
            counts[i]++;
            rem--;
        }
        displs[i] = sum;
        sum += counts[i];
    }

    int local_n = counts[rank];
    int *local = malloc(local_n * sizeof(int));

    MPI_Scatterv(data, counts, displs, MPI_INT, local, local_n, MPI_INT, 0, MPI_COMM_WORLD);

    seqBubbleSort(local, local_n);

    int *gathered = NULL;
    if (rank == 0)
        gathered = malloc(n * sizeof(int));

    MPI_Gatherv(local, local_n, MPI_INT, gathered, counts, displs, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        seqBubbleSort(gathered, n);
        memcpy(data, gathered, n * sizeof(int));
        free(gathered);
    }

    free(local);
    free(counts);
    free(displs);
}

void hybridBubbleSort(int *data, int n, int rank, int size)
{
    int *counts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int));
    int rem = n % size;
    int sum = 0;

    for (int i = 0; i < size; i++)
    {
        counts[i] = n / size;
        if (rem > 0)
        {
            counts[i]++;
            rem--;
        }
        displs[i] = sum;
        sum += counts[i];
    }

    int local_n = counts[rank];
    int *local = malloc(local_n * sizeof(int));

    MPI_Scatterv(data, counts, displs, MPI_INT, local, local_n, MPI_INT, 0, MPI_COMM_WORLD);

    parBubbleSort(local, local_n);

    int *gathered = NULL;
    if (rank == 0)
        gathered = malloc(n * sizeof(int));

    MPI_Gatherv(local, local_n, MPI_INT, gathered, counts, displs, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        parBubbleSort(gathered, n);
        memcpy(data, gathered, n * sizeof(int));
        free(gathered);
    }

    free(local);
    free(counts);
    free(displs);
}

// Calculate Accuracy
double calculate_accuracy(int *ref, int *test, int n)
{
    int correct = 0;
    for (int i = 0; i < n; i++)
    {
        if (ref[i] == test[i])
        {
            correct++;
        }
    }
    return (double)correct * 100.0 / n;
}

// Search for fasted method
const char *find_fastest_method(double seq, double openmp, double mpi, double hybrid)
{
    double min = seq;
    const char *method = "Sequential";

    if (openmp < min)
    {
        min = openmp;
        method = "OpenMP";
    }
    if (mpi < min)
    {
        min = mpi;
        method = "MPI";
    }
    if (hybrid < min)
    {
        min = hybrid;
        method = "Hybrid";
    }
    return method;
}

int main()
{
    int rank, size;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n;
    if (rank == 0)
    {
        printf("=== Sorting Algorithm Comparison ===\n");
        printf("Enter array size: ");
        fflush(stdout);
    }

    if (scanf("%d", &n) != 1 || n <= 0)
    {
        if (rank == 0)
            printf("Invalid input. Exiting.\n");
        MPI_Finalize();
        return 1;
    }

    if (n <= 0)
    {
        if (rank == 0)
            printf("Invalid array size!\n");
        MPI_Finalize();
        return 1;
    }

    int *original = malloc(n * sizeof(int));
    if (rank == 0)
    {
        srand(time(NULL));
        for (int i = 0; i < n; i++)
            original[i] = rand() % 100000;

        printf("\n=== Original Array (First 100 Elements): ===\n");
        for (int i = 0; i < 100 && i < n; i++)
            printf("%d ", original[i]);
        printf("\n");
    }

    int *seqQuick = malloc(n * sizeof(int));
    int *parQuick = malloc(n * sizeof(int));
    int *mpiQuick = malloc(n * sizeof(int));
    int *hybridQuick = malloc(n * sizeof(int));
    int *seqBubble = malloc(n * sizeof(int));
    int *parBubble = malloc(n * sizeof(int));
    int *mpiBubble = malloc(n * sizeof(int));
    int *hybridBubble = malloc(n * sizeof(int));

    if (rank == 0)
    {
        memcpy(seqQuick, original, n * sizeof(int));
        memcpy(parQuick, original, n * sizeof(int));
        memcpy(mpiQuick, original, n * sizeof(int));
        memcpy(hybridQuick, original, n * sizeof(int));
        memcpy(seqBubble, original, n * sizeof(int));
        memcpy(parBubble, original, n * sizeof(int));
        memcpy(mpiBubble, original, n * sizeof(int));
        memcpy(hybridBubble, original, n * sizeof(int));
    }

    double seqQ_time, parQ_time, seqB_time = 0, parB_time, mpiQ_time, mpiB_time, hybridQ_time, hybridB_time;

    if (rank == 0)
    {
        double start = omp_get_wtime();
        seqQuickSort(seqQuick, 0, n - 1);
        seqQ_time = omp_get_wtime() - start;

        start = omp_get_wtime();
#pragma omp parallel
        {
#pragma omp single
            parQuickSort(parQuick, 0, n - 1);
        }

        parQ_time = omp_get_wtime() - start;

        if (n <= 50000)
        {
            start = omp_get_wtime();
            seqBubbleSort(seqBubble, n);
            seqB_time = omp_get_wtime() - start;
        }

        start = omp_get_wtime();
        parBubbleSort(parBubble, n);
        parB_time = omp_get_wtime() - start;
    }

    MPI_Bcast(original, n, MPI_INT, 0, MPI_COMM_WORLD);

    double start = MPI_Wtime();
    mpiQuickSort(mpiQuick, n, rank, size);
    mpiQ_time = MPI_Wtime() - start;

    start = MPI_Wtime();
    mpiBubbleSort(mpiBubble, n, rank, size);
    mpiB_time = MPI_Wtime() - start;

    start = MPI_Wtime();
    hybridQuickSort(hybridQuick, n, rank, size);
    hybridQ_time = MPI_Wtime() - start;

    start = MPI_Wtime();
    hybridBubbleSort(hybridBubble, n, rank, size);
    hybridB_time = MPI_Wtime() - start;

    if (rank == 0)
    {

        // Accuracy comparisons
        double acc_parQuick = calculate_accuracy(seqQuick, parQuick, n);
        double acc_mpiQuick = calculate_accuracy(seqQuick, mpiQuick, n);
        double acc_hybridQuick = calculate_accuracy(seqQuick, hybridQuick, n);
        double acc_parBubble = 0.0, acc_mpiBubble = 0.0, acc_hybridBubble = 0.0;

        if (n <= 50000)
        {
            acc_parBubble = calculate_accuracy(seqBubble, parBubble, n);
            acc_mpiBubble = calculate_accuracy(seqBubble, mpiBubble, n);
            acc_hybridBubble = calculate_accuracy(seqBubble, hybridBubble, n);
        }

        // Find Fastest Method
        const char *fastestQuick = find_fastest_method(seqQ_time, parQ_time, mpiQ_time, hybridQ_time);
        const char *fastestBubble = NULL;

        if (n <= 50000)
        {
            fastestBubble = find_fastest_method(seqB_time, parB_time, mpiB_time, hybridB_time);
        }
        else
        {
            fastestBubble = find_fastest_method(9999.0, parB_time, mpiB_time, hybridB_time); // Dummy high value for sequential
        }

        /////////////////////////////////////////////////////////////////////
        printf("\n=== Sorting Results (Array Size: %d) ===\n", n);
        printf("--------------------------------------------------------------------------------\n");
        printf("| Algorithm   | Sequential | OpenMP    | MPI       | Hybrid    | Fastest       |\n");
        printf("--------------------------------------------------------------------------------\n");
        printf("| QuickSort   | %9.6fs | %9.6fs | %9.6fs | %9.6fs | %-13s |\n",
               seqQ_time, parQ_time, mpiQ_time, hybridQ_time, fastestQuick);

        if (n <= 50000)
            printf("| BubbleSort  | %9.6fs | %9.6fs | %9.6fs | %9.6fs | %-13s |\n",
                   seqB_time, parB_time, mpiB_time, hybridB_time, fastestBubble);
        else
            printf("| BubbleSort  |     ---   | %9.6fs | %9.6fs | %9.6fs | %-13s |\n",
                   parB_time, mpiB_time, hybridB_time, fastestBubble);

        printf("--------------------------------------------------------------------------------\n");

        // Shorted Array
        printf("\nSample of Sorted Array (Sequential QuickSort - First 100):\n");
        for (int i = 0; i < 100 && i < n; i++)
            printf("%d ", seqQuick[i]);

        printf("--------------------------------------------------------------------------\n");

        // Accuracy Table
        printf("\nAccuracy Details(Compared to Sequential):\n");
        printf("QuickSort - OpenMP: %.2f%% | MPI: %.2f%% | Hybrid: %.2f%%\n",
               acc_parQuick, acc_mpiQuick, acc_hybridQuick);

        if (n <= 50000)
            printf("BubbleSort - OpenMP: %.2f%% | MPI: %.2f%% | Hybrid: %.2f%%\n",
                   acc_parBubble, acc_mpiBubble, acc_hybridBubble);
        printf("\n");
    }

    free(original);
    free(seqQuick);
    free(parQuick);
    free(seqBubble);
    free(parBubble);
    free(mpiQuick);
    free(mpiBubble);
    free(hybridQuick);
    free(hybridBubble);

    MPI_Finalize();
    return 0;
}
