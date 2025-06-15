#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

void swap(int* a, int* b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

// Parallel QuickSort using OpenMP tasks
void quickSort(int arr[], int low, int high) {
    if (low < high) {
        int pivot = arr[high];
        int i = low - 1;
        
        for (int j = low; j <= high - 1; j++) {
            if (arr[j] < pivot) {
                i++;
                swap(&arr[i], &arr[j]);
            }
        }
        swap(&arr[i + 1], &arr[high]);
        int pi = i + 1;

        #pragma omp task firstprivate(arr, low, pi)
        {
            quickSort(arr, low, pi - 1);
        }

        #pragma omp task firstprivate(arr, high, pi)
        {
            quickSort(arr, pi + 1, high);
        }

        #pragma omp taskwait
    }
}

// Parallel BubbleSort using OpenMP
void bubbleSort(int arr[], int n) {
    int swapped = 1;
    while (swapped) {
        swapped = 0;
        #pragma omp parallel for shared(swapped)
        for (int i = 0; i < n - 1; i++) {
            if (arr[i] > arr[i + 1]) {
                swap(&arr[i], &arr[i + 1]);
                #pragma omp critical
                swapped = 1;
            }
        }
    }
}

int main() {
    int n;
    printf("=== HPC Parallel Sorting (OpenMP) ===\n");
    printf("Enter array size: ");
    
    if (scanf("%d", &n) != 1 || n <= 0) {
        printf("Invalid array size!\n");
        return 1;
    }

    // Allocate memory
    int *original = malloc(n * sizeof(int));
    int *arr_quick = malloc(n * sizeof(int));
    int *arr_bubble = malloc(n * sizeof(int));
    if (!original || !arr_quick || !arr_bubble) {
        printf("Memory allocation failed!\n");
        return 1;
    }

    // Generate random array
    srand(time(NULL));
    for (int i = 0; i < n; i++) {
        original[i] = rand() % 1000;
        arr_quick[i] = original[i];
        arr_bubble[i] = original[i];
    }

    // Display sample elements
    printf("\nGenerated Array Sample: ");
    for (int i = 0; i < (n < 10 ? n : 10); i++) {
        printf("%d ", original[i]);
    }
    if(n> 10 ) printf("...");
    printf("\n");

    // QuickSort timing
    double start = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single
        quickSort(arr_quick, 0, n - 1);
    }
    double quick_time = omp_get_wtime() - start;

    // BubbleSort timing
    start = omp_get_wtime();
    bubbleSort(arr_bubble, n);
    double bubble_time = omp_get_wtime() - start;

    // Display timing results
    printf("\n=== Parallel Results (OpenMP) ===");
    printf("\nQuickSort Time: %.6f seconds", quick_time);
    printf("\nBubbleSort Time: %.6f seconds", bubble_time);
    printf("\nQuickSort is %.2fx faster than BubbleSort", bubble_time / quick_time);

    // Display sorted sample
    printf("\n\n=== First 10 Sorted Elements ===");
    printf("\nQuickSort: ");
    for (int i = 0; i < (n < 10 ? n : 10); i++) {
        printf("%d ", arr_quick[i]);
    }
    printf("\nBubbleSort: ");
    for (int i = 0; i < (n < 10 ? n : 10); i++) {
        printf("%d ", arr_bubble[i]);
    }

    // Free memory
    free(original);
    free(arr_quick);
    free(arr_bubble);

    printf("\n");
    return 0;
}

