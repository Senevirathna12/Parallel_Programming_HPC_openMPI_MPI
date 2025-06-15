#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void swap(int* a, int* b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

int partition(int arr[], int low, int high) {
    int pivot = arr[high];
    int i = (low - 1);
    
    for (int j = low; j <= high - 1; j++) {
        if (arr[j] < pivot) {
            i++;
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]);
    return (i + 1);
}

void quickSort(int arr[], int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}

void bubbleSort(int arr[], int n) {
    int i, j, swapped;
    for (i = 0; i < n-1; i++) {
        swapped = 0;
        for (j = 0; j < n-i-1; j++) {
            if (arr[j] > arr[j+1]) {
                swap(&arr[j], &arr[j+1]);
                swapped = 1;
            }
        }
        if (!swapped) break;
    }
}

int main() {
    int n;
    printf("=== HPC Sorting Performance Analysis ===\n");
    
    // Get array size with validation
    printf("Enter array size: ");
    if(scanf("%d", &n) != 1 || n <= 0) {
        printf("Invalid array size!\n");
        return 1;
    }
    
    // Generate random array
    int *arr = (int*)malloc(n * sizeof(int));
    srand(time(0));
    for(int i=0; i<n; i++) {
        arr[i] = rand() % 1000;  // Random numbers between 0-999
    }
    
    // Create copies for sorting
    int *arr_quick = (int*)malloc(n * sizeof(int));
    int *arr_bubble = (int*)malloc(n * sizeof(int));
    memcpy(arr_quick, arr, n*sizeof(int));
    memcpy(arr_bubble, arr, n*sizeof(int));
    
    // Display sample elements
    printf("\nGenerated Array Sample: ");
    for(int i=0; i<n; i++) {
        printf("%d ", arr[i]);
    }

    // Time measurement variables
    clock_t start, end;
    
    // QuickSort execution
    start = clock();
    quickSort(arr_quick, 0, n-1);
    end = clock();
    double quick_time = ((double)(end - start))/CLOCKS_PER_SEC;
    
    // BubbleSort execution
    start = clock();
    bubbleSort(arr_bubble, n);
    end = clock();
    double bubble_time = ((double)(end - start))/CLOCKS_PER_SEC;
    
    // Display results
    printf("\nQuickSort Time: %.6f seconds", quick_time);
    printf("\nBubbleSort Time: %.6f seconds", bubble_time);
    printf("\nSpeed Difference (bubble_time/quick_time ): %.2fx faster\n", 
          bubble_time/quick_time);
    
    // Verification samples
    printf("\nSorted Array elements:");
    printf("\nQuickSort: ");
    for(int i=0; i<n; i++) printf("%d ", arr_quick[i]);
    printf("\nBubbleSort: ");
    for(int i=0; i<n; i++) printf("%d ", arr_bubble[i]);
    
    // Cleanup
    free(arr);
    free(arr_quick);
    free(arr_bubble);
    
    printf("\n");
    return 0;
}

