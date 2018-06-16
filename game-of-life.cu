/** Author: alexge50
 *  How to use: input should be given in a file input.txt, in the same directory as the binary. Output is given in output.txt
 *  Input:  [Number of steps]
 *          [height - number of rows] [width - number of columns]
 *          board
 *  Output: [time] ms
 *          board at the current state when execution was stopped
 *  Compilation requires no other option than default:
 *     nvcc game-of-life.cout
 *     ./a.out
 **/

#include <stdio.h>

#include <sys/time.h>

inline long long GetTime()
{
    struct timeval tv;

    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000LL + tv.tv_usec;
}

__global__ void update(int *board, int *result_board, int nRows, int nColumns)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.x;
  int i = blockIdx.x;

  if(i >= 1 && i <= nRows - 2 && j >= 1 && j <= nColumns - 2)
  {
    int n_neighbors = 0;
    /*for(int di = -1; di <= 1; di++)
    {
      for(int dj = -1; dj <= 1; dj++)
      {
        int _index = (i + di) * nColumns + (j + dj);
        n_neighbors += board[_index] && !(di == 0 && dj == 0);
      }
    }*/
    n_neighbors = board[(i - 1) * nColumns + (j + 1)] +
                  board[(i + 1) * nColumns + (j + 1)] +
                  board[(i + 1) * nColumns + (j - 1)] +
                  board[(i + 0) * nColumns + (j + 1)] +
                  board[(i + 0) * nColumns + (j - 1)] +
                  board[(i + 1) * nColumns + (j + 0)] +
                  board[(i - 1) * nColumns + (j + 0)] +
                  board[(i - 1) * nColumns + (j - 1)];

    if(board[index])
      atomicExch(&result_board[index], n_neighbors == 2 || n_neighbors == 3);
    else atomicExch(&result_board[index], n_neighbors == 3);
  }
}

int main()
{
    FILE *fin = fopen("input.txt", "r");
    FILE *fout = fopen("output.txt", "w");
    int nSteps;
    int nRows, nColumns;
    long long timeStart, timeStop;

    int *board;
    int *device_board0, *device_board1;
    //char *device_board[2];

    fscanf(fin, "%d %d %d ", &nSteps, &nRows, &nColumns);
    //cudaMalloc((void **) &device_board[0], sizeof(char) * (nRows + 2) * (nColumns + 2));
    //cudaMalloc((void **) &device_board[1], sizeof(char) * (nRows + 2) * (nColumns + 2));

    nColumns += 2;
    nRows += 2;

    board = (int*)malloc(sizeof(int) * (nRows) * (nColumns));

    cudaMalloc((void **) &device_board0, sizeof(int) * (nRows) * (nColumns));
    cudaMalloc((void **) &device_board1, sizeof(int) * (nRows) * (nColumns));


    for(int i = 0; i < nRows; i++)
      for(int j = 0; j < nColumns; j++)
        board[i * nColumns + j] = 0;

    for(int i = 1; i <= nRows - 2; ++i)
        for (int j = 1; j <= nColumns - 2; ++j)
        {
            char cell;
            fscanf(fin, "%c ", &cell);
            board[i * nColumns + j] = (cell == '*');
        }

    cudaMemcpy(device_board0, board, sizeof(int) * (nRows) * (nColumns), cudaMemcpyHostToDevice);
    cudaMemcpy(device_board1, board, sizeof(int) * (nRows) * (nColumns), cudaMemcpyHostToDevice);

    timeStart = GetTime();
    int i = 0;
    for (int k = 0; k < nSteps; ++k)
    {
      //UpdateCall
      //update<<<nRows, nColumns>>>(device_board[i], device_board[i - 1], nRows, nColumns);
      if(i == 0)
    	  update<<<nRows, nColumns>>>(device_board0, device_board1, nRows, nColumns);
      else
    	  update<<<nRows, nColumns>>>(device_board1, device_board0, nRows, nColumns);
      i = 1 - i;
    }
    //cudaMemcpy(board, device_board[i], sizeof(char) * (nRows + 2) * (nColumns + 2), cudaMemcpyDeviceToHost);
    if(i == 0)
    	cudaMemcpy(board, device_board0, sizeof(int) * (nRows) * (nColumns), cudaMemcpyDeviceToHost);
    else cudaMemcpy(board, device_board1, sizeof(int) * (nRows) * (nColumns), cudaMemcpyDeviceToHost);
    timeStop = GetTime();

    long double deltaTime = static_cast<long double>(timeStop - timeStart) / static_cast<long double>(1000.); // microseconds to milli seconds

    fprintf(fout, "[%Lf ms]\n", deltaTime);

    for(int i = 1; i <= nRows - 2; ++i)
    {
        for(int j = 1; j <= nColumns - 2; ++j)
            fprintf(fout, "%c", board[i * nColumns + j] ? '*' : '.');
        fprintf(fout, "\n");
    }

    printf("Time: %Lf ms\n", deltaTime);

    free(board);
    cudaFree(device_board0);
    cudaFree(device_board1);

    fclose(fin);
    fclose(fout);
    return 0;
}
