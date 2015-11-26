//nvcc -arch=sm_20 -lcublas  main.cu

#include <sys/types.h>
#include <time.h>
#include <string.h>
#include <stdlib.h> 
#include <stdio.h>
#include <math.h> 

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>

#define BS 32 
#define R2C(i,j,s) (((j)*(s))+(i))
#define AT(i,j,s) (((i)*(s)) + (j))

#define INFMAX 10000000
#define EPS 0.00001
#define MAX_ITER 10^16

int iter;
int kn, km, km1;
int *devidx;
float *devred, *devtemp;
/**
 * Arrays’ indexes follow the C convention (0 <= i < N)
 **/
//#define DEBUG 
#define CUDA_ERROR_CHECK
 
#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CublasSafeCall( stat ) __cublasSafeCall( stat, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )
 
inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif
 
    return;
}

inline void __cublasSafeCall( cublasStatus_t stat, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if (stat != CUBLAS_STATUS_SUCCESS) { 
        fprintf( stderr, "cublasSafeCall() failed at %s:%i : \n",
                 file, line );
	if (stat == CUBLAS_STATUS_MAPPING_ERROR)
                printf("Error accessing device memory.\n");
        else printf("Setting error.\n");
        exit( -1 );
    }
#endif

    return;
}
 
inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
 
    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif
 
    return;
}

void display_array(const char *name, float *a, int m, int n) {
    int i, j;
    printf("Array %s:\n", name);
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++)
            printf("%f ", a[R2C(i, j, m)]);
        printf("\n");
    }
}


int read_array(FILE *file, float *a, int m, int n) {
    int i, j;
    for (i = 0; i < m; i++)
        for (j = 0; j < n; j++) {
            fscanf(file, "%f", &a[R2C(i, j, m)]); 
        } 
    return 0;
}

/************* KERNELS ***********************/
__global__ void reduce_min(float *f, int n, float *min) {
    int tid = threadIdx.x;
    int j = blockIdx.x * blockDim.x + tid;
    //Each block loads its elements into shared mem,
    //padding if not multiple of BS
    __shared__ float sf[BS];
    sf[tid] = (j < n) ? f[j] : INFMAX;
    __syncthreads();
    //Apply reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sf[tid] = sf[tid] > sf[tid + s] ? sf[tid + s] : sf[tid];
        __syncthreads();
    }
    if (tid == 0) min[blockIdx.x] = sf[0];
}

__global__ void get_val(float *f, int index, float *val) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j == index) *val = f[j];
}


__global__ void get_idx(float *f, int *index, float *val, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j == 0)
        index[0] = -1;
    __syncthreads();
    if (j < n) {
        float diff = f[j] - val[0];
        if (diff >= -EPS && diff <= EPS) atomicCAS(index, -1, j);
    }
}


__global__ void init_cInD(float *c, float *D, int m, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int s = gridDim.x * blockDim.x;
    int id = AT(i, j, s);
    if (id < n) {
        i = id / n;
        j = id % n;
        D[R2C(i, j, m)] = -c[id];
    }
}

__global__ void init_AInD(float *A, float *D, int m, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int s = gridDim.x * blockDim.x;
    int id = AT(i, j, s);
    if (id < m * n) {
        i = id / n;
        j = id % n;
        D[R2C(i + 1, j, m + 1)] = A[R2C(i, j, m)];
    }
}

__global__ void init_I(float *I, int m) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int s = gridDim.x * blockDim.x;
    int id = AT(i, j, s);
    if (id < m * m) {
        i = id / m;
        j = id % m;
        I[R2C(i, j, m)] = (float) (i == j);
    }

}


__global__ void init_bi(int *bi, int m, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int s = gridDim.x * blockDim.x;
    int id = AT(i, j, s);
    if (id < m)
        bi[id] = (n - m) + id;
}
//num_max counts how many alpha[i] are <= 0

__global__ void compute_theta(float *xb, float *alpha, float *theta,
        int *theta_flag, int m, int *num_max) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int s = gridDim.x * blockDim.x;
    int id = AT(i, j, s);
    if (id < m) {
        int cond = (alpha[id] > 0);
        theta_flag[id] = cond;
        theta[id] = xb[id] / alpha[id] * cond + INFMAX * (1 - cond);
        atomicAdd(num_max, 1 - cond);
    }
}

__global__ void compute_new_E(float *E, float *alpha, int m,
        int li, float qth) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int s = gridDim.x * blockDim.x;
    int id = AT(i, j, s);
    if (id < m) {
        alpha[id] = -alpha[id] / qth;
        if (id == li) alpha[id] = 1 / qth;
        E[R2C(id, li, m)] = alpha[id];
    }
}


__global__ void update_bi_cb(int *bi, float *cb, float *c, int li, int ei) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j == li) {
        bi[j] = ei;
        cb[j] = c[ei];
    }
}



/************* WRAPPERS ***********************/
int get_min_idx(cublasHandle_t handle, float *a, int n, float *val) {
    int numBlocks = (int) ceil((float) n / BS);
    int size = n;
    int min_idx = -1;
    cublasScopy(handle, size, a, 1, devtemp, 1);
    do {
        reduce_min <<<numBlocks, BS>>>(devtemp, size, devred);
        size = numBlocks;
        if (numBlocks > 1) {
            cublasScopy(handle, size, devred, 1, devtemp, 1);
            numBlocks = (int) ceil((float) numBlocks / BS);
        }

    } while (size > 1);
    numBlocks = (int) ceil((float) n / BS);
    get_idx <<<numBlocks, BS>>>(a, devidx, devred, n);
    cudaMemcpy(&min_idx, devidx, sizeof (int), cudaMemcpyDeviceToHost);
    if (val != NULL)
        cudaMemcpy(val, devred, sizeof (float), cudaMemcpyDeviceToHost);
    return min_idx;
}

int entering_index(cublasHandle_t handle, float *e, int n) {
    float val_min;
    int min_i = get_min_idx(handle, e, n, &val_min);
    return (val_min >= -EPS) ? -1 : min_i;
}


int leaving_index(cublasHandle_t handle, float *t, int *flag, int size) {
    return get_min_idx(handle, t, size, NULL);
}


float lpsolve(float *A, float *b, float *c, float *xb, int *bi, int m, int n) 
{
    int i, opt;
    cublasStatus_t stat;
    float *devc, *devA, *devb;
    // Binv: Basis matrix inverse
    // newBinv: temporary matrix inverse for swap purposes
    // E: used to compute the inversion using just one mm multiplication
    // newBinv = E * Binv
    float *devBinv, *devnewBinv, *devE;
    // e: cost contributions vector used to determine the entering variable // D, y, yb: arrays used to compute the cost contributions vector //D=[-c;A] y=cb*Binv yb=[1y] e=yb*D
    float *devD, *devy, *devyb, *deve;
    // xb: current basis
    // cb: basis costs
    // xb = Binv * b
    float *devcb, *devxb;
    // A_e: entering variable column of constraint factors
    // alpha: the pivotal vector used to determine the leaving variable
    // theta: Increases vector
    // alpha = Binv * A_e
    float *devA_e, *devalpha, *devtheta;
    // Vector of flags indicating valid increases
    // (valid alpha[i]) <==> (theta_flag[i] == 1)
    int *devtheta_flag;
    // Vector containing basis variables’ indexes
    int *devbi;
    //Counter for unbounded solution checking
    int *devnum_max;
    // Indexes of the entering and leaving variables.

    int ei, li;
    // Cost to optimize
    // z = c * x
    float z;
    //Proper dimensions for kernel grids
    kn = (int) ceil((float) n / BS);
    km = (int) ceil((float) m / BS);
    km1 = (int) ceil((float) (m + 1) / BS);

    printf("grid:%d*%d; kn: %d; km: %d; \n", BS,BS, kn, km);
    //CUBLAS initialization
    cublasHandle_t handle;
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("Device memory allocation failed.\n");
        return 1;
    }
    iter = -1;
    // c
    CudaSafeCall( cudaMalloc((void **) &devc, n*sizeof(*c)) );
    // b
    CudaSafeCall( cudaMalloc((void **) &devb, m*sizeof(*b)) );
    // A
    CudaSafeCall( cudaMalloc((void **) &devA, m*n*sizeof(*A)) ); 
    // Binv, newBinv, E
    CudaSafeCall( cudaMalloc((void **) &devBinv, m*m*sizeof(*devBinv)) );
    CudaSafeCall( cudaMalloc((void **) &devnewBinv, m*m*sizeof(*devnewBinv)) );
    CudaSafeCall( cudaMalloc((void **) &devE, m*m*sizeof(*devE)) ); 
    // D, y, yb, e
    CudaSafeCall( cudaMalloc((void **) &devD, (m + 1)*n*sizeof(*devD)) ); 
    CudaSafeCall( cudaMalloc((void **) &devy, m*sizeof(*devy)) );
    CudaSafeCall( cudaMalloc((void **) &devyb, (m + 1)*sizeof(*devyb)) ); 
    CudaSafeCall( cudaMalloc((void **) &deve, n*sizeof(*deve)) ); 
    // cb, xb
    CudaSafeCall( cudaMalloc((void **) &devcb, m*sizeof(*devcb)) ); 
    CudaSafeCall( cudaMalloc((void **) &devxb, m*sizeof(*devxb)) ); 
    // A_e, alpha, theta
    CudaSafeCall( cudaMalloc((void **) &devA_e, m*sizeof(*devA_e)) );
    CudaSafeCall( cudaMalloc((void **) &devalpha, m*sizeof(*devalpha)) );
    CudaSafeCall( cudaMalloc((void **) &devtheta, m*sizeof(*devtheta)) );
    // red, temp, idx
    CudaSafeCall( cudaMalloc((void **) &devred, km*sizeof(*devred)) );
    CudaSafeCall( cudaMalloc((void **) &devtemp, km*sizeof(*devtemp)) );
    CudaSafeCall( cudaMalloc((void **) &devidx, sizeof(*devidx)) );
    // num_max
    CudaSafeCall( cudaMalloc((void **) &devnum_max, sizeof(*devnum_max)) ); 
    // theta_flag & bi
    CudaSafeCall( cudaMalloc((void **) &devtheta_flag, m*sizeof(*devtheta_flag)) );
    CudaSafeCall( cudaMalloc((void **) &devbi, m*sizeof(*devbi)) ); 


    CublasSafeCall( cublasSetMatrix(m, n, sizeof(*A), A, m, devA, m) );
    CublasSafeCall( cublasSetVector(m, sizeof(*b), b, 1, devb, 1) );
    CublasSafeCall( cublasSetVector(n, sizeof (*c), c, 1, devc, 1) );

    //Initialize yb
    thrust::device_ptr<float> thrst_devyb(devyb);
    thrust::fill(thrst_devyb, thrst_devyb + m, (float) 0);
    thrust::fill(thrst_devyb, thrst_devyb + 1, (float) 1);

    //Initialize D
    init_cInD <<<kn, BS>>>(devc, devD, m + 1, n);
    init_AInD <<<dim3(kn, km1), dim3(BS, BS)>>>(devA, devD, m, n);
    //Initialize devBinv <- Im
    init_I <<<dim3(km, km), dim3(BS, BS)>>>(devBinv, m);
	#ifdef DEBUG
	char str_devBinv[] = "devBinv"; 	
	int DSIZE_x = m;
	int DSIZE_y = m;
	int DSIZE = DSIZE_x * DSIZE_y * sizeof(float);
  	float* h_data = (float *)malloc(DSIZE);
	printf("Iteration %d: ", iter); 

	cudaMemcpy(h_data, devBinv, DSIZE , cudaMemcpyDeviceToHost);
	display_array(str_devBinv, h_data, DSIZE_x, DSIZE_y);
	#endif


    //devcb <- devc[n-m] to devc[n]
    CublasSafeCall( cublasScopy(handle, m, &devc[n - m], 1, devcb, 1) );
	#ifdef DEBUG
	char str_var[] = "devcb"; 	
	DSIZE_x = m;
	DSIZE_y = 1;
	DSIZE = DSIZE_x * DSIZE_y * sizeof(float);
  	h_data = (float *)malloc(DSIZE);
	printf("Iteration %d: ", iter); 

	cudaMemcpy(h_data, devcb, DSIZE , cudaMemcpyDeviceToHost);
	display_array(str_var, h_data, DSIZE_x, DSIZE_y);
	#endif


    //devxb <- devb
    CublasSafeCall( cublasScopy(handle, m, devb, 1, devxb, 1) );
    //devbi[i] = (n-m)+i
     init_bi <<<km, BS>>>(devbi, m, n);
    i = 0;
    iter = 0;
    float al = 1.0f;
    float bet = 0.0f;
    do {
		#ifdef DEBUG
		printf("Iteration %d: \n", iter); 
		#endif
        // y = cb*Binv
        CublasSafeCall( cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, m, m, &al, devcb, 1, devBinv, m, &bet, devy, 1) );
		#ifdef DEBUG
		char str_var[] = "devcb"; 	
		int DSIZE_x = m;
		int DSIZE_y = 1;
		int DSIZE = DSIZE_x * DSIZE_y * sizeof(float);
		float* h_data;
		h_data = (float *)malloc(DSIZE);

		cudaMemcpy(h_data, devcb, DSIZE , cudaMemcpyDeviceToHost);
		display_array(str_var, h_data, DSIZE_x, DSIZE_y);
		#endif

        CublasSafeCall( cublasScopy(handle, m, devy, 1, &devyb[1], 1) );
        // e = [1 y]*[-c ; A]
        CublasSafeCall( cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, n, m + 1, &al, devyb, 1, devD, m + 1, &bet, deve, 1) );
		#ifdef DEBUG
		char str_deve[] = "deve"; 	
		DSIZE_x = n;
		DSIZE_y = 1;
		DSIZE = DSIZE_x * DSIZE_y * sizeof(float);
		h_data = (float *)malloc(DSIZE);

		cudaMemcpy(h_data, deve, DSIZE , cudaMemcpyDeviceToHost);
		display_array(str_deve, h_data, DSIZE_x, DSIZE_y);
		#endif

        ei = entering_index(handle, deve, n);
		#ifdef DEBUG
		printf("entering index(ei): %d\n", ei);
		#endif

        if (ei < 0) {
            opt = 1;
            break;
        }
        // alpha = Binv*A_e
        cublasScopy(handle, m, &devA[R2C(0, ei, m)], 1, devA_e, 1);
		#ifdef DEBUG
		char str_devA_e[] = "devA_e"; 	
		DSIZE_x = m;
		DSIZE_y = 1;
		DSIZE = DSIZE_x * DSIZE_y * sizeof(float);
		h_data = (float *)malloc(DSIZE);

		cudaMemcpy(h_data, devA_e, DSIZE , cudaMemcpyDeviceToHost);
		display_array(str_devA_e, h_data, DSIZE_x, DSIZE_y);
		#endif


        CublasSafeCall( cublasSgemv(handle, CUBLAS_OP_N, m, m, &al, devBinv, m, devA_e, 1, &bet, devalpha, 1) );
		#ifdef DEBUG
		char str_devalpha[] = "devalpha"; 	
		DSIZE_x = m;
		DSIZE_y = 1;
		DSIZE = DSIZE_x * DSIZE_y * sizeof(float);
		h_data = (float *)malloc(DSIZE);

		cudaMemcpy(h_data, devalpha, DSIZE , cudaMemcpyDeviceToHost);
		display_array(str_devalpha, h_data, DSIZE_x, DSIZE_y);
		#endif


        int num_max;
        CudaSafeCall( cudaMemset(devnum_max, 0, 1) );
        compute_theta <<<km, BS>>>(devxb, devalpha, devtheta,
                devtheta_flag, m, devnum_max);
        CudaSafeCall( cudaMemcpy(&num_max, devnum_max, sizeof (int), cudaMemcpyDeviceToHost) );
        if (num_max == m) {
            opt = 2;
            break;
        }
        li = leaving_index(handle, devtheta, devtheta_flag, m);

	#ifdef DEBUG
	    printf("leaving index(li): %d \n", li);
	#endif

	//Compute E, update the basis
	float qth;
        CudaSafeCall( cudaMemcpy(&qth, &devalpha[li], sizeof(float), cudaMemcpyDeviceToHost) );
	if ((qth >= -EPS) && (qth <= EPS)) {
		opt = 3;
		break;
	}
	init_I <<<dim3(km, km), dim3(BS, BS)>>>(devE, m);
	compute_new_E<<<km, BS>>>(devE,devalpha,m,li,qth); 
        // Binv = E*Binv
        CublasSafeCall( cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, m, m, &al, devE, m, devBinv, m, &bet,devnewBinv, m) );
        CublasSafeCall( cublasScopy(handle, m*m, devnewBinv, 1, devBinv, 1) );
		#ifdef DEBUG
		char str_devnewBinv[] = "updated devBinv"; 	
		DSIZE_x = m;
		DSIZE_y = m;
		DSIZE = DSIZE_x * DSIZE_y * sizeof(float);
		h_data = (float *)malloc(DSIZE);

		cudaMemcpy(h_data, devBinv, DSIZE , cudaMemcpyDeviceToHost);
		display_array(str_devnewBinv, h_data, DSIZE_x, DSIZE_y);
		#endif


        //bi[lv] = ev;
        //cb[lv] = c[ev];
        update_bi_cb <<<km, BS>>>(devbi, devcb, devc, li, ei);
		#ifdef DEBUG
		char str_updevcb[] = "updated devcb"; 	
		DSIZE_x = m;
		DSIZE_y = 1;
		DSIZE = DSIZE_x * DSIZE_y * sizeof(float);
		h_data = (float *)malloc(DSIZE);

		cudaMemcpy(h_data, devcb, DSIZE , cudaMemcpyDeviceToHost);
		display_array(str_updevcb, h_data, DSIZE_x, DSIZE_y);
		#endif


        // xb=Binv*b
        CublasSafeCall( cublasSgemv(handle, CUBLAS_OP_N, m, m, &al, devBinv, m, devb, 1, &bet, devxb, 1) );
		#ifdef DEBUG
		char str_updevxb[] = "updated devxb"; 	
		DSIZE_x = m;
		DSIZE_y = 1;
		DSIZE = DSIZE_x * DSIZE_y * sizeof(float);
		h_data = (float *)malloc(DSIZE);

		cudaMemcpy(h_data, devxb, DSIZE , cudaMemcpyDeviceToHost);
		display_array(str_updevxb, h_data, DSIZE_x, DSIZE_y);
		#endif


        i++;
	iter++;
    } while (i < MAX_ITER);
    if (opt == 1) {
        CublasSafeCall( cublasSdot(handle, m, devcb, 1, devxb, 1, &z) );
        CublasSafeCall( cublasGetVector(m, sizeof (*devxb), devxb, 1, xb, 1) );
        CublasSafeCall( cublasGetVector(m, sizeof (*devbi), devbi, 1, bi, 1) );
    } else if (opt == 2)
        z = INFINITY;
    else z = NAN;

    cudaFree(devc);
    cudaFree(devb);
    cudaFree(devA);
    cudaFree(devBinv);
    cudaFree(devnewBinv);
    cudaFree(devE);
    cudaFree(devD);
    cudaFree(devy);
    cudaFree(devyb);
    cudaFree(deve);
    cudaFree(devcb);
    cudaFree(devxb);
    cudaFree(devA_e);
    cudaFree(devalpha);
    cudaFree(devtheta);
    cudaFree(devnum_max);
    cudaFree(devtheta_flag);
    cudaFree(devbi);
    cudaFree(devidx);
    cudaFree(devtemp);
    cudaFree(devred);
    cublasDestroy(handle);
    return z;
}

/****************** MAIN *********************/



int main(int argc, char **argv) {
    cudaEvent_t startEvent, stopEvent;

    CudaSafeCall( cudaEventCreate(&startEvent) );
    CudaSafeCall( cudaEventCreate(&stopEvent) );
    // Main problem arrays: costs and constrains
    float *c, *A, *b, *xb;
    int *bi;
    float z;

    int i, m, n;
    FILE *sourcefile;
    switch (argc) {
        case 2:
                if ((sourcefile = fopen(argv[1], "r")) == NULL) {
                printf("Error opening %s\n", argv[2]);
                return 1;
		}
            break;
        default:
            printf("Wrong parameter sequence.\n");
            return 1;
    }
    // read m and n
    fscanf(sourcefile, "%d %d", &m, &n);


    if (m > n) {
        printf("Error: it should be n>=m\n");
        return 1;
    }
    printf("m=%d n=%d\n", m, n);
    printf("Size: %d\n", m * n);
    //Initialize all arrays

    cudaMallocHost((void **) &c, 1 * n * sizeof (float));
    cudaMallocHost((void **) &b, m * 1 * sizeof (float));
    cudaMallocHost((void **) &A, m * n * sizeof (float));

    // c
    read_array(sourcefile, c, 1, n);
    /*for(int i=0; i < n; i++) {
	c[i] = -c[i];
    }*/
    // b
    read_array(sourcefile, b, m, 1);
    // A
    read_array(sourcefile, A, m, n);

    //Close source file
    fclose(sourcefile);

    #ifdef DEBUG
    char str_c[] = "c ";
    display_array(str_c, c, 1, n);
    char str_b[] = "b ";
    display_array(str_b, b, m, 1);
    char str_A[] = "A ";
    display_array(str_A, A, m, n);
    #endif

    // xb
    cudaMallocHost((void **) &xb, 1 * m * sizeof (float));
    // bi
    cudaMallocHost((void **) &bi, m * 1 * sizeof (int));

    CudaSafeCall( cudaEventRecord(startEvent, 0) );

    z = lpsolve(A, b, c, xb, bi, m, n);

    CudaSafeCall( cudaEventRecord(stopEvent, 0) );
    CudaSafeCall( cudaEventSynchronize(stopEvent) );

    float time;
    CudaSafeCall( cudaEventElapsedTime(&time, startEvent, stopEvent) );

    printf("LP Algorithm running time:( %f )s\n", time/(float)1000);

    FILE* file = fopen("result.txt", "w");
    if (isnan(z))
        printf("Problem unsolvable: either qth==0 or loop too long.\n");
    else if (isinf(z))
        printf("Problem unbounded.\n");
    else {
        printf("Optimum found: %f\n", z);
        for (i = 0; i < m; i++)
            fprintf(file, "x_%d = %f\n", bi[i], xb[i]);
    }
    fclose(file);
    // Deallocate arrays
    cudaFreeHost(A);
    cudaFreeHost(b);
    cudaFreeHost(c);
    cudaFreeHost(xb);
    cudaFreeHost(bi);

}
