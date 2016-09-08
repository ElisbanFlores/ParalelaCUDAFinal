
#pragma once 
//////////////////////////////////////////////////////
// Elisban Flores
// Franci Suni
// Universidad Catolica San Pablo - Arequipa
// Maestria en Ciencia de la Computacion
// Programacion Paralela - CUDA
/////////////////////////////////////////////////////

/**********************************************
* LIBRERIAS DE OPENCV PARA MANEJO DE IMAGENES
**********************************************/

#include <opencv2/core/core.hpp> 
#include <opencv2/highgui/highgui.hpp> 

/**********************************************
* LIBRERIAS DE CUDA
**********************************************/

#include "cuda_runtime.h" 
#include "device_launch_parameters.h" 

/**********************************************
* OTRAS LIBRERIAS
**********************************************/
#include <stdio.h> 
#include <type_traits> 
#include <cmath> 
#include <time.h>
#include <iostream>
#include <math.h>
#include <vector>
#include <float.h>


/**************************
* DEFINICION DEL TIPO PIXEL
**************************/

typedef unsigned char uInt;
typedef unsigned char Pixel;
//typedef float Pixel;
//typedef float Scalar;

/*********************************************
* PARA VERIFICAR ERRORES DE CUDA QUE SE DESENCADENA DESDE EL HOST
*********************************************/

#define checkCudaErrors(err)           __checkCudaErrors (err, __FILE__, __LINE__)

inline void __checkCudaErrors(cudaError err, const char *file, const int line)
{
	if (cudaSuccess != err)
	{
		fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
			file, line, (int)err, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}


/**********************************************
* FUNCION PARA OBTENER EL TIEMPO EN CPU
**********************************************/

double getMilisegundos(clock_t c)
{
	double tiempo = 0;
	tiempo = ((c / (double)CLOCKS_PER_SEC) * 1000);
	return tiempo;
}


using namespace std;
using namespace cv;


#define vec2dd vector<vector<double> >
#define vec2di vector<vector<int> >
#define vec2db vector<vector<bool> >


#define NR_ITERATIONS 10

struct h_Point
{
	size_t i;
	size_t j;
};

struct h_Scalar
{
	Pixel val[3];
};

struct h_Center
{
	double val[5];
};

int step, nc, ns;
int nr_superpixels;
size_t nr_rows, nr_cols, nr_centers;
int modo;

clock_t h_tIni, h_tFin, h_tTotal; //  Para calculo de tiempo en CPU
cudaEvent_t d_tIni, d_tFin; float d_tTotal; // Para calculo de tiempo en GPU

/*********************************************
* extrae el contenido de un cv::Mat_<T> a un puntero a T
*********************************************/

template<class T>
static T *Mat2Pointer(cv::Mat img)
{
	T *ptr = new T[img.rows * img.cols];
	for (int i = 0; i < img.rows; i++)
		memcpy(&(ptr[i*img.cols]), img.ptr<T>(i, 0), img.cols * sizeof(T));
	return ptr;
}

/*********************************************
* copia el contenido de un puntero a T hacia cv::Mat_<T>
*********************************************/

template<class T>
static cv::Mat Pointer2Mat(T *ptr, size_t rows, size_t cols)
{
	cv::Mat img;

	img = cv::Mat::zeros(rows, cols, CV_8UC3);

	for (int i = 0; i < img.rows; i++)
		memcpy(img.ptr<T>(i, 0), &(ptr[i*img.cols]), img.cols * sizeof(T));
	return img;
}


/*********************************************
*  H  O  S  T      H  O  S  T
*********************************************/

/*********************************************
* ESTRUCTURA H_MAT PARA HOST
*********************************************/

template<class T>
struct h_Mat
{
	T *ptr;
	size_t 	*pitch, *rows, *cols;

	inline T Get(size_t row, size_t col)
	{
		return *((T*)((char*)ptr + row * *pitch) + col);
	}

	inline void Set(size_t row, size_t col, T value)
	{
		*((T*)((char*)ptr + row * *pitch) + col) = value;
	}
};


/*********************************************
* Matriz 2d en memoria del CPU (rows x cols)
*********************************************/
template<class T>
static inline h_Mat<T> h_CrearMat2D(size_t rows, size_t cols, T *src)
{
	h_Mat<T> dst;
	size_t 	width;
	width = cols * sizeof(T);
	dst.ptr = src;
	dst.pitch = new size_t[1]{width};
	dst.rows = new size_t[1]{rows};
	dst.cols = new size_t[1]{cols};
	return dst;
}

////////////////////////
// Variables para Host
////////////////////////


h_Mat<h_Scalar> h_img;
h_Mat<int> clusters;
h_Mat<double> distances;
h_Center *centers;

int *center_counts;

//////////////////////////////////////////////
// PARA PARALELIZAR
////////////////////////////
template <class T>
void h_calculateDistancesClusters(h_Mat<T> _image, h_Center *_centers, int _nr_centers, h_Mat<double> _distances, h_Mat<int> _clusters, int _step, size_t _nr_rows, size_t _nr_cols)
{
	for (int j = 0; j < _nr_centers; j++)
	{

		
		for (int k = _centers[j].val[3] - _step; k < _centers[j].val[3] + _step; k++)
		{
			for (int l = _centers[j].val[4] - _step; l < _centers[j].val[4] + _step; l++)
			{
		
				if (k >= 0 && k < _nr_rows && l >= 0 && l < _nr_cols)
				{
					T colour = _image.Get(k, l);
					h_Point c;
					c.i = k;
					c.j = l;
					double d;
					d = h_compute_dist(j, c, colour);

					if (d < _distances.Get(k, l))//distances[k][l]) 
					{	
						_distances.Set(k, l, d);		
						_clusters.Set(k, l, j);						
					}
				}
			}
		}
	}
}


template <class T>
double h_compute_dist(int ci, h_Point pixel, T colour)
{
	double dc = sqrt(pow(centers[ci].val[0] - colour.val[0], 2) +
		pow(centers[ci].val[1] - colour.val[1], 2) +
		pow(centers[ci].val[2] - colour.val[2], 2));
	double ds = sqrt(pow(centers[ci].val[3] - pixel.i, 2) +
		pow(centers[ci].val[4] - pixel.j, 2));
	return sqrt(pow(dc / nc, 2) + pow(ds / ns, 2));
}



void LimpiarMemoriaHost()
{
	delete h_img.ptr;
	delete clusters.ptr;
	delete distances.ptr;
	delete centers;
	delete center_counts;	
}


/*********************************************
*  H  A  S  T  A     A  C  A     H  O  S  T
*********************************************/


/////////////////////////////////////////////////////////////////////
//
/////////////////////////////////////////////////////////////////////



/*********************************************
*  D  E  V  I  C  E        D  E  V  I  C  E
*********************************************/

template<class T>
struct d_Mat
{
	T *ptr;
	size_t 	*pitch, *rows, *cols;

	inline __device__ T Get(size_t row, size_t col)
	{
		return *(ptr + row*(*cols) + col);
	}

	inline __device__ void Set(size_t row, size_t col, T value)
	{		
		*(ptr + row*(*cols) + col) = value;
	}		
};

/*********************************************
* Matriz 2d en memoria del GPU (rows x cols)
*********************************************/

template<class T>
static inline d_Mat<T> d_CrearMat2D(size_t rows, size_t cols, T *src)
{
	d_Mat<T> dst;
	size_t pitch, width, height;
	pitch = cols * sizeof(T);
	width = cols;
	height = rows;
		
	checkCudaErrors(cudaMalloc(&(dst.ptr), width* height*sizeof(T)));
	checkCudaErrors(cudaMalloc(&(dst.pitch), sizeof(size_t)));
	checkCudaErrors(cudaMalloc(&(dst.rows), sizeof(size_t)));
	checkCudaErrors(cudaMalloc(&(dst.cols), sizeof(size_t)));
	
	checkCudaErrors(cudaMemcpy(dst.ptr, src, width* height*sizeof(T), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dst.pitch, &pitch, sizeof(size_t), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dst.rows, &rows, sizeof(size_t), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dst.cols, &cols, sizeof(size_t), cudaMemcpyHostToDevice));
	return dst;
}

///////////////////////////
// Variables para Device
///////////////////////////


d_Mat<h_Scalar> d_img;
d_Mat<int> d_clusters;
d_Mat<double> d_distances;
h_Center *d_centers;
double *d_distances_res;//not use

int *d_center_counts;

//////////////////////////////////////////////
// PARA PARALELIZAR
////////////////////////////
template <class T>
__global__ void d_calculateDistancesClusters(d_Mat<T> _image, h_Center *_centers, int _nr_centers, d_Mat<double> _distances, d_Mat<int> _clusters, int _step, size_t _nr_rows, size_t _nr_cols, double _nc, double _ns)
{	
	T colour;
	double d;
	int k, l;
	double dc, ds;

	size_t tid =  threadIdx.x + blockIdx.x * blockDim.x;
		
	if (tid < _nr_centers)// && tid_y < _nr_cols)//_nr_centers)
	{			
		//__syncthreads();
		d_updateDistancesClusters(_image, _distances, _clusters, _centers[tid], _step, tid, _nr_rows, _nr_cols,_nc,_ns);
		//__syncthreads();	
	}
}

template <class T>
__device__ double d_compute_dist(h_Point pixel, T colour, h_Center centers, double _nc, double _ns)
{
	double dc = sqrt(pow(centers.val[0] - colour.val[0], 2) +
		pow(centers.val[1] - colour.val[1], 2) +
		pow(centers.val[2] - colour.val[2], 2));
	double ds = sqrt(pow(centers.val[3] - pixel.i, 2) +
		pow(centers.val[4] - pixel.j, 2));
	return sqrt(pow(dc / _nc, 2) + pow(ds / _ns, 2));
}

__device__ void d_updateDistancesClusters(d_Mat<h_Scalar> _image, d_Mat<double> _distances, d_Mat<int> _clusters, h_Center _centers, int _step, size_t ind, size_t _nr_rows, size_t _nr_cols,double _nc, double _ns )
{
	h_Scalar colour;
	int k, l;
	double d;
	double dc, ds;
	int count = 0;

	for (k = _centers.val[3] - _step; k < _centers.val[3] + _step; k++)
	{
		for (l = _centers.val[4] - _step; l < _centers.val[4] + _step; l++)
		{
			if (k >= 0 && k < _nr_rows && l >= 0 && l < _nr_cols)
			{
				colour = _image.Get(k, l);
				h_Point c;	c.i = k;	c.j = l;
								
				d = d_compute_dist(c, colour, _centers, _nc, _ns);
			
				if (d < _distances.Get(k, l))
				{					
					_distances.Set(k, l, d);
					_clusters.Set(k, l, ind);					
				}			
			}	
		}
	}
}


void LimpiarMemoriaDevice()
{
	checkCudaErrors(cudaFree(d_img.ptr));
	checkCudaErrors(cudaFree(d_clusters.ptr));
	checkCudaErrors(cudaFree(d_distances.ptr));
	if (modo = 2)
	{
		checkCudaErrors(cudaFree(d_centers));
		checkCudaErrors(cudaFree(d_center_counts));
	}
}


/**************************************************
*  H  A  S  T  A     A  C  A     D  E  V  I  C  E
**************************************************/

/*************************
* Inicializar data
*************************/
template <class T>
void init_data(h_Mat<T> image) 
{
	
	for (int i = 0; i < nr_rows; i++) 
	{	
		vector<double> dr;
		for (int j = 0; j < nr_cols; j++) 
		{
			clusters.Set(i, j, -1);	
			distances.Set(i, j, FLT_MAX);
		}		
	}

	int k = 0;
	
	for (int i = step; i < nr_rows - step / 2; i += step) 
	{
		for (int j = step; j < nr_cols - step / 2; j += step) 
		{
			vector<double> center;
	
			h_Point nc;
			h_Point c;
			c.i = i;
			c.j = j;
	
			nc=find_local_minimum(image, c);
			
			T colour = image.Get(nc.i, nc.j);
			
			centers[k].val[0] = colour.val[0];
			centers[k].val[1] = colour.val[1];
			centers[k].val[2] = colour.val[2];
			centers[k].val[3] = nc.i;
			centers[k].val[4] = nc.j;
						
			center_counts[k]=0;
			k = k + 1;
		}
	}
	cout << "Init data terminado" << endl;
}


/**********************************
* Encontrar el gradiente minimo
***********************************/

template <class T>
h_Point find_local_minimum(h_Mat<T> image, h_Point center) 
{
	double min_grad = FLT_MAX;
	h_Point loc_min;
	loc_min.i = center.i;
	loc_min.j = center.j;
	for (int i = center.i - 1; i < center.i + 2; i++) 
	{
		for (int j = center.j - 1; j < center.j + 2; j++) 
		{
			T c1 = image.Get(i, j + 1);
			T c2 = image.Get(i + 1, j);
			T c3 = image.Get(i,j);

			double i1 = c1.val[0];
			double i2 = c2.val[0];
			double i3 = c3.val[0];			

			if (sqrt(pow(i1 - i3, 2)) + sqrt(pow(i2 - i3, 2)) < min_grad) {
				min_grad = fabs(i1 - i3) + fabs(i2 - i3);
				loc_min.i = i;
				loc_min.j = j;
			}
		}
	}
	
	return loc_min;
}


/***********************************
* Calcular la sobresegmentacion
***********************************/

template <class T>
void generate_superpixels(h_Mat<T> image) {

	init_data(image); //host
	
	cout << "tamano de nr centros en bytes: " << nr_centers * sizeof(h_Center) << endl;

	// inicializar tiempos
	h_tTotal = 0;
	d_tTotal = 0;
	
	if (modo == 2)// en GPU
	{
		checkCudaErrors(cudaMalloc((void**)&d_centers, nr_centers * sizeof(h_Center)));
		checkCudaErrors(cudaMalloc((void**)&d_distances_res, nr_rows*nr_cols * sizeof(double)));
		checkCudaErrors(cudaMemcpy(d_img.ptr, image.ptr, nr_cols* nr_rows*sizeof(h_Scalar), cudaMemcpyHostToDevice));
	}
	
	
	for (int i = 0; i < NR_ITERATIONS; i++)		
	{	
		for (int j = 0; j < nr_rows; j++)
		{
			for (int k = 0; k < nr_cols; k++)
			{	
				distances.Set(j, k, FLT_MAX);
			}
		}
		
		if (modo == 2) // En GPU
		{
			int Blocks, Threads;
			Blocks = nr_cols;
			Threads = 256;

			float d_ttemp;
			cudaEventCreate(&d_tIni);
			cudaEventCreate(&d_tFin);
			cudaEventRecord(d_tIni, 0);
			
			checkCudaErrors(cudaMemcpy(d_clusters.ptr, clusters.ptr, nr_cols* nr_rows*sizeof(int), cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(d_distances.ptr, distances.ptr, nr_cols* nr_rows*sizeof(double), cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(d_centers, centers, nr_centers * sizeof(h_Center), cudaMemcpyHostToDevice));

			cudaEventRecord(d_tFin, 0);
			cudaEventSynchronize(d_tFin);
			cudaEventElapsedTime(&d_ttemp, d_tIni, d_tFin);
			cout << "Iter: " << i << " - Copiar device, ejecutado en: " << d_ttemp << "ms" << endl;
			d_tTotal += d_ttemp;


			cudaEventCreate(&d_tIni);
			cudaEventCreate(&d_tFin);
			cudaEventRecord(d_tIni, 0);

			d_calculateDistancesClusters<h_Scalar> << <Blocks, Threads >> >(d_img, d_centers, nr_centers, d_distances, d_clusters, step, nr_rows, nr_cols, nc, ns);

			cudaEventRecord(d_tFin, 0);
			cudaEventSynchronize(d_tFin);
			cudaEventElapsedTime(&d_ttemp, d_tIni, d_tFin);
			cout << "Iter: " << i << " - En device, ejecutado en: " << d_ttemp << "ms" << endl;
			d_tTotal += d_ttemp;

			cudaEventCreate(&d_tIni);
			cudaEventCreate(&d_tFin);
			cudaEventRecord(d_tIni, 0);

			checkCudaErrors(cudaMemcpy(clusters.ptr, d_clusters.ptr, nr_cols* nr_rows*sizeof(int), cudaMemcpyDeviceToHost));
			checkCudaErrors(cudaMemcpy(distances.ptr, d_distances.ptr, nr_cols* nr_rows*sizeof(double), cudaMemcpyDeviceToHost));
			
			cudaEventRecord(d_tFin, 0);
			cudaEventSynchronize(d_tFin);
			cudaEventElapsedTime(&d_ttemp, d_tIni, d_tFin);
			cout<<"Iter: "<<i<<" - Copiar a host, ejecutado en: "<< d_ttemp<<"ms"<<endl<<endl;
			

			d_tTotal += d_ttemp;

		}
		
		else //  en CPU
		{	
			h_tIni = clock();
			h_calculateDistancesClusters<h_Scalar>(image, centers, nr_centers, distances, clusters, step, nr_rows, nr_cols);
			h_tFin = clock();			
			cout << "Iter: " << i << " - ejecutado en: " << getMilisegundos(h_tFin - h_tIni) << "ms" << endl;
			h_tTotal += getMilisegundos(h_tFin - h_tIni);
		}
				
				
		for (int j = 0; j < nr_centers; j++)
		{
			centers[j].val[0] = centers[j].val[1] = centers[j].val[2] = centers[j].val[3] = centers[j].val[4] = 0;
			center_counts[j] = 0;
		}

		
		for (int j = 0; j < nr_rows; j++)
		{
			for (int k = 0; k < nr_cols; k++)
			{
				int c_id = clusters.Get(j, k);

				if (c_id != -1)
				{
					T colour = image.Get(j, k);

					centers[c_id].val[0] += colour.val[0];
					centers[c_id].val[1] += colour.val[1];
					centers[c_id].val[2] += colour.val[2];
					centers[c_id].val[3] += j;
					centers[c_id].val[4] += k;

					center_counts[c_id] += 1;
				}
			}
		}
				
		for (int j = 0; j < nr_centers; j++)
		{
			centers[j].val[0] /= center_counts[j];
			centers[j].val[1] /= center_counts[j];
			centers[j].val[2] /= center_counts[j];
			centers[j].val[3] /= center_counts[j];
			centers[j].val[4] /= center_counts[j];
		}
	}
	cout << "superpixels terminado"<<endl;

	if (modo = 2)
		cout << "Tiempo total utilizado en GPU: " << d_tTotal << " ms" << endl;
	else
		cout << "Tiempo total utilizado en CPU: " << h_tTotal << " ms" << endl;
	
}


/******************************************
* Mostrar los centros de los clusters.
******************************************/
template <class T>
void display_center_grid(h_Mat<T> image, T colour) 
{
	for (int i = 0; i < nr_centers; i++) 
	{	
		image.Set( centers[i].val[3], centers[i].val[4], colour);
	}
}

/************************************
* mostrar el contorno del cluster.
*************************************/

template <class T>
void display_contours(h_Mat<T> image, T colour) 
{
	const int dx8[8] = { -1, -1, 0, 1, 1, 1, 0, -1 };
	const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };
	
	cout <<endl<< "contorno started" << endl;
		
	vector<h_Point> contours;
	h_Point temp;
	
	int *istaken;
	istaken = new int[nr_rows*nr_cols];
	
	
	for (int i = 0; i < nr_rows; i++) 			
		for (int j = 0; j < nr_cols; j++) 
		{
			istaken[i*nr_cols + j] = 0;
		}		
	
		
	for (int i = 0; i < nr_rows; i++) 
	{
		for (int j = 0; j < nr_cols; j++) 
		{
			int nr_p = 0;
				
			for (int k = 0; k < 8; k++) 
			{
				int x = i + dx8[k], y = j + dy8[k];

				if (x >= 0 && x < nr_rows && y >= 0 && y < nr_cols) 
				{
					if (istaken[x*nr_cols+y] == 0 && clusters.Get(i,j) != clusters.Get(x,y)) 
					{
						nr_p += 1;
					}
				}
			}
						
			if (nr_p >= 2) 
			{
				
				temp.i = i;
				temp.j = j;
				contours.push_back(temp);
				istaken[i*nr_cols + j] = 1;
			}
		}
	}

	cout << "contorno; recorrido de pixels completed" << endl;

	for (int i = 0; i < (int)contours.size(); i++) 
	{
		image.Set(contours[i].i, contours[i].j, colour);		
	}
	cout << "contorno finished" << endl;
}

/***********************************
* Asignar un color a cada Cluster
************************************/
template <class T>
void colour_with_cluster_means(h_Mat<T> image) 
{
	
	vector<float> coloursT(nr_centers*3);
	vector<T> colours(nr_centers);
	cout <<endl<< "Asignar color al cluster" << endl;
	
	for (int i = 0; i < nr_rows; i++) 
	{
		for (int j = 0; j < nr_cols; j++) 
		{			
			int index = clusters.Get(i, j);
			
			if (index < 0)
				index = 0;
				
			T colour = image.Get( i, j);
			
			coloursT[index * 3] += colour.val[0];
			coloursT[index * 3 + 1] += colour.val[1];
			coloursT[index * 3 + 2] += colour.val[2];
		}
	}
		
	for (int i = 0; i < (int)colours.size(); i++) 
	{
		coloursT[i * 3] /= center_counts[i];
		coloursT[i * 3 + 1] /= center_counts[i];
		coloursT[i * 3 + 2] /= center_counts[i];

		colours[i].val[0] = (Pixel)(coloursT[i * 3]);
		colours[i].val[1] = (Pixel)(coloursT[i * 3+1]);
		colours[i].val[2] = (Pixel)(coloursT[i * 3+2]);
	}
		
	int index;

	for (int i = 0; i < nr_rows; i++) 
	{	
		for (int j = 0; j < nr_cols; j++) 
		{			
			index = clusters.Get(i, j);
			if (index < 0)
				index = 0;
				
			
			T ncolour = colours[index];			
			image.Set(i, j, ncolour);
		}
	}
	cout << "Asignacion de color finalizado"<< endl;
}


/*********************************************
* PARTE PRINCIPAL
*********************************************/

int main() {

	Mat imagen;
		
	imagen = cv::imread("im1.png", CV_LOAD_IMAGE_COLOR);
	imagen.convertTo(imagen, CV_8UC3);//RGB

	nr_cols = imagen.cols;
	nr_rows = imagen.rows;
	nr_centers=0;
	
	//Definir modo de ejecicion
	modo = 2;// 1 CPU; 2 GPU
	
	// Inicializar variables
	nr_superpixels =4000;
	nc = 75;
	
	step = sqrt((nr_rows * nr_cols) / (double)nr_superpixels);
	ns = step;
	
	for (int i = step; i < nr_rows - step / 2; i += step)
		for (int j = step; j < nr_cols - step / 2; j += step)
			nr_centers += 1;//numero de centros de acuerdo al tamaño del step
			
	// Para el Host

	centers = new h_Center[nr_centers];//RGb y X,Y
	center_counts = new int[nr_centers];//contador de cada superpixel
	int *h_ptrclusters;
	double *h_ptrdistances;
	h_ptrclusters = new int[nr_rows*nr_cols];
	clusters = h_CrearMat2D<int>(nr_rows, nr_cols,h_ptrclusters);//asignando memoria en el host

	h_ptrdistances = new double[nr_rows*nr_cols];
	distances = h_CrearMat2D<double>(nr_rows, nr_cols, h_ptrdistances);
	
	// Para el device
	d_center_counts = new int[nr_centers];
	
	int *d_ptrclusters;
	double *d_ptrdistances;
	d_clusters = d_CrearMat2D<int>(nr_rows, nr_cols, h_ptrclusters);	
	d_distances = d_CrearMat2D<double>(nr_rows, nr_cols, h_ptrdistances);


	cout << "step: " << step << endl;
		
	// Para la imagen
		
	h_Scalar *src_img;	//point RGB
	src_img = Mat2Pointer<h_Scalar>(imagen);//puntero de la imagen

	
	// copiar imagen para host
	h_img = h_CrearMat2D<h_Scalar>(nr_rows, nr_cols, src_img);

	// copiar imagen para device
	d_img = d_CrearMat2D<h_Scalar>(nr_rows, nr_cols, src_img);

		
	// Calcular la sobresegmentacion
	generate_superpixels<h_Scalar>(h_img);	
		
	//	Asignar un color al Cluster
	//colour_with_cluster_means<h_Scalar>(h_img);
	
	h_Scalar color;	color.val[0] = 0;	color.val[1] = 0;	color.val[2] = 0;
		
	// Mostrar el contorno
	display_contours(h_img, color);
	
	//Mostrar el centro del cluster
	//display_center_grid(h_img, color);
	
	imagen = Pointer2Mat<h_Scalar>(h_img.ptr, nr_rows, nr_cols);
	imagen.convertTo(imagen, CV_8UC3);


	// Guardar y visualizar la imagen guardada
	char nimgfile[80];
	strcpy(nimgfile, "CUDA-RES");
	strcat(nimgfile, ".jpg");
	imwrite(nimgfile, imagen);
	imshow("Resultado", imagen);	
	cout << "termine....";
	
	waitKey(0);

	// Limpiar memoria...
	LimpiarMemoriaDevice();
	LimpiarMemoriaHost();

	

	// fin....
}










