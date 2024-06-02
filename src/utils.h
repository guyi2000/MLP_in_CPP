/*
	util.h，存放工具类，常用函数等
*/
#pragma once

#ifndef __UTIL_H__
#define __UTIL_H_

#include <cstdlib>
#include <omp.h>
#include <cmath>
#include <ctime>
using namespace std;

#include "macro.h"

template<typename T>
class numcpp;	//类声明，用作友元

template<typename T>
class MemoryPool	//using for unique DataStructor
{
private:
	T* __p;
	T* __now_ptr;
	int __size;
public:
	MemoryPool(int _size = DEFUALT_MEMORYPOOL_SIZE);
	~MemoryPool() {
		if (__p != nullptr)
			delete []__p; 
		__p = nullptr;
		__now_ptr = nullptr;
		__size = 0;
	}
	T* alloc(int _size);
};

template<typename T>
inline MemoryPool<T>::MemoryPool(int _size) :__size(_size) {
	//units numbers
	if (0 < _size) {
		__p = new T[_size];
		__now_ptr = __p;
		if (__p != nullptr) {
			memset(__p, 0, _size * sizeof(T));
			return;
		}
		throw ERRORS::NOT_ENOUGH_MEMORY_ERROR;
	}
	throw ERRORS::OUT_OF_RANGE_ERROR;
}

template<typename T>
inline T* MemoryPool<T>::alloc(int _size)
{
	if (_size > __size)
		throw ERRORS::NOT_ENOUGH_MEMORYPOOL_ERROR;
	if (0 >= _size)
		throw ERRORS::OUT_OF_RANGE_ERROR;
	__now_ptr += _size;
	__size -= _size;
	return __now_ptr - _size;
}

template<typename T>
class Vector	//using for dot product
{
/*
	A foot data structure: must using MemoryPool for alloc space
*/
private:
	T* __p;
	int __size;
public:
	Vector(MemoryPool<T>& mp, int _size = DEFUALT_VECTOR_SIZE);
	Vector(MemoryPool<T>& mp, const Vector<T>& a);
	int size() const { return __size; }
	T& operator[](RANK i) const { return __p[i]; }
	void operator+=(const Vector<T>& a);
	void operator-=(const Vector<T>& a);
	void operator*=(const Vector<T>& a);
	T operator*(const Vector<T>& a);	//dot product
	friend class numcpp<T>;
};

template<typename T>
inline Vector<T>::Vector(MemoryPool<T>& mp, int _size) : __size(_size)
{
	__p = mp.alloc(_size);
}

template<typename T>
inline Vector<T>::Vector(MemoryPool<T>& mp, const Vector<T>& a)
{
	__size = a.size();
	__p = mp.alloc(__size);
	memcpy(__p, &a[0], __size * sizeof(T));
}

template<typename T>
inline void Vector<T>::operator+=(const Vector<T>& a)
{
	if (a.size() != __size)
		throw ERRORS::LENGTH_NOT_MATCH_ERROR;
#pragma omp parallel for
	for (int i = 0; i < __size; i++) {
		this->__p[i] = this->__p[i] + a[i];
	}
}

template<typename T>
inline void Vector<T>::operator-=(const Vector<T>& a)
{
	if (a.size() != __size)
		throw ERRORS::LENGTH_NOT_MATCH_ERROR;
#pragma omp parallel for
	for (int i = 0; i < __size; i++) {
		this->__p[i] = this->__p[i] - a[i];
	}
}

template<typename T>
inline void Vector<T>::operator*=(const Vector<T>& a)
{
	if (a.size() != __size)
		throw ERRORS::LENGTH_NOT_MATCH_ERROR;
#pragma omp parallel for
	for (int i = 0; i < __size; i++) {
		this->__p[i] = this->__p[i] * a[i];
	}
}

template<typename T>
inline T Vector<T>::operator*(const Vector<T>& a)
{
	if (a.size() != __size)
		throw ERRORS::LENGTH_NOT_MATCH_ERROR;
	T ans = 0;
#pragma omp parallel for reduction(+:ans)
	for (RANK i = 0; i < __size; i++) {
		ans += __p[i] * a[i];
	}
	return ans;
}

template<typename T>
class Matrix {
/*
	Matrix类用作基本矩阵，实现了基本的矩阵运算、操作，高阶操作参见numcpp类
*/
private:
	MemoryPool<T*>* __rowp;
	MemoryPool<T>* __elemp;
	T** __matrix;
	int __x;
	int __y;
public:
	Matrix():__rowp(NULL),__elemp(NULL),__matrix(NULL),__x(NULL),__y(NULL){};
	Matrix(int x, int y);
	Matrix(Matrix<T>& A);	//复制构造，深拷贝
	~Matrix();
	void operator=(Matrix<T>& A);	//赋值，浅拷贝
	Matrix<T>* operator+(Matrix<T>& A);	//基本算术运算
	Matrix<T>* operator-(Matrix<T>& A);
	Matrix<T>* operator*(Matrix<T>& A);
	void operator+=(Matrix<T> & A);
	void operator-=(Matrix<T> & A);
	void operator*=(Matrix<T> & A);
	void operator*=(const T c);
	void copy_(Matrix<T>& A, int start_row, int end_row);	// 行复制，深拷贝，复制矩阵的[start_row, end_row)，左闭右开
	T*& operator[](RANK i) const { return __matrix[i]; }	// 可以类似二维数组调用，便捷
	Matrix<T>* Trans();	//转置
	int x();	//返回__x
	int y();	//返回__y
	friend class numcpp<T>;	//友元类，numcpp，便于对矩阵处理
};

template<typename T>
inline Matrix<T>::Matrix(int x, int y):__x(x),__y(y)
{
	__rowp = new MemoryPool<T*>(x);
	__matrix = __rowp->alloc(x);
	__elemp = new MemoryPool<T>(x * y);
	for (int i = 0; i < x; i++) 
		__matrix[i] = __elemp->alloc(y);	//内存连续，便于处理
}

template<typename T>
inline Matrix<T>::Matrix(Matrix<T>& A):__x(A.__x),__y(A.__y)
{
	__rowp = new MemoryPool<T*>(__x);
	__matrix = __rowp->alloc(__x);
	__elemp = new MemoryPool<T>(__x * __y);
	for (int i = 0; i < __x; i++) {
		__matrix[i] = __elemp->alloc(__y);
	}
	for (int i = 0; i < __x * __y; i++) {
		__matrix[0][i] = A.__matrix[0][i];	//内存连续的好处凸显
	}
}

template<typename T>
inline Matrix<T>::~Matrix()
{
	if (__rowp != nullptr)
		delete __rowp;	//delete调用上级析构函数，完成整个析构过程
	if(__elemp != nullptr)
		delete __elemp;
	__rowp = nullptr;
	__elemp = nullptr;
	__matrix = nullptr;
	__x = 0;
	__y = 0;
}

template<typename T>
inline void Matrix<T>::operator=(Matrix<T>& A)
{	//浅拷贝，便于释放
	this->__elemp = A.__elemp;
	this->__rowp = A.__rowp;
	this->__matrix = A.__matrix;
	this->__x = A.__x;
	this->__y = A.__y;
}

template<typename T>
inline Matrix<T>* Matrix<T>::operator+(Matrix<T>& A)
{	
	auto ans = new Matrix<T>(A.__x, A.__y);
	for (int i = 0; i < A.__x * A.__y; i++)
	{
		ans->__matrix[0][i] = this->__matrix[0][i] + A.__matrix[0][i];	//使用this
	}
	return ans;
}

template<typename T>
inline Matrix<T>* Matrix<T>::operator-(Matrix<T>& A)
{
	auto ans = new Matrix<T>(A.__x, A.__y);
	for (int i = 0; i < A.__x * A.__y; i++)
	{
		ans->__matrix[0][i] = this->__matrix[0][i] - A.__matrix[0][i];
	}
	return ans;
}

template<typename T>
inline Matrix<T>* Matrix<T>::operator*(Matrix<T>& A)
{
	auto ans = new Matrix<T>(A.__x, A.__y);
	for (int i = 0; i < A.__x * A.__y; i++)
	{
		ans->__matrix[0][i] = this->__matrix[0][i] * A.__matrix[0][i];
	}
	return ans;
}

template<typename T>
inline void Matrix<T>::operator+=(Matrix<T>& A) {
	for (int i = 0; i < A.__x * A.__y; i++)
		this->__matrix[0][i] += A.__matrix[0][i];	//就地运算，节省内存
}

template<typename T>
inline void Matrix<T>::operator-=(Matrix<T>& A) {
	for (int i = 0; i < A.__x * A.__y; i++)
		this->__matrix[0][i] -= A.__matrix[0][i];
}

template<typename T>
inline void Matrix<T>::operator*=(Matrix<T>& A) {
	for (int i = 0; i < A.__x * A.__y; i++)
		this->__matrix[0][i] *= A.__matrix[0][i];
}

template<typename T>
inline void Matrix<T>::operator*=(const T c)
{
	for (int i = 0; i < __x * __y; i++)
		this->__matrix[0][i] *= c;	//乘以常数
}

template<typename T>
inline void Matrix<T>::copy_(Matrix<T>& A, int start_row, int end_row)
{	// 行复制，深拷贝，复制矩阵的[start_row, end_row)，左闭右开
	int row_num = end_row - start_row;
	this->__x = row_num;
	this->__y = A.__y;
	__rowp = new MemoryPool<T*>(__x);
	__matrix = __rowp->alloc(__x);
	__elemp = new MemoryPool<T>(__x * __y);
	for (int i = 0; i < __x; i++) {
		__matrix[i] = __elemp->alloc(__y);
	}
	for (int i = 0; i < __x * __y; i++) {
		__matrix[0][i] = A.__matrix[start_row][i];
	}
}

template<typename T>
inline Matrix<T>* Matrix<T>::Trans()
{	//非就地转置，生成了新的矩阵，需要释放
	auto ans = new Matrix<T>(__y, __x);
	for (int i = 0; i < __x; i++) {
		for (int j = 0; j < __y; j++) {
			ans->__matrix[j][i] = __matrix[i][j];
		}
	}
	return ans;
}

template<typename T>
inline int Matrix<T>::x()
{
	return __x;
}

template<typename T>
inline int Matrix<T>::y()
{
	return __y;
}

template<typename T>
class numcpp {
/*
	Matrix的友元类，实现了大量矩阵的操作
*/
public:
	numcpp(){ srand((unsigned int)time(NULL)); }	//初始化随机种子
	Matrix<T>* dot(Matrix<T>& A, Matrix<T>& B);	//矩阵乘法
	Matrix<T>* exp(Matrix<T>& A);	//矩阵元素exp函数
	Matrix<T>* ones(int x, int y);	//新建一个全为1的矩阵
	Matrix<T>* sigmod(Matrix<T>& A);	//sigmod
	Matrix<T>* dsigmod(Matrix<T>& A);	//sigmod的导数
	void exp_(Matrix<T>& A);	//就地exp
	void ones_(Matrix<T>& A);	//就地全为1
	void random_(Matrix<T>& A);	//随机
	void sigmod_(Matrix<T>& A);	//就地sigmod
	void dsigmod_(Matrix<T>& A);	//就地sigmod求导
	Matrix<T>* sum(Matrix<T>& A, int axis);	//按axis轴求和，结果仍是矩阵
	T sum(Matrix<T>& A);	//全部求和，返回一个数
	Matrix<T>* max(Matrix<T>& A, int axis);	//按axis求最大值，结果仍是矩阵
	Matrix<T>* argmax(Matrix<T>& A, int axis);	//按axis求最大值所在的位置，结果仍是矩阵
};

template<typename T>
inline Matrix<T>* numcpp<T>::dot(Matrix<T>& A, Matrix<T>& B)
{
	if (A.__y != B.__x) throw ERRORS::LENGTH_NOT_MATCH_ERROR;
	auto ans = new Matrix<T>(A.__x, B.__y);
	for (int i = 0; i < A.__x; i++) {
		for (int j = 0; j < B.__y; j++) {
			for (int k = 0; k < A.__y; k++) {
				ans->__matrix[i][j] += A.__matrix[i][k] * B.__matrix[k][j];	//一个非常慢的点乘方案，局部性差，cache利用率低
			}
		}
	}
	return ans;
}

template<typename T>
inline Matrix<T>* numcpp<T>::exp(Matrix<T>& A)
{
	auto ans = new Matrix<T>(A.__x, A.__y);
	for (int i = 0; i < A.__x * A.__y; i++)
	{
		ans->__matrix[0][i] = std::exp(A.__matrix[0][i]);
	}
	return ans;
}

template<typename T>
inline Matrix<T>* numcpp<T>::ones(int x, int y)
{
	auto ans = new Matrix<T>(x, y);
	for (int i = 0; i < x * y; i++)
	{
		ans->__matrix[0][i] = 1;
	}
	return ans;
}

template<typename T>
inline Matrix<T>* numcpp<T>::sigmod(Matrix<T>& A)
{
	auto ans = new Matrix<T>(A.__x, A.__y);
	for (int i = 0; i < A.__x * A.__y; i++)
	{
		ans->__matrix[0][i] = 1.0 / (1.0 + std::exp(A.__matrix[0][i]));
	}
	return ans;
}

template<typename T>
inline Matrix<T>* numcpp<T>::dsigmod(Matrix<T>& A)
{
	auto ans = new Matrix<T>(A.__x, A.__y);
	for (int i = 0; i < A.__x * A.__y; i++)
	{
		ans->__matrix[0][i] = (1.0 / (1.0 + std::exp(A.__matrix[0][i])))
			* (1.0 - (1.0 / (1.0 + std::exp(A.__matrix[0][i]))));
	}
	return ans;
}

template<typename T>
inline void numcpp<T>::exp_(Matrix<T>& A)
{
	for (int i = 0; i < A.__x * A.__y; i++)
	{
		A.__matrix[0][i] = std::exp(A.__matrix[0][i]);
	}
}

template<typename T>
inline void numcpp<T>::ones_(Matrix<T>& A)
{
	for (int i = 0; i < A.__x * A.__y; i++) {
		A.__matrix[0][i] = 1;
	}
}

template<typename T>
inline void numcpp<T>::random_(Matrix<T>& A)
{
	double k = (double)RAND_MAX;
	for (int i = 0; i < A.__x * A.__y; i++) {
		A.__matrix[0][i] = rand() / k - 0.5;	//[-0.5,0.5]的随机分布
	}
}

template<typename T>
inline void numcpp<T>::sigmod_(Matrix<T>& A)
{
	for (int i = 0; i < A.__x * A.__y; i++)
	{
		A.__matrix[0][i] = 1.0 / (1.0 + std::exp(A.__matrix[0][i]));
	}
}

template<typename T>
inline void numcpp<T>::dsigmod_(Matrix<T>& A)
{
	for (int i = 0; i < A.__x * A.__y; i++)
	{
		A.__matrix[0][i] = (1.0 / (1.0 + std::exp(A.__matrix[0][i])))
			* (1.0 - (1.0 / (1.0 + std::exp(A.__matrix[0][i]))));
	}
}

template<typename T>
inline Matrix<T>* numcpp<T>::sum(Matrix<T>& A, int axis)
{
	if (0 == axis) {
		auto ans = new Matrix<T>(A.__x, 1);
		for (int i = 0; i < A.__x; i++)
		{
			for (int j = 0; j < A.__y; j++)
				ans->__matrix[i][0] += A.__matrix[i][j];
		}
		return ans;
	}
	else if(1 == axis){
		auto ans = new Matrix<T>(1, A.__y);
		for (int i = 0; i < A.__y; i++)
		{
			for (int j = 0; j < A.__x; j++) {
				ans->__matrix[0][i] += A.__matrix[j][i];
			}
		}
		return ans;
	}
	else {
		throw ERRORS::OUT_OF_RANGE_ERROR;	//轴数错误
	}
}

template<typename T>
inline T numcpp<T>::sum(Matrix<T>& A)
{
	T s;
	for (int i = 0; i < A.__x * A.__y; i++) {
		s += A.__matrix[0][i];
	}
	return s;
}

template<typename T>
inline Matrix<T>* numcpp<T>::max(Matrix<T>& A, int axis)
{
	if (0 == axis) {
		auto ans = new Matrix<T>(A.__x, 1);
		for (int i = 0; i < A.__x; i++)
		{
			ans->__matrix[i][0] = A.__matrix[i][0];
			for (int j = 0; j < A.__y; j++)
				if (ans->__matrix[i][0] < A.__matrix[i][j])
					ans->__matrix[i][0] = A.__matrix[i][j];
		}
		return ans;
	}
	else if(1 == axis){
		auto ans = new Matrix<T>(1, A.__y);
		for (int i = 0; i < A.__y; i++)
		{
			ans->__matrix[0][i] = A.__matrix[0][i];
			for (int j = 0; j < A.__x; j++) {
				if (ans->__matrix[0][i] < A.__matrix[j][i])
					ans->__matrix[0][i] = A.__matrix[j][i];
			}
		}
		return ans;
	}
	else {
		throw ERRORS::OUT_OF_RANGE_ERROR;
	}
}

template<typename T>
inline Matrix<T>* numcpp<T>::argmax(Matrix<T>& A, int axis)
{
	T temp;
	if (0 == axis) {
		auto ans = new Matrix<T>(A.__x, 1);
		for (int i = 0; i < A.__x; i++)
		{
			ans->__matrix[i][0] = 0;
			temp = A.__matrix[i][0];
			for (int j = 0; j < A.__y; j++) {
				if (temp < A.__matrix[i][j]) {
					temp = A.__matrix[i][j];
					ans->__matrix[i][0] = j;
				}
			}
		}
		return ans;
	}
	else if (1 == axis){
		auto ans = new Matrix<T>(1, A.__y);
		for (int i = 0; i < A.__y; i++)
		{
			ans->__matrix[0][i] = 0;
			temp = A.__matrix[0][i];
			for (int j = 0; j < A.__x; j++) {
				if (temp < A.__matrix[j][i]) {
					temp = A.__matrix[j][i];
					ans->__matrix[0][i] = j;
				}
			}
		}
		return ans;
	}
	else {
		throw ERRORS::OUT_OF_RANGE_ERROR;
	}
}

#endif // __UTIL_H__