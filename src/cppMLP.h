/*
	cppMLP.h MLP实现，继承自classifier
*/
#pragma once

#ifndef __cppMLP_H__
#define __cppMLP_H__

#include <iostream>
#include <cstring>
#include <fstream>
using namespace std;
#include "macro.h"
#include "utils.h"
#include "cppDL.h"

class MLPclassifier : public classifier {
private:
	int __L1;	//第一层维数
	int __L2;	//隐藏层神经元个数
	int __L3;	//输出层维数
	int __iteration;	//迭代次数
	int batchSize = DEFUALT_BATCHSIZE;	//batchSize
	bool isPredict;
	double __learningRate;	//学习率
	Matrix<double>* Weight_12;	//权重1->2
	Matrix<double>* Bias_12;	//偏置1->2
	Matrix<double>* Weight_23;	//权重2->3
	Matrix<double>* Bias_23;	//偏置2->3
	numcpp<double> nc;	//矩阵操作用
public:
	MLPclassifier(int L1,int L2,int L3,double learningRate, int iteration, bool isPredict):
		__L1(L1), __L2(L2), __L3(L3), 
		__iteration(iteration), __learningRate(learningRate), 
		isPredict(isPredict),Weight_12(nullptr),
		Weight_23(nullptr), Bias_12(nullptr), Bias_23(nullptr){}
	virtual ~MLPclassifier() {
		if (Weight_12 != nullptr)
			delete Weight_12;
		if (Weight_23 != nullptr)
			delete Weight_23;
		if (Bias_12 != nullptr)
			delete Bias_12;
		if (Bias_23 != nullptr)
			delete Bias_23;
		Weight_12 = nullptr;
		Weight_23 = nullptr;
		Bias_12 = nullptr;
		Bias_23 = nullptr;
	}
	void init() final;
	void train(Matrix<double>&, Matrix<double>&) final;
	Matrix<double>* predict(Matrix<double>&) final;
	void save(const string path = "param.data") final;
	void load(const string path = "param.data") final;
	Matrix<double>* FP(Matrix<double>& A, int layer = 1);	//前向传播
	void BP(Matrix<double>& A, Matrix<double>& B);	//后向传播
	int accurancy(Matrix<double>& A, Matrix<double>& B);	//准确率计算
};

void MLPclassifier::init() {
	Weight_12 = new Matrix<double>(__L1, __L2);
	Weight_23 = new Matrix<double>(__L2, __L3);
	Bias_12 = new Matrix<double>(__L2, 1);
	Bias_23 = new Matrix<double>(__L3, 1);

	nc.random_(*Weight_12);
	nc.random_(*Weight_23);
	nc.random_(*Bias_12);
	nc.random_(*Bias_23);
	cout << "Initialization Done!" << endl;
}

void MLPclassifier::train(Matrix<double>& trainingData, Matrix<double>& trainingLabel) {
	cout << "Training..." << endl;

	int batchNum = trainingData.x() / batchSize;

	auto batches = new Matrix<double>*[batchNum];
	auto batchLabels = new Matrix<double>*[batchNum];

	for (int i = 0; i < batchNum; i++) {
		batches[i] = new Matrix<double>(batchSize, __L1);
		for (int j = 0; j < batchSize; j++) {
			for (int k = 0; k < __L1; k++) {
				(*batches[i])[j][k] = trainingData[i * batchSize + j][k];
			}
		}
		batchLabels[i] = new Matrix<double>(batchSize, __L3);
		for (int j = 0; j < batchSize; j++) {
			int index = (int)trainingLabel[batchSize * i + j][0];
			(*(batchLabels[i]))[j][index] = 1;
		}
	}	//划分batch进行训练，注意，此处没有shuffle操作，尚不够科学

	for (int i = 0; i < __iteration; i++) {
		cout << "===Iteration: " << i + 1 << " start!===" << endl;
		for (int batch = 0; batch < batchNum; batch++) {
			BP(*(batches[batch]), *(batchLabels[batch]));	//前馈
			if (batch == 0) cout << "Progress: 0%" << endl;
			if (batchNum - batch == batch || batchNum - batch == batch - 1) cout << "Progress: 50%" << endl;
			if (batchNum == batch + 1) cout << "Progress: 100%" << endl;
		}
		if (isPredict) {
			auto k = predict(*(batches[0]));
			cout << "batch 0 准确率：" << accurancy(*k, trainingLabel) << "%" << endl;
			delete k;	//如果需要预测的话
		}
		cout << "===Iteration: " << i + 1 << " finished!===" << endl;
	}
	delete[] batches;
	delete[] batchLabels;
	cout << "Train Successfully" << endl;
}

Matrix<double>* MLPclassifier::predict(Matrix<double>& testData) {
	cout << "Predicting..." << endl;
	auto c = FP(testData, 2);	//前向传播即可
	auto k = nc.argmax(*c, 0);	//求概率最大
	delete c;
	cout << "Predict Successfully!" << endl;
	return k;
}

void MLPclassifier::save(string path)
{
	ofstream sourcefile(path);
	if (!sourcefile.is_open()) throw ERRORS::FILE_NOT_OPEN;
	sourcefile << __L1 << " " << __L2 << " " << __L3 << endl;
	for (int i = 0; i < __L1 * __L2; i++)
		sourcefile << (*Weight_12)[0][i] << " ";
	sourcefile << endl;
	for (int i = 0; i <  __L2; i++)
		sourcefile << (*Bias_12)[0][i] << " ";
	sourcefile << endl;
	for (int i = 0; i < __L2 * __L3; i++)
		sourcefile << (*Weight_23)[0][i] << " ";
	sourcefile << endl;
	for (int i = 0; i < __L3; i++)
		sourcefile << (*Bias_23)[0][i] << " ";
	sourcefile << endl;
	sourcefile.close();
	cout << "Save Successfully!" << endl;
}

void MLPclassifier::load(string path)
{
	ifstream sourcefile(path);
	if (!sourcefile.is_open()) throw ERRORS::FILE_NOT_OPEN;
	int temp;
	sourcefile >> temp;
	if (temp != __L1) { cout << "Load failed" << endl; exit(-1); }
	sourcefile >> temp;
	if (temp != __L2) { cout << "Load failed" << endl; exit(-1); }
	sourcefile >> temp;
	if (temp != __L3) { cout << "Load failed" << endl; exit(-1); }
	for (int i = 0; i < __L1 * __L2; i++)
		sourcefile >> (*Weight_12)[0][i];
	for (int i = 0; i < __L2; i++)
		sourcefile >> (*Bias_12)[0][i];
	for (int i = 0; i < __L2 * __L3; i++)
		sourcefile >> (*Weight_23)[0][i];
	for (int i = 0; i < __L3; i++)
		sourcefile >> (*Bias_23)[0][i];
	sourcefile.close();
	cout << "Load Successfully!" << endl;
}

inline Matrix<double>* MLPclassifier::FP(Matrix<double>& A, int layer)
{
	if (layer == 1) {
		auto temp = nc.ones(1, A.x());
		auto dotProduct = nc.dot(*Bias_12, *temp);
		auto Bias = (*dotProduct).Trans();
		auto dotProduct2 = nc.dot(A, *Weight_12);
		auto sumAns = *dotProduct2 + *Bias;
		nc.sigmod_(*sumAns);
		delete temp; 
		delete dotProduct; 
		delete Bias; 
		delete dotProduct2;
		return sumAns;
	}
	else if (layer == 2) {
		auto k = FP(A);	//代码重用，传播一层
		auto temp = nc.ones(1, A.x());
		auto dotProduct = nc.dot(*Bias_23, *temp);
		auto Bias = (*dotProduct).Trans();
		auto dotProduct2 = nc.dot(*k, *Weight_23);
		auto sumAns = *dotProduct2 + *Bias;
		nc.sigmod_(*sumAns);
		delete k;
		delete temp; 
		delete dotProduct; 
		delete dotProduct2; 
		delete Bias;
		return sumAns;
	}
	else {
		throw ERRORS::OUT_OF_RANGE_ERROR;
	}
}

inline void MLPclassifier::BP(Matrix<double>& A, Matrix<double>& B)
{
/*
	这是神经网络的核心代码，涉及大量矩阵运算
*/
	auto temp = nc.ones(1, A.x());
	auto dotProduct = nc.dot(*Bias_12, *temp);
	auto Bias = (*dotProduct).Trans();
	auto dotProduct2 = nc.dot(A, *Weight_12);
	auto z1 = *dotProduct2 + *Bias;
	auto o1 = nc.sigmod(*z1);

	delete dotProduct;
	delete Bias;
	delete dotProduct2;
	dotProduct = nullptr; Bias = nullptr; dotProduct2 = nullptr;	// 计算z1, o1

	auto dotProduct3 = nc.dot(*Bias_23, *temp);
	auto Bias2 = (*dotProduct3).Trans();
	auto dotProduct4 = nc.dot(*o1, *Weight_23);
	auto z2 = *dotProduct4 + *Bias2;
	auto o2 = nc.sigmod(*z2);

	delete temp; 
	delete dotProduct3; 
	delete dotProduct4; 
	delete Bias2;
	temp = nullptr; dotProduct3 = nullptr; dotProduct4 = nullptr; Bias2 = nullptr;	// 计算z2，o2

	auto delta1 = *o2 - B;
	nc.dsigmod_(*z2);
	auto delta_B23 = (*delta1) * (*z2);

	auto z1T = z1->Trans();
	auto delta_W23 = nc.dot(*z1T, *delta1);

	auto Weight_23T = Weight_23->Trans();
	auto delta2 = nc.dot(*delta1, *Weight_23T);
	delete Weight_23T;
	Weight_23T = nullptr;

	nc.dsigmod_(*z1);
	auto delta_B12 = (*delta2) * (*z1);

	auto Ts = A.Trans();
	auto delta_W12 = nc.dot(*Ts, *delta2);
	delete Ts;
	Ts = nullptr;	//至此，计算完成各层delta_W与delta_B

	auto delta_B12_ = nc.sum(*delta_B12, 1);
	auto delta_B23_ = nc.sum(*delta_B23, 1);
	auto delta_B12_ALL = delta_B12_->Trans();
	auto delta_B23_ALL = delta_B23_->Trans();	//将delta_B汇总

	*delta_B12_ALL *= __learningRate / batchSize;
	*delta_B23_ALL *= __learningRate / batchSize;
	*delta_W12 *= __learningRate / batchSize;
	*delta_W23 *= __learningRate / batchSize;	//乘以学习率

	*Weight_12 -= *delta_W12;
	*Weight_23 -= *delta_W23;
	*Bias_12 -= *delta_B12_ALL;
	*Bias_23 -= *delta_B23_ALL;		// 调节模型参数

	delete z1; 
	delete z2; 
	delete o1; 
	delete o2; 
	delete delta1; 
	delete delta2; 
	delete z1T; 
	delete delta_B12; 
	delete delta_B23; 
	delete delta_W12; 
	delete delta_W23; 
	delete delta_B12_;
	delete delta_B23_;
	delete delta_B12_ALL; 
	delete delta_B23_ALL;
	z1 = nullptr; z2 = nullptr; o1 = nullptr; o2 = nullptr; delta1 = nullptr; delta2 = nullptr;
	z1T = nullptr; delta_B12 = nullptr; delta_B23 = nullptr; delta_W12 = nullptr; delta_W23 = nullptr;
	delta_B12_ALL = nullptr; delta_B23_ALL = nullptr;	//清理内存，防止泄漏
}

int MLPclassifier::accurancy(Matrix<double>& A, Matrix<double>& B)
{
/*
	计算两个矩阵之间相同元素的个数，矩阵A元素个数须小于等于矩阵B元素个数
*/
	int count = 0;
	int num = A.x() * A.y();
	if (num > B.x()* B.y()) throw ERRORS::OUT_OF_RANGE_ERROR;
	for (int i = 0; i < num; i++) {
		if (A[0][i] == B[0][i]) count++;
	}
	return count;
}

#endif