/*
	cppDL.h classifier抽象类，用于classifier接口规范。
*/
#pragma once
#ifndef __cppDL_H__
#define __cppDL_H__

#include <cstring>
using namespace std;
#include "utils.h"

class classifier {
/*
	classifier接口，只要实现分类器，就必须实现以下五个函数，包含基本的初始化，训练，预测，保存，读取
便于以后增加新的分类器的同时，保证其兼容性。
*/
public:
	virtual void init() = 0;	//初始化
	virtual void train(Matrix<double>&, Matrix<double>&) = 0;	//训练
	virtual Matrix<double>* predict(Matrix<double>&) = 0;	//预测
	virtual void save(const string path) = 0;	//保存
	virtual void load(const string path) = 0;	//读取
};

#endif