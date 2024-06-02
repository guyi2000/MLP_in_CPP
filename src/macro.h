/*
	macro.h 用于存放宏定义
*/
#pragma once

#ifndef __MACRO_H__
#define __MACRO_H__

#define DEFUALT_MEMORYPOOL_SIZE 1024	//默认momerypool大小
#define RANK int	//秩
#define DEFUALT_VECTOR_SIZE 128	//默认Vector大小
#define DEFUALT_BATCHSIZE 100	//默认batchsize

enum class ERRORS {
	OUT_OF_RANGE_ERROR = 1,
	NOT_ENOUGH_MEMORY_ERROR,
	NOT_ENOUGH_MEMORYPOOL_ERROR,
	LENGTH_NOT_MATCH_ERROR,
	FILE_NOT_OPEN
};

#define WEIGHT 0
#define BIAS 1

#endif // __MACRO_H__