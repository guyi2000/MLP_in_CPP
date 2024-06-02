#include <iostream>
#include <ctime>
#include <fstream>
#include "getopt.h"
#include "cppMLP.h"

static const struct option long_option[] = {
	{(char*)"train_dimensions", required_argument, NULL, '1'},
	{(char*)"output_dimensions", required_argument, NULL, '2'},
	{(char*)"save_file", required_argument, NULL, '3'},
	{(char*)"train_file", required_argument, NULL, '4'},
	{(char*)"predict_file", required_argument, NULL, '5'},
	{(char*)"hidden_layers", required_argument, NULL, '6'},
	{(char*)"load_file", required_argument, NULL, '7'},
	{(char*)"train_rows", required_argument, NULL, '8'},
	{(char*)"predict_rows", required_argument, NULL, '9'},
	{(char*)"learning_rate", required_argument, NULL, 'r'},
	{(char*)"iterations", required_argument, NULL, 'i'},
	{(char*)"train", no_argument, NULL, 't'},
	{(char*)"predict", no_argument, NULL, 'p'},
	{(char*)"save", no_argument, NULL, 's'},
	{(char*)"load", no_argument, NULL,'l'},
	{(char*)"help", no_argument, NULL, '?'},
	{(char*)"version", no_argument, NULL, 'V'},
	{0, 0, NULL, 0}
};	//options

void printUsage() {
/*
	输出帮助文档
*/
	cout << "Usage: ./CPlusPlusDeepLearning.exe [-tpsl?hV] [-i <num>] ..." << endl;
	cout << "Options" << endl;
	cout << "  -h -?      Print this help message." << endl;
	cout << "  -V         Print the verison." << endl;
	cout << "  -t         train the model." << endl;
	cout << "  -p         predict with the model." << endl;
	cout << "  -s         save the model." << endl;
	cout << "  -l         load the model." << endl;
	cout << "  -i <num>   define the iterations." << endl;
	cout << "  -r <num>   define the learning_rate." << endl;
	cout << endl;
	cout << "  --help                     Print this help message." << endl;
	cout << "  --version                  Print the verison." << endl;
	cout << "  --train                    train the model." << endl;
	cout << "  --predict                  predict with the model." << endl;
	cout << "  --save                     save the model." << endl;
	cout << "  --load                     load the model." << endl;
	cout << "  --iterations <num>         define the iterations." << endl;
	cout << "  --learning_rate <num>      define the learning_rate." << endl;
	cout << "  --save_file <file>         define save path." << endl;
	cout << "  --load_file <file>         define load path." << endl;
	cout << "  --train_file <file>        define train path." << endl;
	cout << "  --predict_file <file>      define predict path." << endl;
	cout << "  --train_dimensions <num>   define train dimensions." << endl;
	cout << "  --hidden_layers <num>      define hidden layers" << endl;
	cout << "  --output_dimensions <num>  define output dimensions." << endl;
	cout << "  --train_rows <num>         define train rows." << endl;
	cout << "  --predict_rows <num>       define predict rows." << endl;
	cout << endl;
	cout << "Examples:" << endl;
	cout << "  cmd>  CPlusPlusDeepLearning.exe -t --hidden_layers 100 -s" << endl;
	cout << "  cmd>  CPlusPlusDeepLearning.exe -l -p" << endl;
}

int main(int argc, char* const argv[]) {

	int train_dimensions = 784;	//输入维数，__L1
	int hidden_layers = 100;	//隐藏层维数，__L2
	int output_dimensions = 10;	//输出维数，__L3
	int train_rows = 60000;		//训练数据量
	int predict_rows = 10000;	//测试数据量
	double learning_rate = 0.1;	//学习率
	int iterations = 2;			//迭代次数

	string train_file = "train.data";	//训练数据来源
	string predict_file = "test.data";	//测试数据来源
	string save_file = "param.data";	//模型参数保存路径
	string load_file = "param.data";	//模型参数加载路径

	bool isTrain = 0, isPredict = 0, isSave = 0, isLoad = 0;
	string versions = "v1.0.0";			//基本参数

	ifstream sourcefile;	//用于检测路径是否存在

	int opt;
	int options_index = 0;	//getopt函数参数
	while ((opt = getopt_long(argc, argv, "1:2:3:4:5:6:7:8:9:r:i:tpsl?hV", long_option, &options_index)) != -1) {
		switch (opt) {
		case '1':
			train_dimensions = atoi(optarg);
			if (train_dimensions <= 0) {
				cout << "Wrong range!" << endl;
				exit(-1);
			}
			break;
		case '2':
			output_dimensions = atoi(optarg);
			if (output_dimensions <= 0) {
				cout << "Wrong range!" << endl;
				exit(-1);
			}
			break;
		case '3':
			save_file = optarg;	//此处未检查，因为模型本身会检查
			break;
		case '4':
			train_file = optarg;
			sourcefile.open(train_file);
			if (!sourcefile.is_open()) {
				cout << "Wrong Path!" << endl;
				exit(-1);
			}
			sourcefile.close();
			break;
		case '5':
			predict_file = optarg;
			sourcefile.open(predict_file);
			if (!sourcefile.is_open()) {
				cout << "Wrong Path!" << endl;
				exit(-1);
			}
			sourcefile.close();
			break;
		case '6':
			hidden_layers = atoi(optarg);
			if (hidden_layers <= 0) {
				cout << "Wrong range!" << endl;
				exit(-1);
			}
			break;
		case '7':
			load_file = optarg;
			sourcefile.open(load_file);
			if (!sourcefile.is_open()) {
				cout << "Wrong Path!" << endl;
				exit(-1);
			}
			sourcefile.close();
			break;
		case '8':
			train_rows = atoi(optarg);
			if (train_rows <= 0) {
				cout << "Wrong range!" << endl;
				exit(-1);
			}
			break;
		case '9':
			predict_rows = atoi(optarg);
			if (predict_rows <= 0) {
				cout << "Wrong range!" << endl;
				exit(-1);
			}
			break;
		case 'r':
			learning_rate = atof(optarg);
			if (learning_rate <= 0) {
				cout << "Wrong range!" << endl;
				exit(-1);
			}
			break;
		case 'i':
			iterations = atoi(optarg);
			if (iterations <= 0) {
				cout << "Wrong range!" << endl;
				exit(-1);
			}
			break;
		case 't':
			isTrain = true;
			break;
		case 'p':
			isPredict = true;
			break;
		case 's':
			isSave = true;
			break;
		case 'l':
			isLoad = true;
			break;
		case 'h':
		case '?':
			printUsage();
			return 0;
			break;
		case 'V':
			cout << versions << endl;
			return 0;
			break;
		}
	}

	MLPclassifier solve(train_dimensions, 
		hidden_layers, output_dimensions, 
		learning_rate, iterations, isPredict);	//构造分类机

	solve.init();	//初始化

	cout << "==================== parameters ====================" << endl;
	cout << "train_dimensions:  " << train_dimensions << endl;
	cout << "hidden_layers:     " << hidden_layers << endl;
	cout << "output_dimensions: " << output_dimensions << endl;
	cout << "learning_rate:     " << learning_rate << endl;
	cout << "iterations:        " << iterations << endl;
	cout << "Tasks:\t";
	if (isLoad) cout << "Load\t";
	if (isTrain) cout << "Train\t";
	if (isPredict) cout << "Predict\t";
	if (isSave) cout << "Save\t";
	cout << endl;
	cout << "Total Parameters:  " << train_dimensions * hidden_layers +
		hidden_layers * output_dimensions + hidden_layers + output_dimensions << endl;
	cout << "==================== parameters ====================" << endl;

	if (isLoad) {
		cout << "==================== Load ====================" << endl;
		cout << "Loading parameters from " << load_file << endl;
		solve.load(load_file);	//加载
		cout << "==================== Load ====================" << endl;
	}

	if (isTrain) {
		cout << "==================== Train ====================" << endl;
		auto trainingData = new Matrix<double>(train_rows, train_dimensions);
		auto trainingLabel = new Matrix<double>(train_rows, 1);

		cout << "Loading train data from " << train_file << endl;
		sourcefile.open(train_file);
		int temp;
		for (int i = 0; i < train_rows; i++) {
			for (int j = 0; j < train_dimensions; j++) {
				sourcefile >> temp;
				(*trainingData)[i][j] = temp / 255.0;
			}
			sourcefile >> temp;
			(*trainingLabel)[i][0] = temp;
		}
		sourcefile.close();	//读入训练数据

		solve.train(*trainingData, *trainingLabel);	//训练，可能在load基础上训练

		delete trainingData;
		delete trainingLabel;	//清理内存
		cout << "==================== Train ====================" << endl;
	}

	if (isPredict) {
		cout << "==================== Predict ====================" << endl;
		auto testData = new Matrix<double>(predict_rows, train_dimensions);
		auto testLabel = new Matrix<double>(predict_rows, 1);

		cout << "Loading test data from " << predict_file << endl;
		sourcefile.open(predict_file);
		int temp;
		for (int i = 0; i < predict_rows; i++) {
			for (int j = 0; j < train_dimensions; j++) {
				sourcefile >> temp;
				(*testData)[i][j] = temp / 255.0;
			}
			sourcefile >> temp;
			(*testLabel)[i][0] = temp;
		}
		sourcefile.close();

		auto k = solve.predict(*testData);
		cout << "正确率：" << 100.0 * solve.accurancy(*k, *testLabel) / predict_rows << "%" << endl;

		delete testData;
		delete testLabel;	
		delete k;	//清理内存
		cout << "==================== Predict ====================" << endl;
	}

	if (isSave) {
		cout << "==================== Save ====================" << endl;
		cout << "Saving parameters to " << save_file << endl;
		solve.save(save_file);	//保存
		cout << "==================== Save ====================" << endl;
	}

	return 0;
}