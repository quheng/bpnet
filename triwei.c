/*
使用三维向量 表示书体
000 为草书
001 为楷书 
010 为隶书 
011 为行书
100	为篆书
*/ 


#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <io.h>


#define In 216
#define Out 3
#define Neuron 45
#define TrainC 20000
#define A  0.2
#define B  0.4
#define a  0.2
#define b  0.3

double d_in[20000][In];
int d_out[20000][Out],OutputData[Out]; 
double w[Neuron][In],o[Neuron],v[Out][Neuron];
double dv[Out][Neuron],dw[Neuron][In];
int e;

int Data;

void readData(){													//读取训练样本 
	struct _finddata_t files;
 	int File_Handle;
	File_Handle = _findfirst("C:/Users/quheng/Desktop/bpnet/train/*.txt",&files);
	if(File_Handle==-1) {
		 printf("文件目录不存在\n");
		 system("pause");
  		 return 0; 
  	}
  	
 	do{
		char filepath[10000]="C:/Users/quheng/Desktop/bpnet/train/";
		printf("读取样本%d中\n",Data);
		strcat(filepath,files.name);
		FILE *fp ;
		if	(files.name[0]=='c') {d_out[Data][0] = 0;d_out[Data][1] = 0;d_out[Data][2] = 0;} 
		if	(files.name[0]=='k') {d_out[Data][0] = 0;d_out[Data][1] = 0;d_out[Data][2] = 1;} 
		if	(files.name[0]=='l') {d_out[Data][0] = 0;d_out[Data][1] = 1;d_out[Data][2] = 0;} 
		if	(files.name[0]=='x') {d_out[Data][0] = 0;d_out[Data][1] = 1;d_out[Data][2] = 1;} 
		if	(files.name[0]=='z') {d_out[Data][0] = 1;d_out[Data][1] = 0;d_out[Data][2] = 0;} 
		
		fp=fopen(filepath,"r");
		int i;
		for (i=0;i<216;i++) {
			fscanf(fp,"%lf",&d_in[Data][i]);
 		}
		if (fp==NULL) printf("%s\n",files.name); 

 		Data++;
 		fclose(fp);
 	}while(0==_findnext(File_Handle,&files));		
	 _findclose(File_Handle);
	 printf("样本读取完毕\n"); 
}

void initBPNework(){													//初始化神经网络 
	int i,j;
	for (i = 0; i < Neuron; ++i)	
		for (j = 0; j < In; ++j){	
			w[i][j]=rand()*2.0/RAND_MAX-1;
			dw[i][j]=0;
		}

		for (i = 0; i < Neuron; ++i)	
			for (j = 0; j < Out; ++j){
				v[j][i]=rand()*2.0/RAND_MAX-1;
				dv[j][i]=0;
			}
}

void comput(int var){

	int i,j;
	double sum,y;
	memset(OutputData,0,sizeof(OutputData));
	for (i = 0; i < Neuron; ++i){
		sum=0;
		for (j = 0; j < In; ++j)
			sum+=w[i][j]*d_in[var][j];
		o[i]=1/(1+exp(-1*sum));
	}

	for (i = 0; i < Out; ++i){
		sum=0;
		for (j = 0; j < Neuron; ++j)
			sum+=v[i][j]*o[j];
		if (sum>0.5) OutputData[i]=1; 
	}	
}

void backUpdate(int var)
{
	int i,j;
	double t;
	for (i = 0; i < Neuron; ++i)
	{
		t=0;
		for (j = 0; j < Out; ++j){
			t+=(OutputData[j]-d_out[var][j])*v[j][i];
			dv[j][i]=A*dv[j][i]+B*(OutputData[j]-d_out[var][j])*o[i];
			v[j][i]-=dv[j][i];
		}

		for (j = 0; j < In; ++j){
			dw[i][j]=a*dw[i][j]+b*t*o[i]*(1-o[i])*d_in[var][j];
			w[i][j]-=dw[i][j];
		}
	}
}

void  trainNetwork(){												//训练神经网络 
	printf("开始训练神经网络\n");
	int i,c=0,j;
	int check;
	do{
		e=0;
		for (i = 0; i < Data; ++i){
			comput(i);
			check=1;
			for (j = 0; j < Out; ++j)
				if (OutputData[j]!=d_out[i][j]) check=0;
			if (check!=1) e++;
			backUpdate(i);
//			printf("%d  %d   %d\n",OutputData[0],OutputData[1],OutputData[2]);
//			printf("%d  %d   %d\n\n\n",d_out[i][0],d_out[i][1],d_out[i][2]);
		}
		c++;
		printf("第%d次训练网络，误差精度为：%f\n",c,1.0*e/Data);
	}while(c<TrainC && 1.0*e/Data>0.001);
	printf("训练神经网络完毕，读取测试样本\n");
}

void testNetwork(){
	struct _finddata_t files;
 	int File_Handle;
	File_Handle = _findfirst("C:/Users/quheng/Desktop/bpnet/test/*.txt",&files);
	if(File_Handle==-1) {
		 printf("文件目录不存在\n");
		 system("pause");
  		 return 0; 
  	}	
  	Data=0; 
 	do{
		char filepath[10000]="C:/Users/quheng/Desktop/bpnet/test/";
		strcat(filepath,files.name);
		FILE *fp ;
		if	(files.name[0]=='c') {d_out[Data][0] = 0;d_out[Data][1] = 0;d_out[Data][2] = 0;} 
		if	(files.name[0]=='k') {d_out[Data][0] = 0;d_out[Data][1] = 0;d_out[Data][2] = 1;} 
		if	(files.name[0]=='l') {d_out[Data][0] = 0;d_out[Data][1] = 1;d_out[Data][2] = 0;} 
		if	(files.name[0]=='x') {d_out[Data][0] = 0;d_out[Data][1] = 1;d_out[Data][2] = 1;} 
		if	(files.name[0]=='z') {d_out[Data][0] = 1;d_out[Data][1] = 0;d_out[Data][2] = 0;} 
		
		fp=fopen(filepath,"r");
		int i;
		for (i=0;i<216;i++)	fscanf(fp,"%lf",&d_in[Data][i]);
 		Data++;
		fclose(fp);
 	}while(0==_findnext(File_Handle,&files));	
	 Data--;	
	 _findclose(File_Handle);
	 printf("读取%d个测试样本\n",Data); 
	 int right=0,i,j,check;
	 for (i=0;i<Data;i++) {
	 	comput(i);
	 	check=1;
		 	 for (j = 0; j < Out; ++j){
				if (OutputData[j]!=d_out[i][j]) check=0;
			 	printf("%d  ",OutputData[j]);
			 }
			 printf("\n"); 
		if (check!=1) e++;	
	 }
	 printf("测试完成，共测试%d个样本，错误%d个，错误率%f。\n",Data,e,1.0*e/Data);
}


int  main(int argc, char const *argv[])
{
	readData();
	initBPNework();
	trainNetwork();
	testNetwork();
	system("pause"); 
	return 0;
}
