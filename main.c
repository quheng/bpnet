/*
字体编号： 
 0:草书 
 1:楷书  
 2:隶书 
 3:行书 
 4:篆书 
*/ 


#include  "stdio.h"
#include  "io.h"
#include  "stdlib.h"
#include  "string.h"
#include  "math.h" 
   
#define IN 216			//输入向量维数
#define OUT 5			//输出向量维数
#define NEURON 100		//神经元数量 
#define TRAINC 5500		//训练次数 

#define A  0.2					   //学习误差 
#define B  0.4
#define a  0.2
#define b  0.3

double matrix [20000][250];		  //存储训练集数据
int    font[20000][10];		  	  //储存每一个训练样本的字体代号   分量为1 的  对应字体编码 
double w[250][1250];			  //训练集到隐层的权值；
double dw[250][1250];			  //每次训练 w 的修正量数组 
double v[1250][10];				  //隐函数层到输出层的权值
double dv[1250][10];			  //每次训练 v 的修正量数组 
double o[2000];					  //神经元通过激活函数对外的输出

int OutputData[10]; 			 //输出向量 分量为一代表是这种字体；
int number=0;					 //样本数量  


int getdata(){                                       //读取训练集 
	struct _finddata_t files;
 	int File_Handle;
	File_Handle = _findfirst("C:/Users/quheng/Desktop/main/test/*.txt",&files);
	memset(font,0,sizeof(font));
	if(File_Handle==-1) {
		 printf("文件目录不存在\n");
		 system("pause");
  		 return 0; 
  	}
  	
 	do{
		char filepath[10000]="C:/Users/quheng/Desktop/main/test/";
		printf("读取样本%d中\n",number);
		strcat(filepath,files.name);
		FILE *fp ;
		if	(files.name[0]=='c') font[number][0] = 1;
		if	(files.name[0]=='k') font[number][1] = 1;
		if	(files.name[0]=='l') font[number][2] = 1;
		if	(files.name[0]=='x') font[number][3] = 1;
		if	(files.name[0]=='z') font[number][4] = 1;
		
		fp=fopen(filepath,"r");
		int i;
		for (i=0;i<216;i++) {
			fscanf(fp,"%lf",&matrix[number][i]);
 		}
 		number++;
 		fclose(fp);
 	}while(0==_findnext(File_Handle,&files));
 		
	 _findclose(File_Handle);
	 printf("样本读取完毕\n"); 
}

int initialize(){								//初始化网络权值 
	int i,j;
	 printf("初始化神经网络\n"); 
	for (i = 0; i < IN; i++)    
        for (j = 0; j < NEURON; j++){    
            w[i][j]=(rand()*2.0/RAND_MAX-1)/2;
            dw[i][j]=0;
        }
    for (i = 0; i < NEURON; i++)    
         for (j = 0; j < OUT; j++){
             v[i][j]=(rand()*2.0/RAND_MAX-1)/2;
              dv[i][j]=0;
         }
     printf("初始化完毕\n"); 
} 

void calculate(int vector){
	int i,j;
    double sum,y;                       //隐藏层神经元的输出 
        
	// 输入层到隐藏层
	  
    for (i = 0; i < NEURON ; i++ ){
        sum=0;
        for (j = 0; j < IN; j++)
            sum+=w[i][j]*matrix[vector][j];
        o[i]=1/(1+ exp(-1*sum));
    }

/*  隐藏层到输出层输出 */

    for (i = 0; i < OUT; i++ ){
        sum=0;
        for (j = 0; j < NEURON; ++j)
            sum+=v[i][j]*o[j];
		if ((1/(1+exp(-1*sum)))>0.5) OutputData[i]=1;
		else OutputData[i]=0;
    }    
	
}	


void backUpdate(int vector)
{
    int i,j;
    double t;
    for (i = 0; i < NEURON; i++)
    {
        t=0;
        for (j = 0; j < OUT; j++){
            t+=(OutputData[j]-font[vector][j])*v[j][i];
            dv[j][i]=A*dv[j][i]+B*(OutputData[j]-font[vector][j])*o[i];
            v[j][i]-=dv[j][i];
        }

        for (j = 0; j < IN; j++){
            dw[i][j]=a*dw[i][j]+b*t*o[i]*(1-o[i])*matrix[vector][j];
            w[i][j]-=dw[i][j];
        }
    }
}
		

int training(){									//训练神经网络 
	int i,j,times=0;								//times为训练次数 
    int e; 									//训练误差 
	do{
		printf("第  %d  次训练神经网络\n",times+1);
        e=0;
        for (i = 0; i < number; i++){
        	calculate(i); 						//计算各层输出 
        	for (j=0;j<5;j++) {
				if (OutputData[j]!=font[i][j])  e++;
			}
			backUpdate(i);
		}
        times++;
    }while(times<TRAINC && 1.0*e/number>0.01);		//反复训练神经网络直到达到训练次数 或者 符合误差精度 	
    printf("训练完毕\n");
}

int output(){										//将训练后的权值保存在weight.txt文件里 

	FILE *fp;
	 printf("开始输出权值\n");
	fp = fopen("weight.txt","w");
	int i,j;
	for (i=0;i<IN;i++)
		for (j=0;j<NEURON;j++) fprintf (fp,"%d ",w[i][j]);
	for (i=0;i<NEURON;i++)
		for (j=0;j<OUT;j++) fprintf (fp,"%d ",v[i][j]);
	printf("输出完毕\n") ;
} 

int main()
{
	getdata();									//读取训练集 
	initialize(); 								//初始化 
	training();									//训练 
	output();									//输出权值						
 	system("pause");
	return 0;
}
