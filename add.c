#include  "stdio.h"
#include  "io.h"
#include  "stdlib.h"
#include  "string.h"
#include  "math.h" 
   
#define IN 2			//输入向量维数
#define OUT 1			//输出向量维数
#define NEURON 60		//神经元数量 
#define TRAINC 20000		//训练次数 

#define A  0.2					   //学习误差 
#define B  0.4
#define a  0.2
#define b  0.3

double matrix [20000][250];		  //存储训练集数据
//int    font[20000][10];		  	  //储存每一个训练样本的字体代号   分量为1 的  对应字体编码 
double w[250][1250];			  //训练集到隐层的权值；
double dw[250][1250];			  //每次训练 w 的修正量数组 
double v[1250][10];				  //隐函数层到输出层的权值
double dv[1250][10];			  //每次训练 v 的修正量数组 
double o[2000];					  //神经元通过激活函数对外的输出

double OutputData[10]; 			 //输出向量 分量为一代表是这种字体；
double out[20000];
int number=0;					 //样本数量  


int getdata(){                                       //读取训练集 
	FILE *fp=NULL;
	fp = fopen("123.txt","r");
	for (number=0;;number++){
		fscanf(fp,"%lf%lf%lf",&matrix[number][0],&matrix[number][1],&out[number]);
		if (matrix[number][0]==0) break;
	} 
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
		OutputData[i]=sum;
    } 
  // printf("ans:  %f\n" ,OutputData[0]);

}	


void backUpdate(int vector)
{
    int i,j;
    double t;
    for (i = 0; i < NEURON; i++)
    {
        t=0;
        for (j = 0; j < OUT; j++){
            t+=(OutputData[j]-out[vector])*v[j][i];
            dv[j][i]=A*dv[j][i]+B*(OutputData[j]-out[vector])*o[i];
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
    double e; 									//训练误差 
	do{
		printf("第  %d  次训练神经网络\n",times+1);
        e=0;
        for (i = 0; i < number; i++){
        	calculate(i); 						//计算各层输出 
        	e+=fabs((OutputData[0]-out[i])/out[i]);
			backUpdate(i);
		}
        times++;
    }while(times<TRAINC && e/number>0.01);		//反复训练神经网络直到达到训练次数 或者 符合误差精度 	
    printf("训练完毕\n");
}

int output(){										//将训练后的权值保存在weight.txt文件里 
	for (number==0;;number++){
		scanf("%lf%lf",&matrix[number][0],&matrix[number][1]);
		calculate(number); 
		printf("answer : %f\n",OutputData[0]);
		if (matrix[number][0]==0) break;
	} 
	
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
