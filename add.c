#include  "stdio.h"
#include  "io.h"
#include  "stdlib.h"
#include  "string.h"
#include  "math.h" 
   
#define IN 2			//��������ά��
#define OUT 1			//�������ά��
#define NEURON 60		//��Ԫ���� 
#define TRAINC 20000		//ѵ������ 

#define A  0.2					   //ѧϰ��� 
#define B  0.4
#define a  0.2
#define b  0.3

double matrix [20000][250];		  //�洢ѵ��������
//int    font[20000][10];		  	  //����ÿһ��ѵ���������������   ����Ϊ1 ��  ��Ӧ������� 
double w[250][1250];			  //ѵ�����������Ȩֵ��
double dw[250][1250];			  //ÿ��ѵ�� w ������������ 
double v[1250][10];				  //�������㵽������Ȩֵ
double dv[1250][10];			  //ÿ��ѵ�� v ������������ 
double o[2000];					  //��Ԫͨ���������������

double OutputData[10]; 			 //������� ����Ϊһ�������������壻
double out[20000];
int number=0;					 //��������  


int getdata(){                                       //��ȡѵ���� 
	FILE *fp=NULL;
	fp = fopen("123.txt","r");
	for (number=0;;number++){
		fscanf(fp,"%lf%lf%lf",&matrix[number][0],&matrix[number][1],&out[number]);
		if (matrix[number][0]==0) break;
	} 
}

int initialize(){								//��ʼ������Ȩֵ
	int i,j;
	printf("��ʼ��������\n"); 
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
     printf("��ʼ�����\n"); 
} 

void calculate(int vector){
	int i,j;
    double sum,y;                       //���ز���Ԫ����� 
        
	// ����㵽���ز�
    for (i = 0; i < NEURON ; i++ ){
        sum=0;
        for (j = 0; j < IN; j++)
            sum+=w[i][j]*matrix[vector][j];
        o[i]=1/(1+ exp(-1*sum));
    }

/*  ���ز㵽�������� */

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
		

int training(){									//ѵ�������� 
	int i,j,times=0;								//timesΪѵ������ 
    double e; 									//ѵ����� 
	do{
		printf("��  %d  ��ѵ��������\n",times+1);
        e=0;
        for (i = 0; i < number; i++){
        	calculate(i); 						//���������� 
        	e+=fabs((OutputData[0]-out[i])/out[i]);
			backUpdate(i);
		}
        times++;
    }while(times<TRAINC && e/number>0.01);		//����ѵ��������ֱ���ﵽѵ������ ���� �������� 	
    printf("ѵ�����\n");
}

int output(){										//��ѵ�����Ȩֵ������weight.txt�ļ��� 
	for (number==0;;number++){
		scanf("%lf%lf",&matrix[number][0],&matrix[number][1]);
		calculate(number); 
		printf("answer : %f\n",OutputData[0]);
		if (matrix[number][0]==0) break;
	} 
	
} 

int main()
{
	getdata();									//��ȡѵ���� 
	initialize(); 								//��ʼ�� 
	training();									//ѵ�� 
	output();									//���Ȩֵ						
 	system("pause");
	return 0;
}
