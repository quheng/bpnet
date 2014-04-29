/*
�����ţ� 
 0:���� 
 1:����  
 2:���� 
 3:���� 
 4:׭�� 
*/ 


#include  "stdio.h"
#include  "io.h"
#include  "stdlib.h"
#include  "string.h"
#include  "math.h" 
   
#define IN 216			//��������ά��
#define OUT 5			//�������ά��
#define NEURON 100		//��Ԫ���� 
#define TRAINC 5500		//ѵ������ 

#define A  0.2					   //ѧϰ��� 
#define B  0.4
#define a  0.2
#define b  0.3

double matrix [20000][250];		  //�洢ѵ��������
int    font[20000][10];		  	  //����ÿһ��ѵ���������������   ����Ϊ1 ��  ��Ӧ������� 
double w[250][1250];			  //ѵ�����������Ȩֵ��
double dw[250][1250];			  //ÿ��ѵ�� w ������������ 
double v[1250][10];				  //�������㵽������Ȩֵ
double dv[1250][10];			  //ÿ��ѵ�� v ������������ 
double o[2000];					  //��Ԫͨ���������������

int OutputData[10]; 			 //������� ����Ϊһ�������������壻
int number=0;					 //��������  


int getdata(){                                       //��ȡѵ���� 
	struct _finddata_t files;
 	int File_Handle;
	File_Handle = _findfirst("C:/Users/quheng/Desktop/main/test/*.txt",&files);
	memset(font,0,sizeof(font));
	if(File_Handle==-1) {
		 printf("�ļ�Ŀ¼������\n");
		 system("pause");
  		 return 0; 
  	}
  	
 	do{
		char filepath[10000]="C:/Users/quheng/Desktop/main/test/";
		printf("��ȡ����%d��\n",number);
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
	 printf("������ȡ���\n"); 
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
		

int training(){									//ѵ�������� 
	int i,j,times=0;								//timesΪѵ������ 
    int e; 									//ѵ����� 
	do{
		printf("��  %d  ��ѵ��������\n",times+1);
        e=0;
        for (i = 0; i < number; i++){
        	calculate(i); 						//���������� 
        	for (j=0;j<5;j++) {
				if (OutputData[j]!=font[i][j])  e++;
			}
			backUpdate(i);
		}
        times++;
    }while(times<TRAINC && 1.0*e/number>0.01);		//����ѵ��������ֱ���ﵽѵ������ ���� �������� 	
    printf("ѵ�����\n");
}

int output(){										//��ѵ�����Ȩֵ������weight.txt�ļ��� 

	FILE *fp;
	 printf("��ʼ���Ȩֵ\n");
	fp = fopen("weight.txt","w");
	int i,j;
	for (i=0;i<IN;i++)
		for (j=0;j<NEURON;j++) fprintf (fp,"%d ",w[i][j]);
	for (i=0;i<NEURON;i++)
		for (j=0;j<OUT;j++) fprintf (fp,"%d ",v[i][j]);
	printf("������\n") ;
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
