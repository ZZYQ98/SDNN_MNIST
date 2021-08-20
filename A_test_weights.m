    %SDNN��������  main���� 4.9
%���������������
clc
clear

%����ͼƬ·��
path_list='D:\git_code\F_SDNN\datasets';
spike_times_to_learn=[path_list,'\LearningSet'];
spike_times_to_train=[path_list,'\TrainingSet'];
spike_times_to_test=[path_list,'\TestingSet'];
path_set_weights='D:\git code\bisheSNN\set_weight';
%����洢·��
path_save_weights='D:\git code\bisheSNN\set_weights';
path_features='D:\git code\bisheSNN\features';
%����FLAG
global learn_SDNN
global relearn_SDNN
global set_weights
global save_weights
global save_feature
global DoG
global total_time
%��־λ����
learn_SDNN=0;   %��������ѧϰ��־λ��learn_SDNN����1ʱ���������STDPѧϰ����learn_SDNN����0ʱ�����粻����ѧϰ��ֱ�Ӷ�ȡ�Ѿ��õ���Ȩֵ���ݣ����в���
relearn_SDNN=0;
%���ȿ������Ƿ����ѵ��      
if  learn_SDNN==1                  
    set_weights=0;         %ѵ��ʱ��������Ȩֵ��������Ȩֵ��ʼ��
    save_weights=1;         %����ѵ��������Ȩֵ��Ϣ��·����Ϣ
    save_feature=1;
else                       %��ѵ��ʱ������Ȩֵ��
    set_weights=1;
    save_weights=0;
    save_feature=0;
end

DoG=1;            %����DoG�˲���־λ��DoG����1ʱ������ͼ������˲��õ����巢��ʱ�䣬��DoG����0ʱ�����ļ��ж�ȡ���巢��ʱ�� 

%SDNN������������������������������������������������������������������������
img_size=struct('img_sizeH',160,'img_sizeW',250);%����ͼ���ģ
DoG_params=struct('img_size', img_size, 'DoG_size', 5, 'std1', 1, 'std2', 2);%����DoG����
%�����������
l1=struct('type','input', 'num_filters', 1, 'pad',0, 'H_layer',DoG_params.img_size.img_sizeH,'W_layer', DoG_params.img_size.img_sizeW);
l2=struct('type', 'conv', 'num_filters', 4, 'filter_size', 5, 'th', 6);
l3=struct('type', 'pool', 'num_filters', 4, 'filter_size', 17, 'th', 0., 'stride', 16);
l4=struct('type', 'conv', 'num_filters',10, 'filter_size', 15, 'th', 18);
learnable_layers=[2,4];
network_params={l1,l2,l3,l4};
weight_params=struct('mean',0.8,'std',0.01);%����Ȩֵ��ʼ������ 
max_learn_iter=[0,1000,0,500];
STDP_per_layer=[0,4,0,1];
max_iter=sum(max_learn_iter);
a_minus=[0,0.003,0,0.003];
a_plus=[0,0.005,0,0.005];
offset=[0 5 0 0];
%���STDPȨֵ���¾���
tao_minus=40;
tao_plus=20;
STDP_time=35;%STDP ����ʱ��
deta_STDP_minus=deta_STDP(0.03,STDP_time,tao_minus);  %���ô�����ΪSTDP_time_minus=35��ʱ����Ϊ40
deta_STDP_plus=deta_STDP(0.07,STDP_time,tao_plus);    %���ô�����ΪSTDP_time_plus=30��ʱ����Ϊ20

STDP_params=struct('max_learning_iter',max_learn_iter,'STDP_per_layer',STDP_per_layer,...
                   'max_iter',max_iter,'a_minus',a_minus,'a_plus',a_plus,'offset',offset);
total_time=30;

%ǿ��ѧϰ�����е�STDPѧϰ
retrain_params=struct('a_minus_r', 0.03, 'a_plus_r', 0.05, 'a_minus_p', 0.07, 'a_plus_p', 0.01,...
                     'tao_minus_r',40, 'tao_plus_r',20, 'tao_minus_p', 80, 'tao_plus_p', 10,'retrain_iter',1500);
 svm_train1=zeros(1,20);
 svm_test1=zeros(1,20);
for i=1:20
%-------------------------------------�������������������SDNN_net----------------------------------------------------
%����ṹ��ʼ��
network_struct=init_net_struct(network_params);%���ú���network_struct���������ò�������������г�ʼ��
%network_structΪ����ṹ�壬learnable_layersΪ�ɽ���ѧϰ�ľ�����

layers = init_layers(network_struct);%���ú���init_layers�������в�ĳ�ʼ��

%Ȩֵ�����ʼ��
[weights]= init_weights( weight_params,network_struct);%���ú���init_weight��������������Ȩֵ���󣬲���Ҫ���ʼ��Ȩֵ
%weightsΪ��ʼ����Ȩֵ����W_shapeΪȨֵ����ĳߴ�

%check_dimensions( network_struct,W_shape )%����õ�Ȩֵ����ά��������ά���Ƿ���ͬ�������������򷵻ض�Ӧ�Ĵ���ԭ��

%��������������������������   ��������������ͼ����������������������������������������������������������������������������
% label�У�faceΪ1��motobikeΪ2
if DoG==1
     [spike_times_learn,label_learn]=get_iter_path1(spike_times_to_learn);%��ͼƬ���룬����Ҫ�����˲�
     [spike_times_train,label_train]=get_iter_path1(spike_times_to_train);
     [spike_times_test,label_test]=get_iter_path1(spike_times_to_test);
     [~,num_img_learn]=size(spike_times_learn);
     [~,num_img_train]=size(spike_times_learn);
     [~,num_img_test]=size(spike_times_test);
 else
     spike_times_learn=spike_times_to_learn;       %�������˲���ֱ�ӽ�����ʱ������,��Щ��������ʱ�������Ϊ֮ǰ�õ��ģ��������ļ���
     [~,num_img_learn]=size(spike_times_learn);
     spike_times_train=spike_times_to_train;
     [~,num_img_train]=size(spike_times_train);
     spike_times_test=spike_times_to_test;
     [num_img_,~]=size(spike_times_texst);
end       

% %------------------------------------ѵ���õ����-----------------------------------------------------------------------


%����Ȩֵ���߽���STDPѧϰ
if set_weights==1
   %��ȡ���ļ��ж�ȡȨֵ
   if i==1
       weights_path_list='weights_01.mat';    %Ȩֵ��ȡ·��
   elseif i==2
       weights_path_list='weights_02.mat';
   elseif i==3
       weights_path_list='weights_03.mat';
   elseif i==4
       weights_path_list='weights_04.mat';
   elseif i==5
       weights_path_list='weights_05.mat';   
   elseif i==6
       weights_path_list='weights_06.mat';   
   elseif i==7
       weights_path_list='weights_07.mat';
   elseif i==8
       weights_path_list='weights_08.mat';       
   elseif i==9
       weights_path_list='weights_09.mat';       
   elseif i==10
       weights_path_list='weights_10.mat';       
   elseif i==11
       weights_path_list='weights_11.mat';       
   elseif i==12
       weights_path_list='weights_12.mat';       
   elseif i==13
       weights_path_list='weights_13.mat';       
   elseif i==14
       weights_path_list='weights_14.mat';       
   elseif i==15
       weights_path_list='weights_15.mat';       
   elseif i==16
       weights_path_list='weights_16.mat';       
   elseif i==17
       weights_path_list='weights_17.mat';
   elseif i==18
       weights_path_list='weights_18.mat';       
   elseif i==19
       weights_path_list='weights_19.mat';       
   elseif i==20
       weights_path_list='weights_20.mat';             
   end
   weights =load(weights_path_list);  %���ļ��е�Ȩֵ����ȡ����ֱ�ӽ��и�ֵ
   weights=weights.weights;
else
    weights=train_SDNN(weights,layers,network_struct,spike_times_learn,DoG_params,STDP_params,num_img_learn,learnable_layers,total_time,deta_STDP_minus,deta_STDP_plus);
    save('weights_5_7_train_SDNN.mat','weights');
end   
% %����δ����relearn��Ȩֵ���з���-----------------------------------------------------------------------------------
% [X_learn,T_learn] = get_feature(weights,layers,network_struct,spike_times_learn,num_img_learn,DoG_params,total_time);
% %%��Ӧ��ǩΪlabel_learn
% [learn_label] = check_T(T_learn);
% learning_correct_rate=correct_rate(label_learn,learn_label);
% 
% [X_train,T_train] = get_feature(weights,layers,network_struct,spike_times_train,num_img_train,DoG_params,total_time);
% %��Ӧ��ǩΪlabel_train
% [train_label] = check_T(T_train);
% training_correct_rate=correct_rate(label_train,train_label);
% 
% [X_test,T_test] = get_feature(weights,layers,network_struct,spike_times_test,num_img_test,DoG_params,total_time);
% %��Ӧ��ǩΪlabel_test
% [test_label] = check_T(T_test);
% testing_correct_rate=correct_rate(label_test,test_label);



if relearn_SDNN==1
    weights=retrain_SDNN(weights,layers,network_struct,spike_times_train,DoG_params,STDP_params,num_img_train,total_time,STDP_time,retrain_params,label_train);

% %���ڽ��й�relearn��Ȩֵ���з���-----------------------------------------------------------------------------------------------
% [X_learn,T_learn] = get_feature(weights,layers,network_struct,spike_times_learn,num_img_learn,DoG_params,total_time);
% %%��Ӧ��ǩΪlabel_learn
% [learn_label] = check_T(T_learn);
% learning_correct_rate=correct_rate(label_learn,learn_label);
% 
% [X_train,T_train] = get_feature(weights,layers,network_struct,spike_times_train,num_img_train,DoG_params,total_time);
% %��Ӧ��ǩΪlabel_train
% [train_label] = check_T(T_train);
% training_correct_rate=correct_rate(label_train,train_label);
% 
% [X_test,T_test] = get_feature(weights,layers,network_struct,spike_times_test,num_img_test,DoG_params,total_time);
% %��Ӧ��ǩΪlabel_test
% [test_label] = check_T(T_test);
% testing_correct_rate=correct_rate(label_test,test_label);
 end      
%Ȩֵ�洢
if save_weights==1
    save('weights_5_7_retrain_SDNN.mat','weights');
end
%----------------------------------�����б����������--------------------------------------------------------------------
% %�����б�
[X_learn,T_learn] = get_feature(weights,layers,network_struct,spike_times_learn,num_img_learn,DoG_params,total_time);
[got_learn_label] = check_T(T_learn);%%��Ӧ��ǩΪlabel_learn
 learn_correct = correct_rate(label_learn,got_learn_label);
 
%  
% [X_train,T_train] = get_feature(weights,layers,network_struct,spike_times_train,num_img_train,DoG_params,total_time);
% [got_train_label] = check_T(T_train);%��Ӧ��ǩΪlabel_train
% train_correct = correct_rate(label_train,got_train_label);


[X_test,T_test] = get_feature(weights,layers,network_struct,spike_times_test,num_img_test,DoG_params,total_time);
[got_test_label] = check_T(T_test);%��Ӧ��ǩΪlabel_test
test_correct = correct_rate(label_test,got_test_label);

% %---------------------------------������------------------------------------------------------------------------
SVM_learn=fitcsvm(X_learn,label_learn);
result_learn=predict(SVM_learn,X_learn);
% result_train=predict(SVM_train,X_train);
result_test=predict(SVM_learn,X_test);
SVM_learning_correct_rate=correct_rate(label_learn,result_learn);
% SVM_training_correct_rate=correct_rate(label_train,result_train);
SVM_testing_correct_rate=correct_rate(label_test,result_test);

 svm_train1(i)=SVM_learning_correct_rate;
 svm_test1(i)=correct_rate(label_test,result_test);






end