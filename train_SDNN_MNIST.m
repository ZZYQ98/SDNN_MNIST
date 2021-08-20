 function [weights]=train_SDNN_MNIST(weights,layers,network_struct,spike_times_learn,STDP_params,num_img_learn,learnable_layers,total_time,deta_STDP_minus,deta_STDP_plus)
%UNTITLED2 此处显示有关此函数的摘要.
%   此处显示详细说明
% STDP_per_layer=STDP_params.STDP_per_layer;
% offset_STDP=STDP_params.offset_STDP;
[~,num_layers]=size(network_struct);
max_iter=STDP_params.max_iter;
curr_lay_idx=1;
learning_layer=learnable_layers(1);%训练层对应的层数
counter=1;
cl_store=zeros(1,1000);
n=1;
 fprintf('-------------------- STARTING LEARNING---------------------\n')             %开始训练，之后进行迭代
 for i=1:max_iter  %max_iter 为最大迭代次数
     perc=i/max_iter;
     fprintf('---------------------LEARNING PROGRESS %1.0f/%1.0f --- %2.4f-------------------- \n',i,max_iter,perc)  %显示当前的训练进度
      if i==100
          save("weights_100.mat","weights");
      elseif i==200
          save("weights_200.mat","weights");
      elseif i==300
          save("weights_300.mat","weights");
      elseif i==400
          save("weights_400.mat","weights");
      elseif i==500
          save("weights_500.mat","weights");
      elseif i==600
          save("weights_600.mat","weights");
      elseif i==700
          save("weights_700.mat","weights");
      elseif i==800
          save("weights_800.mat","weights");
      elseif i==900
          save("weights_900.mat","weights");
      elseif i==1000
          save("weights_1000.mat","weights");
      elseif i==1100
          save("weights_1100.mat","weights");
      elseif i==1200
          save("weights_1200.mat","weights");
      elseif i==1300
          save("weights_1300.mat","weights");
      elseif i==1400
          save("weights_1400.mat","weights");
      elseif i==1500
          save("weights_1500.mat","weights");
      elseif i==1600
          save("weights_1600.mat","weights");
      elseif i==1700
          save("weights_1700.mat","weights");
      elseif i==1800
          save("weights_1800.mat","weights");
      end
          
      
     if counter>STDP_params.max_learning_iter(learning_layer)      %max_learning_iter由输入参数定义，表示最大的迭代次数
         curr_lay_idx=curr_lay_idx+1;%切换到下一个学习层
         learning_layer=learnable_layers(curr_lay_idx);%learning layer的定义，得到用于当前学习的矩阵
         counter=1;
     end
      counter=counter+1;
     %调用函数 reset_layers
     layers=reset_layers(layers,num_layers);%将所有的层进行恢复
     
      %得到输入脉冲矩阵
    % path_img=spike_times_learn{n};
     MNIST_data=spike_times_learn(:,:,n);
      if n<num_img_learn
          n=n+1;
      else
          n=1;
      end    
    % st=DoG_filter_to_st(path_img,DoG_params.DoG_size,DoG_params.img_size,total_time,num_layers);%  st = spike_time 输入脉冲时间
     st= MNIST_coding( MNIST_data,total_time,num_layers);%其中mnist_data为单一的输入图像
     layers_buff=init_layers(network_struct);%流水线结构
     weights=train_step3( weights,layers,layers_buff,network_struct,total_time,learning_layer,st,STDP_params.STDP_per_layer,deta_STDP_minus,deta_STDP_plus,STDP_params.offset); %调用 函数train_step()  输入脉冲为st，包含了时间信息，针对输入脉冲开始进行权值训练

     w=weights{2};
[Hw,Ww,Mw,Dw]=size(w);
N=Hw*Ww*Dw;
cl=0;
for kw=1:Dw
    for iw=1:Hw
        for jw=1:Ww
            cl=cl+(w(iw,jw,kw)*(1-w(iw,jw,kw)))/N;
        end
    end
end
cl_store(i)=cl;  
 end
    fprintf('---------LEARNING PROGRESS %2.3f------------- \n',perc)
     
    fprintf('-------------------- FINISHED LEARNING---------------------\n')

end

