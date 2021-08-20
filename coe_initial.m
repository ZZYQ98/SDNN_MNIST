t_step_for_ram=zeros(I_num,1);
for i = 1: I_num
    X10=X(i);
    Y10=Y(i);
    X2=dec2bin(X10,8);
    Y2=dec2bin(Y10,8);
    st2=dec2bin(t_step(X(i),Y(i)),8);
    AER=[X2,Y2,st2];
    t_step_for_ram(i)=AER;
end



fs=100;
N=1024;
n=0:N-1;
t=n/fs;
x=0.5*sin(2*pi*20*t);
% 量化位宽
width = 16;%数据宽度16位
% 量化滤波器系数
sin_data  = round(x .* (2^(width-1) - 1));%量化正弦波形数据并取整

data_com_sin = zeros(1,length(sin_data));
for i = 1:length(sin_data)
   if sin_data(i) >= 0
     data_com_sin(i) = sin_data(i);
   else
    data_com_sin(i) = 2^width + sin_data(i);
   end
end
fid=fopen('F:\lyl_project\cos_coe.coe','w'); %创建.coe文件
 fprintf(fid,'%d,\n',data_com_sin);%向.coe文件中写入数据
 fclose(fid); %关闭.coe文件
