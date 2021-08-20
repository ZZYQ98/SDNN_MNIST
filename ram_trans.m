%创建SRAM中存储的t_step的地址
%前八位为X，后八位为Y，之后八位为t_step
[~,spike_num]=size(X);
t_step_for_ram=zeros(spike_num,24);
for i = 1: spike_num
    X10=X(i);
    Y10=Y(i);
    X2=dec2bin(X10,8);
    Y2=dec2bin(Y10,8);
    st2=dec2bin(t_step(X(i),Y(i)),8);
    t_step_for_ram(i,:)=[X2,Y2,st2];
end