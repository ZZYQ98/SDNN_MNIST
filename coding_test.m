load('MNIST.mat');
total_time=20;
for i =1:100
   MNIST_training_data=training_data(:,:,i);
   t_step=MNIST_coding(MNIST_training_data,total_time);
end