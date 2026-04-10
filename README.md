# GPU-Accelerated-Network-Traffic-Classifier
In this project, we have built a highly efficient deep learning pipeline capable of differentiating between malicious and benign network traffic with extremely low latency, using the power of GPUs and CUDA-enabled data transfers.
Objective: Build an efficient IDS system through GPU-enabled hardware and Deep Learning Frameworks.
Approach: In order to increase parallelism and reduce time spent on complex matrix multiplications, a DNN was built using PyTorch with CUDA kernels.
minimizing delays when transferring data from the network card to GPU memory.
Conclusion: A 12 times reduction in inference latency was achieved while delivering 10 Gbps throughput with 98.2% accuracy for five types of malicious traffic.
