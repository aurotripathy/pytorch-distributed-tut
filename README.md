### A 'Hello World' Tutorial on DistributedDataParallel PyTorch

This highlights the use of the PyTorch DistributedDataParallel library and is expected to supplement https://pytorch.org/tutorials/intermediate/ddp_tutorial.html 

Uses the **torch.utils.data.distributed.DistributedSampler** in conjunction with **torch.nn.parallel.DistributedDataParallel**. Each process passes a DistributedSampler instance as a DataLoader sampler and laods a subset of the original dataset that is exclusive to it. 

Reasons for the poor convergence of a small network/model (three layers)   are highlighted in https://github.com/mli/mxnet/tree/master/example/image-classification#speed
