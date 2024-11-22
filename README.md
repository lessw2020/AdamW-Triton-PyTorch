# AdamW-Triton-PyTorch
Can AdamW written in Triton be as performant as fused CUDA impl? 

Progress Updates:  
1 - AdamW written in Triton is passing all unit tests, and thus appears to have numerical equivalency.  
Todo - perf test on larger scale training.

2 - AdamW with RowWise FP8 quantization for Momentum term is passing basic unit tests.    
Immediate gain is 37.5% reduction in total optimizer state memory.   
Todo - perf test on larger scale training. 
