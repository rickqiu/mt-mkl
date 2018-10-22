# mt-mkl
The library performs multitask-multiple kernel learning for time series analysis. We focus on the analysis of time series acquired on focal epileptic patients. In this case the source of the seizure is limited to a specific area of the brain. For drug resistant patients, the ablation is sometimes a solution. To localize at best the epileptic area, clinicians perform invasive measures, through deep filiform electrodes, which are implanted under skull. Each electrode on average has 10 channels of acquisition, and a correspondent time series.
Given the tag assignment (binary label that denotes if the channel has epileptic activity) performed by medical experts, which we use as ground truth, mt-mkl addresses at the same time two problems (i) classification of the activity and the (ii) extraction of relevant features in the frequency domain.  

The library consists of 
**preprocessing step**: we filter each time series to remove powerline effects and then we proceed to the computation of wavelet transform - *NEW for a total of 83 scales*
**kernel computation**: we use different similarity measures. At the moment they are correlation, phase locking value, *NEW cross correlation (not the average value)*
**multikernel**: this part of the library includes minimization methods
**scripts**: the first is test_wavelet_trasform.py for the computation of the wavelet transform and the kernels
                        learning_pipeline.py learning method through random search cv
