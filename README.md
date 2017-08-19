# my_svm
### Simple Implementation of SVM from scratch

The SVM fit function based on simplified version of Sequential Minimal Optimization (SMO) algorithm as described [here](http://cs229.stanford.edu/materials/smo.pdf).
The full SMO algorithm firstly proposed by John Platt and described in this [paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-98-14.pdf).
Additional useful source about SMO can be found [here](http://nshorter.com/ResearchPapers/MachineLearning/A_Roadmap_to_SVM_SMO.pdf)

The main disadvantage of our simplified SMO implementation is a not guaranteed convergence  for all data sets.

### Main Features
- SVM class implemended from scratch
- Currently supported only linear kernel, but more kernels can be easily added by extending  `_kernel(self, a, b)` function
- Comparison to `sklearn.svm.SVC` 
- Model visualization

### Example
``` python
    # Generate random dataset with 2 classes in 2D space
    data, labels = sklearn.datasets.make_blobs(centers=2, cluster_std=2.5)

    # Show the data
    vis_data(data, labels)

    # Fit simple SVM model
    my_svm = SimpleSvm()
    my_svm.fit(data, labels)

    # Compare to sklearn.svm.SVC
    sk_svm = my_svm.compare_to_sklearn()
```
