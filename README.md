DSD: Unsupervised Landmark Detection Based Spatiotemporal Motion Estimation for 4D Dynamic Medical Images
====
Introduction
----
In this study, we provide a novel unsupervised topology guided motion estimation framework, which we termed Dense-Sparse-Dense (DSD) framework, comprising of two stages. In the first stage, we process the raw dense image to extract sparse landmarks to represent the target organ’s anatomical topology. For this process, we introduce an unsupervised 3D landmark detection network to extract spatially sparse but representative landmarks for target organ’s motion estimation, while where suppressing redundant information that is unnecessary for motion estimation. In the second stage, we derive motion displacement from the extracted sparse landmarks of two images of subsequent time-points. Then, we construct the motion field by projecting the sparse landmarks’ displacement back into the dense image domain. 

<div align=center><img width="250" height="180" src="https://github.com/guoyu-niubility/DSD-3D-Unsupervised-Landmark-Detection-Based-Motion-Estimation/blob/main/images/cardiac-1.gif" alt="Case-1: Cardiac motion in landmark representation"/><img width="250" height="180" src="https://github.com/guoyu-niubility/DSD-3D-Unsupervised-Landmark-Detection-Based-Motion-Estimation/blob/main/images/cardiac-2.gif" alt="Case-1: Cardiac motion in landmark representation"/>

![image](https://github.com/guoyu-niubility/DSD-3D-Unsupervised-Landmark-Detection-Based-Motion-Estimation/blob/main/images/cardiac-1.gif)
![image](https://github.com/guoyu-niubility/DSD-3D-Unsupervised-Landmark-Detection-Based-Motion-Estimation/blob/main/images/cardiac-2.gif)
![image](https://github.com/guoyu-niubility/DSD-3D-Unsupervised-Landmark-Detection-Based-Motion-Estimation/blob/main/images/lung-1.gif)
![image](https://github.com/guoyu-niubility/DSD-3D-Unsupervised-Landmark-Detection-Based-Motion-Estimation/blob/main/images/lung-2.gif)
