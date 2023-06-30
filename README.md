# Clustering of HCP Unrelated 100 Subjects by ROI-Correlations in Resting State

This is a Work In Progress (WIP) project. It is part of my undergraduate studies in Biomedical Engineering at
Ostbayerische Technische Hochschule in Germany. This repository incorporates the code and results of my participation
in the course Data Science Projects: Train you own Machine Learning Model. The goal of this project is to demonstrate
that clustering of Resting State fMRI data to distinguish between different gender of subjects is possible. This should 
pave the way for a future diagnosis of different diseases that are connected to reorganisation of the human brain.

## Introduction

MRI (Magnetic Resonance Imaging) is a procedure in medical diagnostics which aims to produce three-dimensional images 
of the body. In comparison to X-Ray and Computertomography it does not incorporate the usage of ionizing radiation and 
is primarily used for evaluating anatomy and physiology of soft tissue. It's based on the nuclear spin of hydrogen 
atoms, which can be measured by applying different techniques regarding magnetic fields. An advancement of this process 
is called fMRI which means functional MRI. This procedure uses the so called BOLD-contrast which stands for 
blood-oxygen-level dependent contrast. Due to the difference in the magneticity of oxygenrich blood and the one low in 
oxygen neural activity in the brain can be measured.
The HCP (Human Connectom Project) is researching human connectomics, the connections in the brain, on different scales 
(e.g. between regions or individual neurons). MEG, EEG and MRI has been performed on about 1200 patients. In addition, 
genetic sequencing and several sensory and motoric test as well as cognitive evaluation took place. At this point this 
work is using only parts of the resting state fMRI data (specifically Resting State fMRI 1 FIX-Denoised Extended) in 
the HCP Unrelated 100 dataset. This is due to considerations regarding storage space as well as processing time and 
power. Therefore, only the first session in the downloaded data has been used until now.

## Acknowledgements and additional notes
Data were provided by the Human Connectome Project, WU-Minn Consortium (Principal Investigators: David Van 
Essen and Kamil Ugurbil; 1U54MH091657) funded by the 16 NIH Institutes and Centers that support the NIH Blueprint for 
Neuroscience Research; and by the McDonnell Center for Systems Neuroscience at Washington University.

Please note that due to the implementation of the "Intel® Extension for Scikit-learn" package only Intel processors and 
no AMD processors can be used for the execution of the code.

## License
Copyright (c) 2023, Lukas Behammer
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.