# TransientVisualizer
The TransientVisualizer was designed to extract calcium transients and sarcomere shortening data from microscope line-scan images.
![image](https://github.com/IuliiaBaglaeva/TransientVisualizer/assets/108415908/7c6a3a36-3fb8-428e-9a86-2378e74e176d)

Screenshot is taken from Ph.D thesis "THE EFFECT OF PHYSIOLOGICAL LOAD ON EXCITATION-CONTRACTION COUPLING OF CARDIOMYOCYTES", Iuliia Baglaeva

The PyQt5 library is used by this software. The software opens microscope images in OME-TIFF format containing fluorescence and transmission channels. 
The software can export files with time cources in .xlsx and .csv formats (it can be chosen by user) for calcium transients and sarcomere shortening separately.

Software was tested on Python 3.9 and does not work on Python 3.10 and 3.11. 

Authors:  Iuliia Baglaeva, Bogdan Iaparov, Ivan Zahradník and Alexandra Zahradníková, Department of Cellular Cardiology, Institute of Experimental Endocrinology, Biomedical Research Center of the Slovak Academy of Sciences.
