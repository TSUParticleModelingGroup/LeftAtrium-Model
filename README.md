# Simulating Left Atrial Arrhythmias with an Interactive N-body Model

### Front View ---- Back View ---- View Through the Mitral Valve
<img src="imgs/LAFront.jpeg" width=30% height=30% class='center'></img>
<img src="imgs/LABack.jpeg" width=30% height=30% class='center'></img>
<img src="imgs/LAThroughMV.jpeg" width=30% height=30% class='center'></img>
### Flutter Between Pulmonary Veins ---- Roof Flutter ---- Micro-reentry
<img src="imgs/FlutterPV.jpeg" width=30% height=30% class='center'></img>
<img src="imgs/RoofFlutter.jpeg" width=30% height=30% class='center'></img>
<img src="imgs/Micro.jpeg" width=30% height=30% class='center'></img>

### Project Aims

Our project has several key objectives. Firstly, we aim to utilize N-body techniques to develop an interactive model of the left atrium. This model will allow users to manipulate various parameters in real-time, facilitating the induction and observation of common arrhythmias.

Secondly, we seek to create a training and study tool for electrophysiologists, researchers, and medical students. Through accurately simulating left atrial arrhythmias and their treatment using simulated ablations, the model will serve as a valuable educational resource, enhancing understanding and skill development in this critical medical field.

Additionally, the project aims to advance research in electrophysiology by providing a platform for exploring novel treatment strategies and studying arrhythmia mechanisms. This could lead to new insights and innovations in the field, ultimately benefiting patients with cardiac arrhythmias.

In summary, the project's objectives include developing a cutting-edge model for arrhythmia simulation, providing an advanced training tool for medical professionals, and advancing research in electrophysiology.


### Table of Contents
- [Installation](#installation)
  - [Hardware Requirements](#hardware-requirements)
  - [Software Requirements](#software-requirements)
  - [Building](#building)
- [Running](#running)
- [Controls](#controls)
- [Units of Measurement](#UnitsofMeasurement)
- [Changelog](#changelog)
- [Contributing](#contributing)
- [Citation](#citation)

## Installation
### Hardware Requirements:
- This simulation requires a CUDA-enabled GPU from Nvidia. Click <a href="https://developer.nvidia.com/cuda-gpus">here </a> for a list of GPUs.

| *Note: These are guidelines, not rules | CPU                            | GPU                   | RAM       |
|----------------------------------------|--------------------------------|-----------------------|-----------|
| Minimum:                               | AMD/Intel Six-Core Processor   | Any CUDA-Enabled GPU  | 16GB DDR4 |
| Recommended:                           | AMD/Intel Eight-Core Processor | RTX 3090/Quadro A6000 | 32GB DDR5 |

### Software Requirements:

#### Disclosure: This simulation only works on Linux-based distros currently. All development and testing was done in Ubuntu 20.04/22.04

#### This Repository contains the following:
- [Nsight Visual Studio Code Edition](https://developer.nvidia.com/nsight-visual-studio-code-edition)
- [CUDA](https://developer.nvidia.com/cuda-downloads)
   - OpenGL
        - [Nvidia Driver For OpenGL](https://developer.nvidia.com/opengl-driver)
        - [OpenGL Index](https://www.khronos.org/registry/OpenGL/index_gl.php)
#### Linux (Ubuntu/Debian)
  Install Nvidia CUDA Toolkit:

    sudo apt install nvidia-cuda-toolkit

  Install Mesa Utils:

    sudo apt install mesa-utils

### Building (Note: this must be done after every code change)

  Navigate to the cloned folder and run the following command to build and compile the simulation:

    ./compile

## Running
  After compiling, run the simulation:

    ./run
  
## Setup File 

### Units of Measurement
   Length is in millimeters (mm)  
   Time is in milliseconds (ms)  
   Mass is in grams (g)  
   
### New Run or Previous Run
   	You can start a new run using the nodes and muscles files, or you can continue a previous run.
   	NodesMusclesFileOrPreviousRunsFile = 0, run from the selected nodes and muscles file
   	NodesMusclesFileOrPreviousRunsFile = 1, run from a previous run file
   
   	If you selected 0, then you must set 
   	InputFileName = ***;
   	to the name of the nodes and muscles file you want to run from the list below. 
   	{Line11, Circle24, CSphere340, CSphere5680, IdealLeftAtrium13.0KNotTriangle, LeftAtriumRealRemovedAppendage}
   
   	If you selected 1, then you must set 
   	PreviousRunFileName = ***;
   	to the name of a previous run file you saved in the PreviousRunsFile folder. The three previous run files listed below
   	are alredy placed in this folder to use as demos. 
   	{PVFlutterDemo, MicroReentryDemo, RoofFlutterDemo}
   
### Nodes and Muscle View Size, and colors
   	You can set the size of the nodes and muscles with the
   	LineWidth = ***;
   	NodeRadiusAdjustment = ***;
   
   	Colors to help distinguish between simulation events can all be customized at the end of the setup file. 
   
   	Note: This only affects the viewing of the simulation; no actual functionality is changed with these parameters.

### Simulation Constants

   	Myocyte Force Per Mass strength = 596.0 mm/ms^2
   	BloodPressure = 80.0; millimeters of merculry converted to g/(mm*ms*ms) in the program.
   	MassOfAtria = 25; g
   	RadiusOfAtria = 17.8; mm
   	BaseMuscleRelaxedStrength = 2.0; This is just a force that helps the model keep its shape.
   	BaseMuscleCompresionStopFraction = 0.7 This only lets a muscle fiber reduce its length by 30%
   	BeatPeriod = 1000.0; (ms)
   	MaxNumberOfperiodicEctopicEvents = 50; This just sets an upper limit to the number ectopic beats a simulation can have.
   	Note: ectopic beats are extra pulse node that the user sets in an active simulation. Ectopic triggers are single events
   	stimulated by mouse clicks.
   
   The above are typical values and are all changable in the setup file. These values are read in at the start of a simulation. 
   They are not changable once the simulation starts. 
   
### Simulation Variables
   BaseMuscleContractionDuration = 200.0; (ms)
   BaseMuscleRechargeDuration = 200.0; (ms)
   BaseMuscleConductionVelocity = 0.5; (mm/ms)
   
   The above are typical values read in from the setup file at the start of a simulation. These values are all changable in an active simulation.
   
### Timing constants
   PrintRate = 100.0; How often the program prints new information to the terminal screen. 
   DrawRate = 1000; How often the program draws a new simulation picture. 
   Dt = 0.001; How many Leap-Frog iterations are done for each ms of simulation time.
   
## Controls
  
  <img src="imgs/commands.png" width=80% height=80%>

## Changelog

Refer to the changelog for details.

## License
  - This code is protected by the MIT License and is free to use for personal and academic use.

## Contributing
  - Dr. Bryant Wyatt (PI)
  - Mr. Gavin McIntosh
  - Mr. Avery Campbell
  - Mr. Derek Hopkins
  - Ms. Leah Rogers
  - Ms. Melanie Little

## Citation
  
      Dubin D. Ion Adventure in the Heartland: Exploring the Heart's Ionic-Molecular Microcosm. 
      Tampa, Florida: Cover Publishing; 2015.
      
      Dubin D. Rapid Interpretation of EKG's: ... an Interactive Course. 6th ed. 
      Fort Myers, Florida: Cover Publishing Company; 2007.
      
      Wynsberghe DV, Carola R, Noback CR. Human Anatomy and Physiology. 3rd ed. 
      London: McGraw-Hill; 1995.
      
      Klabunde RE. Cardiovascular Physiology Concepts. 
      Philadelphia, Pennsylvania: Wolters Kluwer; 2021. 
      
      Bozkurt S. (2019). Mathematical modeling of cardiac function to evaluate clinical cases in adults and children. 
      PloS one, 14(10), e0224663. https://doi.org/10.1371/journal.pone.0224663
      
      Lodish H, Berk A, Zipursky SL, et al. Molecular Cell Biology. 4th edition. 
      New York: W. H. Freeman; 2000. Section 18.4, Muscle: A Specialized Contractile Machine. 
      Available from: https://www.ncbi.nlm.nih.gov/books/NBK21670/
      
      Karki DB, Pant S, Yadava SK, Vaidya A, Neupane DK, Joshi S. 
      Measurement of right atrial volume and diameters in healthy Nepalese with normal echocardiogram. 
      Kathmandu Univ Med J (KUMJ). 2014;12(46):110-112. doi:10.3126/kumj.v12i2.13655
      
      Irie T, Kaneko Y, Nakajima T, et al. Electroanatomically estimated length
      of slow pathway in atrioventricular nodal reentrant tachycardia. Heart Vessels.
      2014;29(6):817-824. doi:10.1007/s00380-013-0424-0
      

The Particle Modeling Group reserves the right to change this policy at any time.
