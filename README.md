# Supraventricular Tachycardia Study Using a Dynamic Computer Generated Heart

This project will allow the user to identify, define, and educate about cardiac ablations and Supraventricular Tachycardia's using an unforeseen dynamical approach to cardiac simulation.

### Table of Contents
- [Installation](#installation)
  - [Requirements](#requirements)
  - [Building](#building)
  - [Linux](#linux)
- [Controls](#controls)
- [Units of Measurement](#UnitsofMeasurement)
- [Changelog](#changelog)
- [License](#license)
- [Contributing](#contributing)
- [Citation](#citation)

## Installation
### Requirements:

#### This Repository contains the following:
- [Nsight Visual Studio Code Edition](https://developer.nvidia.com/nsight-visual-studio-code-edition)
- [CUDA](https://developer.nvidia.com/cuda-downloads)
   - OpenGL
        - [Nvidia Driver For OpenGL](https://developer.nvidia.com/opengl-driver)
        - [OpenGL Index](https://www.khronos.org/registry/OpenGL/index_gl.php)
#### Linux (Ubuntu/Debian)
 - sudo apt install nvidia-cuda-toolkit
 - sudo apt install mesa-utils

### Building

After cloning the repository, checkout out the `main` branch and set up your environment:

  - git checkout main

### Running

Access your terminal and go ahead and compile to test everything is good and run using ./svt4.0

nvcc SVT4.0.cu -o svt4.0 -lglut -lm -lGLU -lGL

./svt 4.0

### Controls

  - Use the terminal to set your basic parameters (Circle or sphere and number of divisions).
  The simulation will be paused at the start. Move to the mouse over the simulation window and type the following commands.
  
  General:
  
    R to Run
    P to Pause
    Q to End
  Camera Controls:
  
    O for Orthogonal View
    F for Fultrum View
    C for Center View
    N for Center on the Sinus Node
  
      Use Ctrl with any command below for the inverse effect.
    
        X to Rotate on the X-Axis
        Y to Rotate on the Y-Axis
        Z to Rotate on the Z-Axis
        W to Zoom Out
      
  Click to ablate a node, make sure you are in orthogonal view
  
  Right-Click to un-ablate a node
  
  
## Units of Measurement

    Length is in millimeters
    Time is in milliseconds
    Mass is in grams
    Viscosity is in grams/(millimeters * milliseconds^2)

     Unit Constants
        Fiber length 100 micrometers or 0.1 millimeters
        Action Potential Speed .5 meters/sec
        Muscle Compression Fraction is 30 percent
        Contraction Duration is ~100 milliseconds
        Relaxation Duration is ~200 milliseconds
        Short Axis Circumference is 200 millimeters
        



## Changelog

Refer to the changelog for details.

## License

## Contributing
  - Dr. Bryant Wyatt (PI)
  - Mr. Gavin McIntosh
  - Mr. Avery Campbell

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
