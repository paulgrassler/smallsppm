# smallsppm

Compact implementation of the Stochastic Progressive Photon Mapping algorithm by [Hachisuka and Jensen, 2009](https://doi.org/10.1145/1618452.1618487). The goal was to create a single source file implemenation in the style of [smallpt by Kevin Beason](https://www.kevinbeason.com/smallpt/). I based this implementation on a [small progressive photon mapper by Hachisuka](https://cs.uwaterloo.ca/~thachisu/smallppm_exp.cpp) and used a kd-tree implementation from another [SPPM implementation](https://github.com/shizhouxing/SPPM). Features include area light sources, a thin lens model, Motion Blur, the modified Phong model for glossy reflections and direct light sampling.

## Quick Start
To render a scene with smallsppm, first compile the program with e.g. g++:

```terminal
g++ -fopenmp -O3 smallsppm.cpp
```

Then execute it with:

```terminal
smallsppm.exe <Number of Rounds> <Scene Number> <Max Time>
```

This command will render the corresponding scene with \<Number of Rounds\> SPPM iterations. An output folder will be created if it does not exist already and a subfolder with date, time, scene number and initial radius will contain the generated renderings. Images will be created at different iteration checkpoints and at the end of the process. It will stop early, if the elapsed time reaches the specified \<Max Time\>.

Iteration checkpoints are specified here:

```C++
std::vector<int> checkpoints = {10, 100, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000};
```

### Creating a new Scene

To create a new scene, one needs to create an array of scene objects and a scene description.

#### Scene Objects

There is a variety of different scene objects that can be used, which we will go over in this section. For each object, I will show the constructor and explain the different parameters.

**Spheres**

```C++
Sphere(double rad_, Vec p_, Vec e_, Vec c_, BRDF brdf_, Vec direction_ = Vec())
```

The constructor takes the sphere radius (rad_), its position (p_), emission (e_), albedo (c_), BRDF (brdf_) and direction (direction_) in case of scenes with Motion Blur.

**Trimesh**

```C++
Trimesh(std::vector<Triangle *> triangles_, Vec p_, Vec e_, Vec c_, BRDF brdf_)
```

The constructor takes a vector of Triangle objects (triangles_), the position of the mesh (p_), its emission (e_), albedo (c_) and BRDF (brdf_).

**Triangle**

```C++
Triangle(Vec a_, Vec b_, Vec c_)
```

Triangles are defined by the position of their three points (a_, b_, c_). They should only ever be used inside a Trimesh.

**Cylinder**

```C++
Cylinder(double rad_, double length_, Vec p_, Vec e_, Vec c_, BRDF brdf_)
```

The cylinder constructor takes a radius (rad_), the length of the cylinder (length_), its position (p_), emission (e_), albedo (c_) and BRDF (brdf_). Only upright cylinders are supported.

**BRDF**

Although not a scene object, this will be required for creating any object. 

```C++
BRDF(Refl_t refl_, double k_d_ = 1.0, double k_s_ = 0.0, double gloss_ = 1.0, Vec sc_ = Vec(1, 1, 1) * .999)
```

The reflection type (refl_) can be chosen from:

- PHONG (modified Phong model)
- SPEC (perfect specular reflection)
- REFR (glass surfaces, **only implemented for sphere!**)

The other parameters are only relevant if the reflection type is PHONG. The user can specifiy the diffuse reflectivity (k_d_), the specular reflectivity (k_s_) and corresponding glossiness (gloss_), and a seperate albedo for specular reflections (sc_, typically this should not bet set).

#### Creating the Scene Objects Array

We need to create an array of object pointers to pass to the scene description later. An example can be seen below:

```C++
Object *scene1[] = {
  new Sphere(1e5, Vec( -1e5 -BOX_HALF_X,0,0), Vec(), Vec(.75,.25,.25), BRDF(PHONG)),//Left
  new Sphere(1e5, Vec(1e5 + BOX_HALF_X,0,0),Vec(), Vec(.25,.25,.75), BRDF(PHONG)),//Right
  new Sphere(1e5, Vec(0, 0, -1e5 - BOX_HALF_Z),     Vec(), Vec(.75,.75,.75), BRDF(PHONG)),//Back
  new Sphere(1e5, Vec(0,0,+1e5+ 3 * BOX_HALF_Z - 0.5), Vec(), Vec(),            BRDF(PHONG)),//Front
  new Sphere(1e5, Vec(0, -1e5 - BOX_HALF_Y, 0),    Vec(), Vec(.75,.75,.75), BRDF(PHONG)),//Bottomm
  new Sphere(1e5, Vec(0,1e5 + BOX_HALF_Y, 0),Vec(), Vec(.75,.75,.75), BRDF(PHONG)),//Top
  new Sphere(0.8,Vec(-1.3, -BOX_HALF_Y + 0.8, -1.3),       Vec(), Vec(1,1,1)*.999, BRDF(SPEC)),//Mirror
  new Sphere(0.4, Vec(0,-BOX_HALF_Y + 0.4,-0.8),        Vec(), Vec(.75,.75,.75), BRDF(PHONG)),//Middle
  new Sphere(0.2, Vec(0,BOX_HALF_Y * 0.8,0.0), Vec(1, 1, 1) * 100, Vec(), BRDF(PHONG)) //LIGHT
  };
```

#### Creating a new Scene Description

 The constructor for a scene description can be seen here:

 ```C++
 SceneDescription(struct Object **scene_, int scene_size_, Vec sensor_origin_, Vec sensor_direction_, Object *light_, double r_0_, bool direct_sampling_ = false, bool motion_blur_ = false, bool dof_ = false, double S_o_ = 0.0, double f_stop_ = 1.0)
 ```

 The first parameter is a pointer to the aforementioned scene objects array. The number of objects should be passed as scene_size_. The user can specify the position and direction of the camera sensor (sensor_origin_ and sensor_direction_). A pointer to the actual light source inside the scene objects array needs to be passed. **Only one light source per scene is supported at the moment.**. An initial radius for the hit points needs to be specified (r_0_). The rest of the rendering options are:

 -  Direct Light Sampling (direct_sampling_): if activated, direct light will be accounted for in the Ray Tracing Passes, which can increase performance as the first bounces of Photons can be ignored.
 - Motion Blur (motion_blur_): if activated, SPPM will sample a timestamp at the beginning of each iteration and move the scene objects as specified by their parameters. (**Only Spheres can be moved at the moment.**).
 - Depth of Field (dof_): if activated, camera rays will sample lens positions to emulate the behavious of real world cameras (thin lens model). An object distance (S_o) and f-Stop number (f_stop_) can be specified.

 When you are finished with your scene description, just include it in the scene_descriptors array:

 ```C++
 SceneDescription scene_descriptors[] = {scene_desc1};
 ```

 Now you should be able to render it, when calling

```terminal
smallsppm.exe <Number of Rounds> 1 <Max Time>
```

