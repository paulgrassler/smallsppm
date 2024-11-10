# smallsppm

Compact implementation of the Stochastic Progressive Photon Mapping algorithm by [Hachisuka and Jensen, 2009](https://doi.org/10.1145/1618452.1618487). The goal was to create a single source file implemenation in the style of [smallpt by Kevin Beason](https://www.kevinbeason.com/smallpt/). I based this implementation on a [small progressive photon mapper by Hachisuka](https://cs.uwaterloo.ca/~thachisu/smallppm_exp.cpp) and used a kd-tree implementation from another [SPPM implementation](https://github.com/shizhouxing/SPPM). Features include area light sources, a thin lens model for Depth of Field, Motion Blur, the modified Phong model for glossy reflections, direct light sampling and some more.

## Quick Start
To render a scene with smallsppm, first compile the program with e.g. g++:

```terminal
g++ -fopenmp -O3 smallsppm.cpp
```

Then execute it with:

```terminal
smallsppm.exe <Number of Rounds> <Scene Number> <Max Time>
```

This command will render the corresponding scene (\<Scene Number\> refers to its position in the scene_descriptors array) with \<Number of Rounds\> SPPM iterations. An output folder will be created if it does not exist already and a subfolder with date, time, scene number and initial radius will contain the generated renderings. Images will be created at different iteration checkpoints and at the end of the process. It will stop early if the elapsed time reaches the specified \<Max Time\> (in seconds).

### Creating a new Scene

To create a new scene, one needs to create an array of scene objects and a scene description.

#### <ins>Scene Objects</ins>

There is a variety of different scene objects that can be used, which we will go over in this section. For each object, I will show the constructor and explain the different parameters.

**Spheres**

```C++
Sphere(double rad_, Vec p_, Vec e_, Material mat_, Vec direction_ = Vec())
```

The constructor takes the sphere radius (rad_), its position (p_), emission (e_), material (mat_), and direction (direction_) in case of scenes with Motion Blur. The direction specifies a vector along which the object will be moved during the shutter interval. More complex motions are not supported.

**Trimesh**

```C++
Trimesh(std::vector<Triangle *> triangles_, Vec p_, Vec e_, Material mat_)
```

The constructor takes a vector of Triangle objects (triangles_), the position of the mesh (p_), its emission (e_), and the material (mat_).

**Triangle**

```C++
Triangle(Vec a_, Vec b_, Vec c_)
```

Triangles are defined by the position of their three points (a_, b_, c_). They should only ever be used inside a Trimesh (even if only a single Triangle is required).

**Cylinder**

```C++
Cylinder(double rad_, double length_, Vec p_, Vec e_, Material mat_)
```

The cylinder constructor takes a radius (rad_), the length of the cylinder (length_), its position (p_), emission (e_) and material (mat_). Only upright cylinders are supported.

**Material and BRDF**

Each scene object requires a material. It can be created via the following constructor: 

```C++
Material(Vec c_, BRDF *brdf_)
```

It takes the albedo (c_) and a pointer to a BRDF (brdf_) as input parameters.

These are the three different BRDF that are currently implemented:

- BRDF_PHONG (modified Phong model)
- BRDF_SPEC (perfect specular reflection)
- BRDF_REFR (glass surfaces, **only implemented for spheres!**)

The constructors for BRDF_PHONG and BRDF_REFR can be seen below (BRDF_SPEC uses the default constructor since it does not need any parameters):

```C++
BRDF_PHONG(double k_d_ = 1.0, double k_s_ = 0.0, double gloss_ = 1.0)
BRDF_REFR(double ior_ = 1.5)
```

The BRDF_PHONG constructor takes the diffuse reflectivity (k_d_), the specular reflectivity (k_s_) and the glossiness (gloss_) which controls how narrow the specular lobe is. The BRDF_REFR only needs the index of refraction (ior_).

#### <ins>Scene Objects Array</ins>

We need to create an array of object pointers to pass to the scene description later. An example can be seen below:

```C++
Object *scene1[] = {
	new Sphere(1e5, Vec(-1e5 - BOX_HALF_X, 0, 0), Vec(), Material(Vec(.75, .25, .25), new BRDF_PHONG())),
	new Sphere(1e5, Vec(1e5 + BOX_HALF_X, 0, 0), Vec(), Material(Vec(1, 1, 1) * .999, new BRDF_SPEC())),
	new Sphere(1e5, Vec(0, 0, -1e5 - BOX_HALF_Z), Vec(), Material(Vec(.75, .75, .75), new BRDF_PHONG())),
	new Sphere(1e5, Vec(0, 0, +1e5 + 3 * BOX_HALF_Z - 0.5), Vec(), Material(Vec(), new BRDF_PHONG())),
	new Sphere(1e5, Vec(0, -1e5 - BOX_HALF_Y, 0), Vec(), Material(Vec(.25, .25, .75), new BRDF_PHONG())),
	new Sphere(1e5, Vec(0, 1e5 + BOX_HALF_Y, 0), Vec(), Material(Vec(.75, .75, .75), new BRDF_PHONG())),
	new Sphere(0.8, Vec(-1.3, -BOX_HALF_Y + 0.8, -1.3), Vec(), Material(Vec(1, 1, 1) * .999, new BRDF_PHONG(0.0, 1.0, 25))),
	new Sphere(0.6, Vec(BOX_HALF_X * 0.7, -BOX_HALF_Y + 0.6, 0), Vec(), Material(Vec(1, 1, 1) * .999, new BRDF_REFR())),
	new Trimesh(createRectangle(0.01, 0, 0.01), Vec(0, BOX_HALF_Y - EPS, 0), Vec(1, 1, 1) * 200000, Material(Vec(), new BRDF_PHONG()))
};
```

#### <ins>Creating a new Scene Description</ins>

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

 Now you should be able to render it when calling

```terminal
smallsppm.exe <Number of Rounds> 1 <Max Time>
```

