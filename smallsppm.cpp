#include <math.h>
#include <algorithm>
#include <stdlib.h>
#include <stdio.h> 
#include <random>
#include <chrono>
#include <omp.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <ctime>
#define MAX(x, y) ((x > y) ? x : y)
#define MIN(x, y) ((x < y) ? x : y)
#define ALPHA ((double)(2.0/3.0)) // the alpha parameter of SPPM
#define PI ((double)3.14159265358979) // ^^^^^^:number of photons emitted
#define EPS ((double) 0.0000001)

inline double clamp(double x) { return x < 0 ? 0 : x > 1 ? 1 : x; }
int toInt(double x){ return int(pow(1-exp(-x),1/2.2)*255+.5); } 
double rand01() {
	static thread_local std::default_random_engine generator(omp_get_thread_num() +
			std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
	static std::uniform_real_distribution<double> distr(0.0, 1.0);
	return distr(generator);
}

struct Vec {double x, y, z; // vector: position, also color (r,g,b)
	Vec(double x_ = 0, double y_ = 0, double z_ = 0) {x = x_; y = y_; z = z_;}
	inline Vec operator+(const Vec &b) const {return Vec(x+b.x, y+b.y, z+b.z);}
	inline Vec operator-(const Vec &b) const {return Vec(x-b.x, y-b.y, z-b.z);}
	inline Vec operator+(double b) const {return Vec(x + b, y + b, z + b);}
	inline Vec operator-(double b) const {return Vec(x - b, y - b, z - b);}
	inline Vec operator*(double b) const {return Vec(x * b, y * b, z * b);}
	inline Vec operator/(double b) const {return Vec(x / b, y / b, z / b);}
	inline Vec operator/(Vec b) const {return Vec(x / b.x, y / b.y, z / b.z);}
	inline double operator[](int i) const {return (i == 0) ? x : (i == 1) ? y : z;}
	inline double & operator[](int i) {return (i == 0) ? x : (i == 1) ? y : z;}
	inline Vec mul(const Vec &b) const {return Vec(x * b.x, y * b.y , z * b.z);}
	inline Vec norm() {return (*this) * (1.0 / sqrt(x*x+y*y+z*z));}
	inline float length() const {return sqrt(x*x+y*y+z*z);}
	inline double dot(const Vec &b) const {return x * b.x + y * b.y + z * b.z;}
	Vec operator%(Vec&b) {return Vec(y*b.z-z*b.y,z*b.x-x*b.z,x*b.y-y*b.x);}};

double det(const Vec &a, const Vec &b, const Vec &c);

struct Ray {Vec o, d; double time; Ray(){}; Ray(Vec o_, Vec d_, double t_ = 0) : o(o_), d(d_), time(t_) {}};

struct AABB {Vec min, max; // axis aligned bounding box
	inline void fit(const Vec &p)
	{
		if (p.x<min.x)min.x=p.x; // min
		if (p.y<min.y)min.y=p.y; // min
		if (p.z<min.z)min.z=p.z; // min
		max.x=MAX(p.x, max.x);
		max.y=MAX(p.y, max.y);
		max.z=MAX(p.z, max.z);
	}
	inline void reset() {
		min=Vec(1e20,1e20,1e20); 
		max=Vec(-1e20,-1e20,-1e20);
	}
	inline bool intersect(const Ray &r) const
	{
		Vec invdir = Vec(1 / r.d.x, 1 / r.d.y, 1 / r.d.z);
		int sign[3] = {invdir.x < 0, invdir.y < 0, invdir.z < 0};
		float tmin, tmax, tymin, tymax, tzmin, tzmax;
		Vec bounds[2] = {min, max};
		tmin = (bounds[sign[0]].x - r.o.x) * invdir.x;
		tmax = (bounds[1-sign[0]].x - r.o.x) * invdir.x;
		tymin = (bounds[sign[1]].y - r.o.y) * invdir.y;
		tymax = (bounds[1-sign[1]].y - r.o.y) * invdir.y;
		if ((tmin > tymax) || (tymin > tmax))
			return false;
		if (tymin > tmin)
			tmin = tymin;
		if (tymax < tmax)
			tmax = tymax;
		tzmin = (bounds[sign[2]].z - r.o.z) * invdir.z;
		tzmax = (bounds[1-sign[2]].z - r.o.z) * invdir.z;
		if ((tmin > tzmax) || (tzmin > tmax))
			return false;
		if (tzmin > tmin)
			tmin = tzmin;
		if (tzmax < tmax)
			tmax = tzmax;
		return true;
	}
};

struct HPoint {
	Vec f,pos,nrm,flux,direct,d,fs; 
	double r2; 
	double time;
	unsigned int n; // n = N / ALPHA in the paper
	int pix;
	bool valid;
};

enum Refl_t {PHONG, SPEC, REFR};  // material types, used in radiance()
struct BRDF {
	Refl_t refl;
	Vec sc;
	double k_d, k_s, gloss;
	BRDF(Refl_t refl_, double k_d_ = 1.0, double k_s_ = 0.0, double gloss_ = 1.0, Vec sc_ = Vec(1, 1, 1) * .999)
	: refl(refl_), sc(sc_), k_d(k_d_), k_s(k_s_), gloss(gloss_) {}
};

struct Object {
	Vec p, e, c;
	BRDF brdf;
	Object(Vec p_, Vec e_, Vec c_, BRDF brdf_)
	: p(p_), e(e_), c(c_), brdf(brdf_)
	{
	}
	virtual inline double intersect(const Ray &r, Vec &normal) const = 0;
	virtual Vec samplePos(float *pdf, Vec *normal) const = 0;
};
struct Sphere: Object {
	double rad;
	Vec direction;
	Sphere(double rad_, Vec p_, Vec e_, Vec c_, BRDF brdf_, Vec direction_ = Vec())
	: Object(p_, e_, c_, brdf_), rad(rad_), direction(direction_)
	{}
	Vec updatePos(double time) const
	{
		return p + direction * time;
	}
	inline double intersect(const Ray &r, Vec &normal) const override {
		Vec new_pos = updatePos(r.time);
		Vec op = new_pos - r.o;
		double t, eps = EPS, b = op.dot(r.d), det = b * b - op.dot(op) + rad * rad;
		if (det < 0) return 1e20; else det = sqrt(det);
		if ((t = b - det) < eps && (t = b + det) < eps) return 1e20;
		Vec x = r.o + r.d*t;
		normal = (x - new_pos).norm();
		return t;
	}
	Vec samplePos(float *pdf, Vec *normal) const override
	{
		double z = 1 - 2 * rand01();
		double r = sqrt(MAX((double)0, (double)1 - z * z));
		double phi = 2 * PI * rand01();
		Vec unitSphereSample = Vec(r * cos(phi), r * sin(phi), z);
		*pdf = 1.0 / (4 * PI * rad * rad);
		*normal = unitSphereSample.norm();
		return p + Vec(r * cos(phi), r * sin(phi), z) * rad;
	}
};
struct Triangle {
	Vec a, b, c;
	Triangle(Vec a_, Vec b_, Vec c_) : a(a_), b(b_), c(c_)
	{}
	inline double intersect(const Ray &r, Vec &normal) const {
		Vec ab = b - a;
		Vec ac = c - a;
		Vec ray_dir = r.d;
		Vec pvec = ray_dir % ac;
		double det = ab.dot(pvec);
		if (fabs(det) < EPS) {
			return 1e20;
		}
		double inv_det = 1.0 / det;
		Vec tvec = r.o - a;
		double u = tvec.dot(pvec) * inv_det;
		if (u < 0.0 || u > 1.0) {
			return 1e20;
		}
		Vec qvec = tvec % ab;
		double v = r.d.dot(qvec) * inv_det;
		if (v < 0.0 || u + v > 1.0) {
			return 1e20;
		}
		double t = ac.dot(qvec) * inv_det;
		if (t < EPS) {
			return 1e20;
		}   
			
		return t;
	}
	Vec samplePos(float *pdf, Vec *normal) const
	{
		double su0 = std::sqrt(rand01());
    	double point[2] = {1 - su0, rand01() * su0};
		Vec sample_pos = a * point[0]  + b * point[1] + c * (1 - point[0] - point[1]);
		Vec ab = b - a;
		Vec ac = c - a;
		*normal = (ab % ac);
		*pdf = 1 / (0.5 * (*normal).length());
		*normal = (*normal).norm();
		return sample_pos;
	}
};
struct Trimesh: Object {
	std::vector<Triangle *> triangles;
	int num_triangles;
	AABB bounds;
	Trimesh(std::vector<Triangle *> triangles_, Vec p_, Vec e_, Vec c_, BRDF brdf_)
	: Object(p_, e_, c_, brdf_)
	{
        bounds.reset();
		num_triangles = triangles_.size();
		triangles = triangles_;
		for (int n = 0; n < num_triangles; n++)
		{
			triangles[n]->a = triangles[n]->a + p;
			triangles[n]->b = triangles[n]->b + p;
			triangles[n]->c = triangles[n]->c + p;
			bounds.fit(triangles[n]->a);
			bounds.fit(triangles[n]->b);
			bounds.fit(triangles[n]->c);
		}
	}
	inline double intersect(const Ray &r, Vec &normal) const override {
		double t = 1e20;
		double d = 0;
		Vec norm_temp;
		if (!bounds.intersect(r))
		{
			return 1e20;
		}
		for (int n = 0; n < num_triangles; n++)
		{
			d = (*triangles[n]).intersect(r, norm_temp);
			if (d > EPS && d < t)
			{
				t = d;
				Vec ab = triangles[n]->b - triangles[n]->a;
				Vec ac = triangles[n]->c - triangles[n]->a;
				Vec norm = (ab % ac).norm();
				//if (norm.dot(r.d) > 0) norm = norm * -1;
				norm.norm();    
				normal = norm; 
			}
		}
		return t;
	}
	Vec samplePos(float *pdf, Vec *normal) const override	//assumes that triangles are equal in size
	{
		Vec sample_pos = (*triangles[std::floor(rand01() * num_triangles)]).samplePos(pdf, normal);
		*pdf = *pdf / num_triangles;
		return sample_pos;
	}
};

struct Cylinder : Object {
	double rad, length;
	Cylinder(double rad_, double length_, Vec p_, Vec e_, Vec c_, BRDF brdf_)
	: Object(p_, e_, c_, brdf_), rad(rad_), length(length_)
	{}
	inline double intersect(const Ray &r, Vec &normal) const override {
		double a = r.d.x * r.d.x + r.d.z * r.d.z;
		double b = 2 * (r.o.x*r.d.x + r.o.z*r.d.z);
		double c = r.o.x * r.o.x + r.o.z * r.o.z - rad * rad;
		if (!(b*b - 4*a*c > 0))
		{
			return 1e20;
		}
		double t1 = (-b + sqrt(b*b - 4*a*c)) / (2*a);
		double t2 = (-b - sqrt(b*b - 4*a*c)) / (2*a);
		double t = 1e20;
		double y1 = r.o.y + r.d.y * t1;
		double y2 = r.o.y + r.d.y * t2;
		bool t1_valid = y1 > p.y && y1 < p.y + length && t1 > EPS;
		bool t2_valid = y2 > p.y && y2 < p.y + length && t2 > EPS;
		if (t1_valid && t2_valid)
		{
			t = MIN(t1, t2);
		}
		else if (t1_valid && !t2_valid)
		{
			t = t1;
		}
		else if (t2_valid && !t1_valid)
		{
			t = t2;
		}
		Vec x = r.o + r.d * t;
		normal = (x - Vec(p.x, x.y, p.z)).norm();
		return t;
	}
	Vec samplePos(float *pdf, Vec* normal) const override //NOT IMPLEMENTED YET
	{
		return Vec();
	}
};

class HitPointKDTreeNode {
public:
    HPoint *hitpoint;
    Vec min, max;
    double maxr2;
    HitPointKDTreeNode *ls, *rs;
};
bool cmpHitPointX(HPoint *a, HPoint *b) {
    return a->pos.x < b->pos.x;
}
bool cmpHitPointY(HPoint *a, HPoint *b) {
    return a->pos.y < b->pos.y;
}
bool cmpHitPointZ(HPoint *a, HPoint *b) {
    return a->pos.z < b->pos.z;
}
Vec min(const Vec &a, const Vec &b) {
    return Vec(std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z));
}
Vec max(const Vec &a, const Vec &b) {
    return Vec(std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z));
}
class HitPointKDTree {
    int n;
    HPoint** hitpoints;
    HitPointKDTreeNode* build(int l, int r, int d) {
        HitPointKDTreeNode *p = new HitPointKDTreeNode;
        p->min = Vec(1e100, 1e100, 1e100);
        p->max = p->min * (-1);
        p->maxr2 = 0;
        for (int i = l; i <= r; ++i) {
            p->min = min(p->min, hitpoints[i]->pos);
            p->max = max(p->max, hitpoints[i]->pos);
            p->maxr2 = MAX(p->maxr2, hitpoints[i]->r2);
        }
        int m = (l + r) >> 1;
        if (d == 0) 
            std::nth_element(hitpoints + l, hitpoints + m, hitpoints + r + 1, cmpHitPointX);
        else if (d == 1) 
            std::nth_element(hitpoints + l, hitpoints + m, hitpoints + r + 1, cmpHitPointY);
        else 
            std::nth_element(hitpoints + l, hitpoints + m, hitpoints + r + 1, cmpHitPointZ);
        p->hitpoint = hitpoints[m];
        if (l <= m - 1) p->ls = build(l, m - 1, (d + 1) % 3); else p->ls = nullptr;
        if (m + 1 <= r) p->rs = build(m + 1, r, (d + 1) % 3); else p->rs = nullptr;
        return p;
    }
    void del(HitPointKDTreeNode *p) {
        if (p->ls) del(p->ls);
        if (p->rs) del(p->rs);
        delete p;
    }
public:
    HitPointKDTreeNode *root;
    HitPointKDTree(std::vector<HPoint*> *hitpoints) {
        n = hitpoints->size();
        this->hitpoints = new HPoint*[n];
        for (int i = 0; i < n; ++i)
            this->hitpoints[i] = (*hitpoints)[i];
        root = build(0, n - 1, 0);
    }
    ~HitPointKDTree() {
        if (!root) return;
        del(root);
        delete[] hitpoints;
    }
    void update(HitPointKDTreeNode *p, Vec photon, Vec fl, Ray r, Vec n, const Object &obj) {
        if (!p) return;
        double mind = 0;
        if (photon.x > p->max.x) mind += pow((photon.x - p->max.x), 2);
        if (photon.x < p->min.x) mind += pow((p->min.x - photon.x), 2);
        if (photon.y > p->max.y) mind += pow((photon.y - p->max.y), 2);
        if (photon.y < p->min.y) mind += pow((p->min.y - photon.y), 2);
        if (photon.z > p->max.z) mind += pow((photon.z - p->max.z), 2);
        if (photon.z < p->min.z) mind += pow((p->min.z - photon.z), 2);
        if (mind > p->maxr2) return;
        Vec v = photon - p->hitpoint->pos;
        if (p->hitpoint->valid && (p->hitpoint->nrm.dot(n) > 1e-3) && v.dot(v) <= p->hitpoint->r2 && r.time == p->hitpoint->time) {
            HPoint* hitpoint = p->hitpoint;
            double g = (hitpoint->n*ALPHA+ALPHA) / (hitpoint->n*ALPHA+1);
            hitpoint->r2=hitpoint->r2*g; 
            hitpoint->n++;
            Vec brdf_factor = hitpoint->f * (1 / PI);
            hitpoint->flux=(hitpoint->flux+brdf_factor.mul(fl)) * g;
        }  
        if (p->ls) update(p->ls, photon, fl, r, n, obj);
        if (p->rs) update(p->rs, photon, fl, r, n, obj);
        p->maxr2 = p->hitpoint->r2;
        if (p->ls && p->ls->hitpoint->r2 > p->maxr2)
            p->maxr2 = p->ls->hitpoint->r2;
        if (p->rs && p->rs->hitpoint->r2 > p->maxr2)
            p->maxr2 = p->rs->hitpoint->r2;
    }
};

struct SceneDescription {
	struct Object **scene;
	int scene_size;
	Vec sensor_origin, sensor_direction;
	Object *light;
	double r_0;
	bool direct_sampling;
	bool motion_blur;
	bool dof;
	double S_o, f_stop;
	SceneDescription(struct Object **scene_, int scene_size_, Vec sensor_origin_, Vec sensor_direction_, Object *light_, double r_0_, bool direct_sampling_ = false, bool motion_blur_ = false, bool dof_ = false, double S_o_ = 0.0, double f_stop_ = 1.0)
	: scene(scene_), scene_size(scene_size_), sensor_origin(sensor_origin_), sensor_direction(sensor_direction_.norm()), light(light_), r_0(r_0_), direct_sampling(direct_sampling_), motion_blur(motion_blur_), dof(dof_), S_o(S_o_), f_stop(f_stop_)
	{}
};

double r_0 = 0.0; //initial hitpoint radius
unsigned int num_photon;
HitPointKDTree *hitpoint_kdtree;
std::vector<HPoint *> *hitpoints;
bool direct_sampling = false;
bool motion_blur = false;

Object **scene;
int scene_size;

//SCENES
#define BOX_HALF_X 2.6
#define BOX_HALF_Y 2
#define BOX_HALF_Z 2.8

Vec default_sensor_origin(0, 0.25 * BOX_HALF_Y, 3 * BOX_HALF_Z - 1.0);
Vec default_sensor_direction(0, -0.06, -1);

Object *scene1[] = {
  new Sphere(1e5, Vec( -1e5 -BOX_HALF_X,0,0), Vec(), Vec(.75,.25,.25), BRDF(PHONG)),//Left
  new Sphere(1e5, Vec(1e5 + BOX_HALF_X,0,0),Vec(), Vec(.25,.25,.75), BRDF(PHONG)),//Right
  new Sphere(1e5, Vec(0, 0, -1e5 - BOX_HALF_Z),     Vec(), Vec(.75,.75,.75), BRDF(PHONG)),//Back
  new Sphere(1e5, Vec(0,0,+1e5+ 3 * BOX_HALF_Z - 0.5), Vec(), Vec(),            BRDF(PHONG)),//Front
  new Sphere(1e5, Vec(0, -1e5 - BOX_HALF_Y, 0),    Vec(), Vec(.75,.75,.75), BRDF(PHONG)),//Bottomm
  new Sphere(1e5, Vec(0,1e5 + BOX_HALF_Y, 0),Vec(), Vec(.75,.75,.75), BRDF(PHONG)),//Top
  new Sphere(0.8,Vec(-1.3, -BOX_HALF_Y + 0.8, -1.3),       Vec(), Vec(1,1,1)*.999, BRDF(SPEC)),//Mirror
  new Sphere(0.4, Vec(0,-BOX_HALF_Y + 0.4,-0.8),        Vec(), Vec(.75,.75,.75), BRDF(PHONG)),//Middle
  new Sphere(0.2, Vec(0,BOX_HALF_Y * 0.8,0.0), Vec(1, 1, 1) * 100, Vec(), BRDF(PHONG)) //LIGHT - always at last position
  };
SceneDescription scene_desc1(scene1, sizeof(scene1) / sizeof(Object *), 
default_sensor_origin, default_sensor_direction, scene1[sizeof(scene1) / sizeof(Object *) - 1], 0.04);

SceneDescription scene_descriptors[] = {scene_desc1};

// find the closet interection
inline bool intersect(const Ray &r,double &t,int &id, Vec &normal){
	int n = scene_size; 
	double d, inf = 1e20; t = inf;
	Vec normal_temp;
	for(int i=0;i<n;i++){
		d=scene[i]->intersect(r, normal_temp);
		if(d<t){
			t=d;
			id=i;
			normal = normal_temp.norm();
		}
	}
	return t<inf;
}

bool rayPlaneIntersection(Vec plane_point, Vec plane_normal, Ray r, double *t)
{
  double denom = plane_normal.dot(r.d);
  if (abs(denom) > EPS)
  {
    *t = (plane_point - r.o).dot(plane_normal) / denom;
    if ((*t) >= 0) return true;
  }
  return false;
}
Vec uniformSampleDisk(double radius, Vec center)
{
  double sample_r = sqrt(radius * radius * rand01());
  double sample_theta = rand01() * 2.0 * PI;
  Vec sample = Vec();
  sample.x = center.x + sample_r * cos(sample_theta);
  sample.y = center.y + sample_r * sin(sample_theta);
  sample.z = center.z;
  return sample;
}

Vec concentricDiskSample()
{
	double u1 = rand01();
	double u2 = rand01();
	double uOffsetX = 2.f * u1 - 1;
	double uOffsetY = 2.f * u2 - 1;
	if (uOffsetX == 0 && uOffsetY == 0) return Vec(0, 0);
	double theta, r;
	if (abs(uOffsetX) > abs(uOffsetY))
	{
		r = uOffsetX;
		theta = (PI / 4) * (uOffsetY / uOffsetX);
	}
	else
	{
		r = uOffsetY;
		theta = PI / 2 - (PI / 4) * (uOffsetX / uOffsetY);
	}
	return Vec(cos(theta), sin(theta)) * r;
}
Vec lightSampleDirect(Ray r, Vec pos, Vec norm, int hit_id, const Object &hit_obj, bool specular = false) //assumes hit_obj has PHONG BRDF
{
	Vec direct_light = Vec();
	for (int i=0; i<scene_size; i++){
        const Object &l_obj = *(scene[i]);
        if ((l_obj.e.x<=0 && l_obj.e.y<=0 && l_obj.e.z<=0) || i==hit_id) continue; // skip non-lights
		float pos_pdf;
		Vec light_normal;
		Vec sample_pos = l_obj.samplePos(&pos_pdf, &light_normal);
		Vec sample_dir = (sample_pos - pos).norm();
		double cos_theta_light = light_normal.dot((sample_dir * -1));
		Vec sampled_light = (cos_theta_light > 0.0) ? l_obj.e : Vec();
		if ((sampled_light.x <= 0 && sampled_light.y <= 0 && sampled_light.z <= 0) || pos_pdf <= 0.0) continue;
		Vec temp_n; double t; int id = -1;
		if (intersect(Ray(pos, sample_dir), t, id, temp_n) && id==i){  // shadow ray
			double dist = (sample_pos-pos).length();
			Vec brdf_factor = hit_obj.c * (1 / PI); //only accounting for the diffuse part
			if (specular)
			{
				Vec perf_refl = r.d-norm*2*norm.dot(r.d);
				brdf_factor = hit_obj.brdf.sc * pow(MAX(perf_refl.dot(sample_dir), 0), hit_obj.brdf.gloss) * (hit_obj.brdf.gloss + 2.0) * (1 / (2 * PI));
			}
			direct_light = direct_light + brdf_factor.mul(sampled_light*MAX(sample_dir.dot(norm), 0)) * (1.0 / pos_pdf) * (cos_theta_light / (dist * dist)); 
		}
	}
	return direct_light;
}
void lightSampleE(Object* light, Ray* r, Vec* f)
{
	float pos_pdf;
	Vec light_normal;
	Vec sample_pos = light->samplePos(&pos_pdf, &light_normal);
	Vec disk_sample = concentricDiskSample();
	double z = sqrt(MAX(0.0, 1 - disk_sample.x * disk_sample.x - disk_sample.y * disk_sample.y));
	Vec sample_dir = Vec(disk_sample.x, disk_sample.y, z);
	Vec v1, v2, n;
	n = light_normal;
	v1 = ((fabs(n.x)>.1?Vec(0,1):Vec(1))%n).norm();
	v2 = n%v1;
	sample_dir = v1 * sample_dir.x + v2 * sample_dir.y + n * sample_dir.z;
	*r = Ray(sample_pos, sample_dir);
	float dir_pdf = MAX(light_normal.dot(sample_dir), 0) / PI;
	if (dir_pdf <= 0.0) 
	{
		*f = Vec();
		return;
	}
	*f = light->e * MAX(light_normal.dot(sample_dir), 0) * (1.0/pos_pdf) * (1.0/dir_pdf);
}

void saveImage(std::vector<HPoint *> *hitpoints, int w, int h, int iterations, int photons_per_pass, double elapsed_time, int scene_nr, std::string folder_name, bool sppm)
{
	Vec *c = new Vec[w * h];
	for (auto hp : *hitpoints) {
		int i = hp->pix;
		c[i]=c[i]+hp->flux*(1.0/(PI*hp->r2*iterations*photons_per_pass)) + hp->direct / iterations;
	}
	std::string time_str = std::string("");
	std::stringstream stream;
	stream << std::fixed << std::setprecision(2) << elapsed_time;
	time_str = stream.str() + std::string("s");
	FILE* file = fopen((folder_name + "/" + time_str + "_" + std::to_string(iterations) + "r" + ".ppm").c_str(), "w"); fprintf(file,"P3\n%d %d\n%d\n",w,h,255);
	for(int i = w * h; i--;) {
		fprintf(file,"%d %d %d ", toInt(c[i].x), toInt(c[i].y), toInt(c[i].z));
	}
	fclose(file);
}

void trace(const Ray &r,int dpt,bool m,const Vec &fl,const Vec &adj, HPoint* hitp = nullptr, bool emissive = true) 
{
	double t;
	int id = 0; 
	dpt++;
	Vec n;
	if(!intersect(r,t,id,n)||(dpt>=20))return;
	const Object &obj = *(scene[id]); 
	Vec x=r.o+r.d*t, f=obj.c;
	Vec nl=n.dot(r.d)<0?n:n*-1; 
	double p=f.x>f.y&&f.x>f.z?f.x:f.y>f.z?f.y:f.z;

	if (obj.brdf.refl == PHONG) { //https://www.cs.princeton.edu/courses/archive/fall03/cs526/papers/lafortune94.pdf
		double phong_action = rand01();
		int chosen_action = 0;
		Vec spec_sample;
		Vec perf_refl = (r.d)-nl*2.0*nl.dot(r.d);
		perf_refl.norm();
		if (phong_action < obj.brdf.k_d)
		{
			chosen_action = 0;
		}
		else if (phong_action < obj.brdf.k_d + obj.brdf.k_s)
		{
			chosen_action = 1;
			double r1 = rand01();
			double cosa = pow(r1, (1 / (obj.brdf.gloss + 1)));
			double sina = sqrt(1 - pow(r1, (2 / (obj.brdf.gloss + 1))));
			double phi = rand01() * 2 * PI;
			Vec w=perf_refl,u=((fabs(w.x)>.1?Vec(0,1):Vec(1))%w).norm();
			Vec v=w%u;
			spec_sample = (w * cosa + u * sina * cos(phi) + v * sina * sin(phi)).norm();
		}
		else 
		{
			return;
		}
		if (m) {
			if (chosen_action == 0)
			{
				hitp->f=f.mul(adj); 
				hitp->pos=x;
				hitp->nrm=n; 
				hitp->valid = true;
				hitp->time = r.time;
				if ((obj.e.x > 0 || obj.e.y > 0 || obj.e.z > 0) && emissive)
				{
					if (emissive) hitp->direct = hitp->direct + adj.mul(obj.e);
					hitp->valid = false;
				}	
				if (direct_sampling)
				{
					hitp->direct = hitp->direct + adj.mul(lightSampleDirect(r, x, nl, id, obj));
				}			
			}
			else if (chosen_action == 1)
			{
				double norm_factor = clamp(((obj.brdf.gloss + 2) / (obj.brdf.gloss + 1)) * MAX(nl.dot(spec_sample), 0.0));
				hitp->direct = hitp->direct + adj.mul(lightSampleDirect(r, x, nl, id, obj, true));	
				trace(Ray(x, spec_sample, r.time), dpt, m, fl, obj.brdf.sc.mul(adj) * norm_factor, hitp, false);
			}
		} 
		else 
		{
			if (dpt > 1 || !direct_sampling) //ignore direct light, was already accounted for in the measurement point
			{
                hitpoint_kdtree->update(hitpoint_kdtree->root, x, fl, r, n, obj);
			}
			Vec new_d;
			if (chosen_action == 0)
			{
				double r1=2.*PI*rand01(),r2=rand01();
				double r2s=sqrt(r2);
				Vec w=nl,u=((fabs(w.x)>.1?Vec(0,1):Vec(1))%w).norm();
				Vec v=w%u; 
				new_d = (u*cos(r1)*r2s + v*sin(r1)*r2s + w*sqrt(1-r2)).norm();
			}
			else if (chosen_action == 1)
			{
				new_d = spec_sample;
			}
			double spec_pdf, diff_pdf;
			diff_pdf = MAX(nl.dot(new_d), 0.0) / PI;
			spec_pdf = ((obj.brdf.gloss + 1) / (2.0 * PI)) * pow(MAX(perf_refl.dot(new_d), 0.0), obj.brdf.gloss);
			double pdf = obj.brdf.k_d * diff_pdf + obj.brdf.k_s * spec_pdf;
			Vec brdf_f = Vec();
			brdf_f = brdf_f + obj.brdf.sc * obj.brdf.k_s * ((obj.brdf.gloss + 2) / (2 * PI)) * pow(MAX(perf_refl.dot(new_d), 0), obj.brdf.gloss);
			brdf_f = brdf_f + obj.c * obj.brdf.k_d * (1 / PI);
			if (rand01()<p) trace(Ray(x,new_d, r.time),dpt,m,brdf_f.mul(fl) * (1./p) * MAX(nl.dot(new_d), 0) * (1/pdf),adj);
		}
	} else if (obj.brdf.refl == SPEC) {
		Vec dr = r.d-n*2.0*n.dot(r.d);
		trace(Ray(x, dr, r.time), dpt, m, f.mul(fl), f.mul(adj), hitp);
	} else {
		Vec dr = r.d-n*2.0*n.dot(r.d);
		Ray lr(x,dr, r.time); 
		bool into = (n.dot(nl)>0.0);
		double nc = 1.0, nt=1.5, nnt = into?nc/nt:nt/nc, ddn = r.d.dot(nl), cos2t;
		if ((cos2t=1-nnt*nnt*(1-ddn*ddn))<0) return trace(lr,dpt,m,fl,adj,hitp);
		Vec td = (r.d*nnt - n*((into?1:-1)*(ddn*nnt+sqrt(cos2t)))).norm();
		double a=nt-nc, b=nt+nc, R0=a*a/(b*b), c = 1-(into?-ddn:td.dot(n));
		double Re=R0+(1-R0)*c*c*c*c*c,P=Re;Ray rr(x,td, r.time);Vec fa=f.mul(adj);
		if (m) {
			(rand01()<P)?trace(lr,dpt,m,fl,fa,hitp):trace(rr,dpt,m,fl,fa,hitp);
		} else {
			(rand01()<P)?trace(lr,dpt,m,fl,fa):trace(rr,dpt,m,fl,fa);
		}
	}
}

int main(int argc, char *argv[]) {
	int photons_per_pass = 1000000;
	int w=960, h=720, samps = (argc>=2) ? MAX(atoi(argv[1]),1) : 20;
	std::vector<int> checkpoints = {10, 100, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000};
	size_t scene_nr = (argc>=3) ? atoi(argv[2]) : 1;
	double max_time = (argc==4) ? atof(argv[3]) : 0.0;
	Vec sensor_origin, sensor_direction;
	double sensor_width = 0.036, sensor_height = 0.024; //4:3
	double S_i = 0.03;

	SceneDescription curr_scene_desc = scene_descriptors[scene_nr - 1];
	scene = curr_scene_desc.scene;
	scene_size = curr_scene_desc.scene_size;
	sensor_origin = curr_scene_desc.sensor_origin;
	sensor_direction = curr_scene_desc.sensor_direction;
	Object *light = curr_scene_desc.light;
	r_0 = curr_scene_desc.r_0;
	direct_sampling = curr_scene_desc.direct_sampling;
	motion_blur = curr_scene_desc.motion_blur;
	bool dof = curr_scene_desc.dof;
	double S_o = curr_scene_desc.S_o;
	double f_stop = curr_scene_desc.f_stop;
	Vec cx=Vec(w*sensor_height/h), cy=(cx%sensor_direction).norm()*sensor_height, r, vw;
	double f = (S_i * S_o) / (S_i + S_o);
	double aperture_diameter = f / f_stop;
	double aperture_r = aperture_diameter / 2.0;

	namespace fs = std::filesystem;
	std::stringstream radius_stream;
	radius_stream << std::fixed << std::setprecision(4) << r_0;
	std::string radius_string = radius_stream.str();
	auto time_stamp = std::time(nullptr);
	auto time_stamp_m = *std::localtime(&time_stamp);
	std::ostringstream oss;
	oss << std::put_time(&time_stamp_m, "%Y-%m-%d_%H-%M-%S");
	std::string date_time_string = oss.str();
	std::string folder_name = ("./output/sppm_scene_" + std::to_string(scene_nr) + "_" + radius_string + "_" + date_time_string).c_str();
	fs::create_directories(folder_name);
	printf("Rendering Scene %lld\n", scene_nr);
	auto tstart = std::chrono::system_clock::now();

	hitpoints = new std::vector<HPoint*>;
	for (int y=0; y<h; y++){
		for (int x=0; x<w; x++) {
			HPoint *hitpoint = new HPoint;
			hitpoint->flux = Vec();
			hitpoint->direct = Vec();
			hitpoint->r2 = r_0 * r_0;
			hitpoint->n = 0;
			hitpoint->pix = x + y * w;
			hitpoints->push_back(hitpoint);
		}
	}
	for (int round = 0; round < samps; round++)
	{	
		double time_sample = rand01();
		#pragma omp parallel for schedule(dynamic, 1)	//RAY TRACING PASS
		for (int y=0; y<h; y++){
			for (int x=0; x<w; x++) {
				HPoint *curr_hp = hitpoints->at(y * w + x);
				curr_hp->valid = false;
				curr_hp->f = Vec(1, 1, 1);
				double offset_x = rand01() - 0.5;
				double offset_y = rand01() - 0.5;
				Vec sensor_sample = cx * ((x + 0.5 + offset_x) / w - 0.5) + cy * (-(y + 0.5 + offset_y) / h + 0.5);
				sensor_sample = sensor_sample + sensor_origin;
				Vec lens_center = sensor_origin + sensor_direction * S_i;
				Vec d, start_point;
				if (dof)
				{
					Ray center_ray(lens_center, (lens_center - sensor_sample).norm());
					Vec focus_plane_point = lens_center + sensor_direction * S_o;
					Vec focus_plane_normal = sensor_direction * -1;
					double t = 0.0;
					if (!rayPlaneIntersection(focus_plane_point, focus_plane_normal, center_ray, &t)) printf("Something went very wrong!\n");
					Vec focus_point = center_ray.o + center_ray.d * t;
					Vec lens_sample = uniformSampleDisk(aperture_r, lens_center);
					start_point = lens_sample;
					d = (focus_point - lens_sample).norm();
				}
				else
				{
					start_point = lens_center;
					d = (lens_center - sensor_sample).norm();
				}
				double time = 0.0;
				if (motion_blur)
					time = time_sample;
				trace(Ray(start_point, d, time), 0, true, Vec(), Vec(1, 1, 1), curr_hp);
			}
		}

		 if (hitpoint_kdtree)
		 	delete hitpoint_kdtree;
		hitpoint_kdtree = new HitPointKDTree(hitpoints);
		vw=Vec(1,1,1);

		#pragma omp parallel for schedule(dynamic, 1)	//PHOTON PASS
		for(int j=0;j<photons_per_pass;j++){
			Ray r;
			Vec f;
			lightSampleE(light, &r, &f);
			if (motion_blur)
				r.time = time_sample;
			trace(r,0,0>1,f,vw);
		}
        fprintf(stderr, "\rFinished Round %d/%d", round + 1, samps);
		
		auto tround = std::chrono::system_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(tround - tstart).count();
		if (max_time > EPS && elapsed >= max_time) //Automatically stop after max_time elapsed
		{
			fprintf(stderr, "\nElapsed Time: %f\nFinished after %d rounds\n", elapsed, round + 1);
			saveImage(hitpoints, w, h, round + 1, photons_per_pass, elapsed, scene_nr, folder_name, true);
			break;
		}
		if (std::find(checkpoints.begin(), checkpoints.end(), round + 1) != checkpoints.end() || (round + 1) == samps)
		{
			saveImage(hitpoints, w, h, round + 1, photons_per_pass, elapsed, scene_nr, folder_name, true);
		}
	}
	delete hitpoints;
	delete hitpoint_kdtree;
}
