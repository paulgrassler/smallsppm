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
#define ALPHA ((double)(2.0 / 3.0))	  // the alpha parameter of SPPM
#define PI ((double)3.14159265358979)
#define EPS ((double)0.00000001)

inline double clamp(double x) { return x < 0 ? 0 : x > 1 ? 1 : x; }
int toInt(double x) { return int(pow(1 - exp(-x), 1 / 2.2) * 255 + .5); }
double rand01() {
	static thread_local std::default_random_engine generator(omp_get_thread_num() +std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
	static std::uniform_real_distribution<double> distr(0.0, 1.0);
	return distr(generator);
}

class Vec {
public:
	double x, y, z;
	Vec(double x_ = 0, double y_ = 0, double z_ = 0) : x(x_), y(y_), z(z_) {}
	inline Vec operator+(const Vec &b) const { return Vec(x + b.x, y + b.y, z + b.z); }
	inline Vec operator-(const Vec &b) const { return Vec(x - b.x, y - b.y, z - b.z); }
	inline Vec operator+(double b) const { return Vec(x + b, y + b, z + b); }
	inline Vec operator-(double b) const { return Vec(x - b, y - b, z - b); }
	inline Vec operator*(double b) const { return Vec(x * b, y * b, z * b); }
	inline Vec operator/(double b) const { return Vec(x / b, y / b, z / b); }
	inline Vec operator/(Vec b) const { return Vec(x / b.x, y / b.y, z / b.z); }
	inline double operator[](int i) const { return (i == 0) ? x : (i == 1) ? y : z; }
	inline double &operator[](int i) { return (i == 0) ? x : (i == 1) ? y : z; }
	inline Vec mul(const Vec &b) const { return Vec(x * b.x, y * b.y, z * b.z); }
	inline Vec norm() { return (*this) * (1.0 / sqrt(x * x + y * y + z * z)); }
	inline double length() const { return sqrt(x * x + y * y + z * z); }
	inline double dot(const Vec &b) const { return x * b.x + y * b.y + z * b.z; }
	inline Vec operator%(Vec &b) { return Vec(y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.x); }
};

class Ray {
public:
	Vec o, d;
	double time;
	Ray() {};
	Ray(Vec o_, Vec d_, double t_ = 0) : o(o_), d(d_), time(t_) {};
};

class AABB {
public:
	Vec min, max;
	AABB() {};
	AABB(Vec min_, Vec max_) : min(min_), max(max_) {};
	void fit(const Vec &p) {
		min = Vec(MIN(min.x, p.x), MIN(min.y, p.y), MIN(min.z, p.z));
		max = Vec(MAX(max.x, p.x), MAX(max.y, p.y), MAX(max.z, p.z));
	}
	void reset() {
		min = Vec(1e20, 1e20, 1e20);
		max = Vec(-1e20, -1e20, -1e20);
	}
	bool intersect(const Ray &r) const {
		Vec invdir = Vec(1 / r.d.x, 1 / r.d.y, 1 / r.d.z);
		int sign[3] = {invdir.x < 0, invdir.y < 0, invdir.z < 0};
		float tmin, tmax, tymin, tymax, tzmin, tzmax;
		Vec bounds[2] = {min, max};
		tmin = (bounds[sign[0]].x - r.o.x) * invdir.x;
		tmax = (bounds[1 - sign[0]].x - r.o.x) * invdir.x;
		tymin = (bounds[sign[1]].y - r.o.y) * invdir.y;
		tymax = (bounds[1 - sign[1]].y - r.o.y) * invdir.y;
		if ((tmin > tymax) || (tymin > tmax)) return false;
		if (tymin > tmin) tmin = tymin;
		if (tymax < tmax) tmax = tymax;
		tzmin = (bounds[sign[2]].z - r.o.z) * invdir.z;
		tzmax = (bounds[1 - sign[2]].z - r.o.z) * invdir.z;
		if ((tmin > tzmax) || (tzmin > tmax)) return false;
		if (tzmin > tmin) tmin = tzmin;
		if (tzmax < tmax) tmax = tzmax;
		return true;
	}
};

enum Refl_t { PHONG, SPEC, REFR }; // material types

class BRDF {
public:
	virtual Refl_t getReflType() = 0;
	virtual void evaluateBRDF(const Vec &n, const Vec d, const Vec &c, const Vec &new_d, Vec *brdf_f) = 0;
	virtual void sampleDir(const Vec &n, const Vec d, Vec *new_d, double *pdf) = 0;
	virtual bool isSpecular() = 0;
	virtual bool isGlossy() = 0;
};

class BRDF_PHONG : public BRDF{
private:
	double k_d, k_s, gloss;
public:
	BRDF_PHONG(double k_d_ = 1.0, double k_s_ = 0.0, double gloss_ = 1.0) : k_d(k_d_), k_s(k_s_), gloss(gloss_) {};
	Refl_t getReflType() override { return PHONG; }
	void evaluateBRDF(const Vec &n, const Vec d, const Vec &c, const Vec &new_d, Vec *brdf_f) override {
		Vec perf_refl = ((d)-n * 2.0 * n.dot(d)).norm();
		*brdf_f = (*brdf_f) + Vec(1, 1, 1) * k_s * ((gloss + 2) / (2 * PI)) * pow(MAX(perf_refl.dot(new_d), 0), gloss);
		*brdf_f = (*brdf_f) + c * k_d * (1 / PI);
	}
	void sampleDir(const Vec &n, const Vec d, Vec *new_d, double *pdf) { // https://www.cs.princeton.edu/courses/archive/fall03/cs526/papers/lafortune94.pdf
		double phong_action = rand01();
		Vec perf_refl = ((d)-n * 2.0 * n.dot(d)).norm();
		if (phong_action < k_d) {
			double r1 = 2. * PI * rand01(), r2 = rand01();
			double r2s = sqrt(r2);
			Vec w = n, u = ((fabs(w.x) > .1 ? Vec(0, 1) : Vec(1)) % w).norm();
			Vec v = w % u;
			*new_d = (u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrt(1 - r2)).norm();
		}
		else if (phong_action < k_d + k_s) {
			double r1 = rand01();
			double cosa = pow(r1, (1 / (gloss + 1)));
			double sina = sqrt(1 - pow(r1, (2 / (gloss + 1))));
			double phi = rand01() * 2 * PI;
			Vec w = perf_refl, u = ((fabs(w.x) > .1 ? Vec(0, 1) : Vec(1)) % w).norm();
			Vec v = w % u;
			*new_d = (w * cosa + u * sina * cos(phi) + v * sina * sin(phi)).norm();
		}
		double diff_pdf = abs(n.dot(*new_d)) / PI;
		double spec_pdf = ((gloss + 1) / (2.0 * PI)) * pow(MAX(perf_refl.dot(*new_d), 0.0), gloss);
		if (n.dot(*new_d) < 0.0) *pdf = 0.0;
		else *pdf = k_d * diff_pdf + k_s * spec_pdf;
	}
	bool isSpecular() override { return false; }
	bool isGlossy() override { return k_s > 0.1; }
};

class BRDF_SPEC : public BRDF {
public:
	Refl_t getReflType() override { return SPEC; }
	void evaluateBRDF(const Vec &n, const Vec d, const Vec &c, const Vec &new_d, Vec *brdf_f) override {
		*brdf_f = Vec(1, 1, 1).mul(c) / abs(n.dot(new_d));
	}
	void sampleDir(const Vec &n, const Vec d, Vec *new_d, double *pdf) {
		Vec perf_refl = (d)-n * 2.0 * n.dot(d);
		*new_d = perf_refl.norm();
		*pdf = 1.0;
	}
	bool isSpecular() override { return true; }
	bool isGlossy() override { return false; }
};

class BRDF_REFR : public BRDF {
private:
	double ior;
public:
	BRDF_REFR(double ior_ = 1.5) : ior(ior_) {};
	Refl_t getReflType() override { return REFR; }
	void evaluateBRDF(const Vec &n, const Vec d, const Vec &c, const Vec &new_d, Vec *brdf_f) override {
		*brdf_f = Vec(1, 1, 1).mul(c) / abs(n.dot(new_d));
	}

	void sampleDir(const Vec &n, const Vec d, Vec *new_d, double *pdf) {
		Vec perf_refl = ((d)-n * 2.0 * n.dot(d)).norm();
		Vec nl = n.dot(d) < 0 ? n : n * -1;
		bool into = (n.dot(nl) > 0.0);
		double nc = 1.0, nt = ior, nnt = into ? nc / nt : nt / nc, ddn = d.dot(nl), cos2t;
		if ((cos2t = 1 - nnt * nnt * (1 - ddn * ddn)) < 0) { // total internal reflection
			*new_d = perf_refl;
			*pdf = 1.0;
			return;
		}
		Vec perf_trans = (d * nnt - n * ((into ? 1 : -1) * (ddn * nnt + sqrt(cos2t)))).norm();
		double a_ = nt - nc, b_ = nt + nc, R0 = a_ * a_ / (b_ * b_), c_ = 1 - (into ? -ddn : perf_trans.dot(n));
		double Re = R0 + (1 - R0) * c_ * c_ * c_ * c_ * c_, P = Re;
		if (rand01() < P) { // reflection
			*new_d = perf_refl;
			*pdf = 1.0;
		}
		else {// transmission
			*new_d = perf_trans;
			*pdf = 1.0;
		}
	}
	bool isSpecular() override { return true; }
	bool isGlossy() override { return false; }
};

class Material {
public:
	Vec c;
	BRDF *brdf;
	Material(Vec c_, BRDF *brdf_) : c(c_), brdf(brdf_) {};
};

struct HPoint {
	Vec f, pos, nrm, flux, direct, d, fs;
	BRDF *brdf;
	double r2;
	double time;
	unsigned int n; // n = N / ALPHA in the paper
	int pix;
	bool valid;
};

class Object {
public:
	Vec p, e;
	Material mat;
	Object(Vec p_, Vec e_, Material mat_) : p(p_), e(e_), mat(mat_) {}
	virtual double intersect(const Ray &r, Vec &normal) const = 0;
	virtual Vec samplePos(float *pdf, Vec *normal) const = 0;
};
class Sphere : public Object {
private:
	double rad;
	Vec direction;
public:
	Sphere(double rad_, Vec p_, Vec e_, Material mat_, Vec direction_ = Vec()) : Object(p_, e_, mat_), rad(rad_), direction(direction_) {}
	Vec updatePos(double time) const {
		return p + direction * time;
	}
	double intersect(const Ray &r, Vec &normal) const override {
		Vec new_pos = r.time == 0.0 ? p : updatePos(r.time);
		Vec op = new_pos - r.o;
		double t, eps = EPS, b = op.dot(r.d), det = b * b - op.dot(op) + rad * rad;
		if (det < 0) return 1e20;
		else det = sqrt(det);
		if ((t = b - det) < eps && (t = b + det) < eps) return 1e20;
		Vec x = r.o + r.d * t;
		normal = (x - new_pos).norm();
		return t;
	}
	Vec samplePos(float *pdf, Vec *normal) const override {
		double z = 1 - 2 * rand01();
		double r = sqrt(MAX((double)0, (double)1 - z * z));
		double phi = 2 * PI * rand01();
		Vec unitSphereSample = Vec(r * cos(phi), r * sin(phi), z);
		*pdf = 1.0 / (4 * PI * rad * rad);
		*normal = unitSphereSample.norm();
		return p + Vec(r * cos(phi), r * sin(phi), z) * rad;
	}
};
class Triangle {
public:
	Vec a, b, c;
	Triangle(Vec a_, Vec b_, Vec c_) : a(a_), b(b_), c(c_) {}
	double intersect(const Ray &r, Vec &normal) const {
		Vec ab = b - a;
		Vec ac = c - a;
		Vec ray_dir = r.d;
		Vec pvec = ray_dir % ac;
		double det = ab.dot(pvec);
		if (fabs(det) < EPS) return 1e20;
		double inv_det = 1.0 / det;
		Vec tvec = r.o - a;
		double u = tvec.dot(pvec) * inv_det;
		if (u < 0.0 || u > 1.0) return 1e20;
		Vec qvec = tvec % ab;
		double v = r.d.dot(qvec) * inv_det;
		if (v < 0.0 || u + v > 1.0) return 1e20;
		double t = ac.dot(qvec) * inv_det;
		if (t < EPS) return 1e20;
		return t;
	}
	Vec samplePos(float *pdf, Vec *normal) const {
		double su0 = std::sqrt(rand01());
		double point[2] = {1 - su0, rand01() * su0};
		Vec sample_pos = a * point[0] + b * point[1] + c * (1 - point[0] - point[1]);
		Vec ab = b - a;
		Vec ac = c - a;
		*normal = (ab % ac);
		*pdf = 1 / (0.5 * (*normal).length());
		*normal = (*normal).norm();
		return sample_pos;
	}
};
struct Trimesh : public Object {
private:
	std::vector<Triangle *> triangles;
	int num_triangles;
	AABB bounds;
public:
	Trimesh(std::vector<Triangle *> triangles_, Vec p_, Vec e_, Material mat_) : Object(p_, e_, mat_) {
		bounds.reset();
		num_triangles = triangles_.size();
		triangles = triangles_;
		for (int n = 0; n < num_triangles; n++) {
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
		if (!bounds.intersect(r)) return 1e20;
		for (int n = 0; n < num_triangles; n++) {
			d = (*triangles[n]).intersect(r, norm_temp);
			if (d > EPS && d < t) {
				Vec ab = triangles[n]->b - triangles[n]->a;
				Vec ac = triangles[n]->c - triangles[n]->a;
				Vec norm = (ab % ac).norm();
				t = d;
				normal = norm;
			}
		}
		return t;
	}
	Vec samplePos(float *pdf, Vec *normal) const override { // assumes that triangles are equal in size
		Vec sample_pos = (*triangles[std::floor(rand01() * num_triangles)]).samplePos(pdf, normal);
		*pdf = *pdf / num_triangles;
		return sample_pos;
	}
};

struct Cylinder : public Object {
private:
	double rad, length;
public:
	Cylinder(double rad_, double length_, Vec p_, Vec e_, Material mat_) : Object(p_, e_, mat_), rad(rad_), length(length_) {}
	double intersect(const Ray &r, Vec &normal) const override {
		double a = r.d.x * r.d.x + r.d.z * r.d.z;
		double b = 2 * (r.o.x * r.d.x + r.o.z * r.d.z);
		double c = r.o.x * r.o.x + r.o.z * r.o.z - rad * rad;
		if (!(b * b - 4 * a * c > 0)) return 1e20;
		double t1 = (-b + sqrt(b * b - 4 * a * c)) / (2 * a);
		double t2 = (-b - sqrt(b * b - 4 * a * c)) / (2 * a);
		double t = 1e20;
		double y1 = r.o.y + r.d.y * t1;
		double y2 = r.o.y + r.d.y * t2;
		bool t1_valid = y1 > p.y && y1 < p.y + length && t1 > EPS;
		bool t2_valid = y2 > p.y && y2 < p.y + length && t2 > EPS;
		if (t1_valid && t2_valid) t = MIN(t1, t2);
		else if (t1_valid && !t2_valid) t = t1;
		else if (t2_valid && !t1_valid) t = t2;
		Vec x = r.o + r.d * t;
		normal = (x - Vec(p.x, x.y, p.z)).norm();
		return t;
	}
	Vec samplePos(float *pdf, Vec *normal) const override { // NOT IMPLEMENTED YET
		printf("Cylinder sampling not implemented yet\n");
		exit(-1);
	}
};

class SceneDescription {
public:
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
		: scene(scene_), scene_size(scene_size_), sensor_origin(sensor_origin_), sensor_direction(sensor_direction_.norm()), light(light_), r_0(r_0_), direct_sampling(direct_sampling_), motion_blur(motion_blur_), dof(dof_), S_o(S_o_), f_stop(f_stop_) {}
};

//This kd-tree iplementation is from https://github.com/shizhouxing/SPPM/
class HitPointKDTreeNode {
public:
	HPoint *hitpoint;
	Vec min, max;
	double maxr2;
	HitPointKDTreeNode *ls, *rs;
};
bool cmpHitPointX(HPoint *a, HPoint *b) { return a->pos.x < b->pos.x; }
bool cmpHitPointY(HPoint *a, HPoint *b) { return a->pos.y < b->pos.y; }
bool cmpHitPointZ(HPoint *a, HPoint *b) { return a->pos.z < b->pos.z; }
Vec min(const Vec &a, const Vec &b) { return Vec(std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z)); }
Vec max(const Vec &a, const Vec &b) { return Vec(std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z)); }
class HitPointKDTree {
private:
	int n;
	HPoint **hitpoints;
	HitPointKDTreeNode *build(int l, int r, int d) {
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
		if (d == 0) std::nth_element(hitpoints + l, hitpoints + m, hitpoints + r + 1, cmpHitPointX);
		else if (d == 1) std::nth_element(hitpoints + l, hitpoints + m, hitpoints + r + 1, cmpHitPointY);
		else std::nth_element(hitpoints + l, hitpoints + m, hitpoints + r + 1, cmpHitPointZ);
		p->hitpoint = hitpoints[m];
		if (l <= m - 1) p->ls = build(l, m - 1, (d + 1) % 3);
		else p->ls = nullptr;
		if (m + 1 <= r) p->rs = build(m + 1, r, (d + 1) % 3);
		else p->rs = nullptr;
		return p;
	}
	void del(HitPointKDTreeNode *p) {
		if (p->ls) del(p->ls);
		if (p->rs) del(p->rs);
		delete p;
	}

public:
	HitPointKDTreeNode *root;
	HitPointKDTree(std::vector<HPoint *> *hitpoints) {
		n = hitpoints->size();
		this->hitpoints = new HPoint *[n];
		for (int i = 0; i < n; ++i)
			this->hitpoints[i] = (*hitpoints)[i];
		root = build(0, n - 1, 0);
	}
	~HitPointKDTree() {
		if (!root) return;
		del(root);
		delete[] hitpoints;
	}
	void update(HitPointKDTreeNode *p, Vec photon_pos, const Vec fl, const Ray r, const Vec n, const Object &obj) {
		if (!p) return;
		double mind = 0;
		if (photon_pos.x > p->max.x) mind += pow((photon_pos.x - p->max.x), 2);
		if (photon_pos.x < p->min.x) mind += pow((p->min.x - photon_pos.x), 2);
		if (photon_pos.y > p->max.y) mind += pow((photon_pos.y - p->max.y), 2);
		if (photon_pos.y < p->min.y) mind += pow((p->min.y - photon_pos.y), 2);
		if (photon_pos.z > p->max.z) mind += pow((photon_pos.z - p->max.z), 2);
		if (photon_pos.z < p->min.z) mind += pow((p->min.z - photon_pos.z), 2);
		if (mind > p->maxr2) return;
		Vec v = photon_pos - p->hitpoint->pos;
		if (p->hitpoint->valid && (p->hitpoint->nrm.dot(n) > 1e-3) && v.dot(v) <= p->hitpoint->r2 && r.time == p->hitpoint->time) {
			HPoint *hitpoint = p->hitpoint;
			double g = (hitpoint->n * ALPHA + ALPHA) / (hitpoint->n * ALPHA + 1);
			hitpoint->r2 = hitpoint->r2 * g;
			hitpoint->n++;
			Vec brdf_factor;
			hitpoint->brdf->evaluateBRDF(hitpoint->nrm, hitpoint->d, hitpoint->f, r.d * -1, &brdf_factor);
			hitpoint->flux = (hitpoint->flux + brdf_factor.mul(fl)) * g;
		}
		if (p->ls) update(p->ls, photon_pos, fl, r, n, obj);
		if (p->rs) update(p->rs, photon_pos, fl, r, n, obj);
		p->maxr2 = p->hitpoint->r2;
		if (p->ls && p->ls->hitpoint->r2 > p->maxr2) p->maxr2 = p->ls->hitpoint->r2;
		if (p->rs && p->rs->hitpoint->r2 > p->maxr2) p->maxr2 = p->rs->hitpoint->r2;
	}
};

//Global Class Instances
HitPointKDTree *hitpoint_kdtree;
SceneDescription *curr_scene_desc;

inline bool intersect(const Ray &r, double &t, int &id, Vec &normal) { // find the closest scene intersection
	int n = curr_scene_desc->scene_size;
	double d, inf = 1e20;
	t = inf;
	Vec normal_temp;
	for (int i = 0; i < n; i++) {
		d = curr_scene_desc->scene[i]->intersect(r, normal_temp);
		if (d < t) {
			t = d;
			id = i;
			normal = normal_temp.norm();
		}
	}
	return t < inf;
}

Vec uniformDiskSample(double radius, Vec center) {
	double sample_r = sqrt(radius * radius * rand01());
	double sample_theta = rand01() * 2.0 * PI;
	Vec sample = Vec();
	sample.x = center.x + sample_r * cos(sample_theta);
	sample.y = center.y + sample_r * sin(sample_theta);
	sample.z = center.z;
	return sample;
}

Vec concentricDiskSample() {
	double u1 = rand01();
	double u2 = rand01();
	double uOffsetX = 2.f * u1 - 1;
	double uOffsetY = 2.f * u2 - 1;
	if (uOffsetX == 0 && uOffsetY == 0) return Vec(0, 0);
	double theta, r;
	if (abs(uOffsetX) > abs(uOffsetY)) {
		r = uOffsetX;
		theta = (PI / 4) * (uOffsetY / uOffsetX);
	}
	else {
		r = uOffsetY;
		theta = PI / 2 - (PI / 4) * (uOffsetX / uOffsetY);
	}
	return Vec(cos(theta), sin(theta)) * r;
}

Vec lightSampleDirect(Ray r, Vec pos, Vec norm, int hit_id, const Object &hit_obj) { // assumes hit_obj has PHONG BRDF
	if (hit_obj.mat.brdf->isSpecular()) {
		printf("lightSampleDirect() called on specular material\n");
		exit(1);
	}
	Vec direct_light = Vec();
	for (int i = 0; i < curr_scene_desc->scene_size; i++) {
		const Object &l_obj = *(curr_scene_desc->scene[i]);
		if ((l_obj.e.x <= 0 && l_obj.e.y <= 0 && l_obj.e.z <= 0) || i == hit_id) continue; // skip non-lights
		float pos_pdf;
		Vec light_normal;
		Vec sample_pos = l_obj.samplePos(&pos_pdf, &light_normal);
		Vec sample_dir = (sample_pos - pos).norm();
		double cos_theta_light = light_normal.dot((sample_dir * -1));
		Vec sampled_light = (cos_theta_light > 0.0) ? l_obj.e : Vec();
		if ((sampled_light.x <= 0 && sampled_light.y <= 0 && sampled_light.z <= 0) || pos_pdf <= 0.0) continue;
		Vec temp_n;
		double t;
		int id = -1;
		if (intersect(Ray(pos, sample_dir), t, id, temp_n) && id == i) { // shadow ray
			double dist = (sample_pos - pos).length();
			Vec brdf_factor;
			hit_obj.mat.brdf->evaluateBRDF(norm, r.d, hit_obj.mat.c, sample_dir, &brdf_factor);
			direct_light = direct_light + brdf_factor.mul(sampled_light * abs(sample_dir.dot(norm))) * (1.0 / pos_pdf) * (cos_theta_light / (dist * dist));
		}
	}
	return direct_light;
}

void lightSampleE(Object *light, Ray *r, Vec *f, double time_sample) { // Cosine Importance Sampling of Photon Directions
	float pos_pdf;
	Vec light_normal;
	Vec sample_pos = light->samplePos(&pos_pdf, &light_normal);
	Vec disk_sample = concentricDiskSample();
	double z = sqrt(MAX(0.0, 1 - disk_sample.x * disk_sample.x - disk_sample.y * disk_sample.y));
	Vec sample_dir = Vec(disk_sample.x, disk_sample.y, z);
	Vec v1, v2, n;
	n = light_normal;
	v1 = ((fabs(n.x) > .1 ? Vec(0, 1) : Vec(1)) % n).norm();
	v2 = n % v1;
	sample_dir = v1 * sample_dir.x + v2 * sample_dir.y + n * sample_dir.z;
	*r = Ray(sample_pos, sample_dir);
	float dir_pdf = MAX(light_normal.dot(sample_dir), 0) / PI;
	if (dir_pdf <= 0.0) {
		*f = Vec();
		return;
	}
	if (curr_scene_desc->motion_blur)
		r->time = time_sample;
	*f = light->e * MAX(light_normal.dot(sample_dir), 0) * (1.0 / pos_pdf) * (1.0 / dir_pdf);
}

bool rayPlaneIntersection(Vec plane_point, Vec plane_normal, Ray r, double *t) {
	double denom = plane_normal.dot(r.d);
	if (abs(denom) > EPS) {
		*t = (plane_point - r.o).dot(plane_normal) / denom;
		if ((*t) >= 0)
			return true;
	}
	return false;
}

void saveImage(std::vector<HPoint *> *hitpoints, int w, int h, int iterations, int photons_per_pass, double elapsed_time, int scene_nr, std::string folder_name, bool sppm) {
	Vec *c = new Vec[w * h];
	for (auto hp : *hitpoints) {
		int i = hp->pix;
		c[i] = c[i] + hp->flux * (1.0 / (PI * hp->r2 * iterations * photons_per_pass)) + hp->direct / iterations;
	}
	std::string time_str = std::string("");
	std::stringstream stream;
	stream << std::fixed << std::setprecision(2) << elapsed_time;
	time_str = stream.str() + std::string("s");
	FILE *file = fopen((folder_name + "/" + time_str + "_" + std::to_string(iterations) + "r" + ".ppm").c_str(), "w");
	fprintf(file, "P3\n%d %d\n%d\n", w, h, 255);
	for (int i = w * h; i--;) {
		fprintf(file, "%d %d %d ", toInt(c[i].x), toInt(c[i].y), toInt(c[i].z));
	}
	fclose(file);
}

std::vector<Triangle *> light_rect = {
	new Triangle(Vec(-0.005, 0, -0.005), Vec(0.005, 0, -0.005), Vec(0.005, 0, 0.005)), // Bottom
	new Triangle(Vec(-0.005, 0, -0.005), Vec(0.005, 0, 0.005), Vec(-0.005, 0, 0.005)), // Bottom
};

// SCENES
#define BOX_HALF_X 2.6
#define BOX_HALF_Y 2
#define BOX_HALF_Z 2.8

Vec default_sensor_origin(0, 0.25 * BOX_HALF_Y, 3 * BOX_HALF_Z - 1.0);
Vec default_sensor_direction(0, -0.06, -1);

Object *scene1[] = {
	new Sphere(1e5, Vec(-1e5 - BOX_HALF_X, 0, 0), Vec(), Material(Vec(.75, .25, .25), new BRDF_PHONG())),							   	// Left
	new Sphere(1e5, Vec(1e5 + BOX_HALF_X, 0, 0), Vec(), Material(Vec(1, 1, 1) * .999, new BRDF_SPEC())),							   	// Right
	new Sphere(1e5, Vec(0, 0, -1e5 - BOX_HALF_Z), Vec(), Material(Vec(.75, .75, .75), new BRDF_PHONG())),							   	// Back
	new Sphere(1e5, Vec(0, 0, +1e5 + 3 * BOX_HALF_Z - 0.5), Vec(), Material(Vec(), new BRDF_PHONG())),								   	// Front
	new Sphere(1e5, Vec(0, -1e5 - BOX_HALF_Y, 0), Vec(), Material(Vec(.25, .25, .75), new BRDF_PHONG())),							   	// Bottom
	new Sphere(1e5, Vec(0, 1e5 + BOX_HALF_Y, 0), Vec(), Material(Vec(.75, .75, .75), new BRDF_PHONG())),							   	// Top
	new Sphere(0.8, Vec(-1.3, -BOX_HALF_Y + 0.8, -1.3), Vec(), Material(Vec(1, 1, 1) * .999, new BRDF_PHONG(0.0, 1.0, 25))),		   	// Glossy
	new Sphere(0.6, Vec(BOX_HALF_X * 0.7, -BOX_HALF_Y + 0.6, 0), Vec(), Material(Vec(1, 1, 1) * .999, new BRDF_REFR())),			   	// Refractive
	new Trimesh(light_rect, Vec(0, BOX_HALF_Y - EPS, 0), Vec(1, 1, 1) * 200000, Material(Vec(), new BRDF_PHONG())),						// Light
};

SceneDescription scene_desc1(scene1, sizeof(scene1) / sizeof(Object *),
							 default_sensor_origin, default_sensor_direction, scene1[sizeof(scene1) / sizeof(Object *) - 1], 0.03, true);

SceneDescription scene_descriptors[] = {scene_desc1};

void hitpointSetup(HPoint *hitp, Vec &f, const Vec &throughput, const Vec &x, const Vec &n, const Ray &r, BRDF *brdf, const Object &obj, bool emissive, int id)
{
    hitp->f = f.mul(throughput);
    hitp->pos = x;
    hitp->nrm = n;
    hitp->valid = true;
    hitp->time = r.time;
    hitp->brdf = brdf;
    if ((obj.e.x > 0 || obj.e.y > 0 || obj.e.z > 0)) {
        if (emissive)
            hitp->direct = hitp->direct + throughput.mul(obj.e);
        hitp->valid = false;
    }
    if (curr_scene_desc->direct_sampling)
        hitp->direct = hitp->direct + throughput.mul(lightSampleDirect(r, x, n, id, obj));
}

void trace(const Ray &r, int dpt, bool eye_ray, const Vec &fl, const Vec &throughput, HPoint *hitp = nullptr, bool emissive = true)
{
    double t;
	int id = 0;
	dpt++;
	Vec n;
	if (!intersect(r, t, id, n) || (dpt >= 20)) return;
	const Object &obj = *(curr_scene_desc->scene[id]);
	Vec x = r.o + r.d * t, f = obj.mat.c;
	double p = f.x > f.y && f.x > f.z ? f.x : f.y > f.z ? f.y : f.z;
	BRDF *brdf = obj.mat.brdf;

	if (eye_ray) {
		if (!brdf->isSpecular() && !brdf->isGlossy()) // only set up a hitpoint if non-specular/non-diffuse material
            hitpointSetup(hitp, f, throughput, x, n, r, brdf, obj, emissive, id);
		else {
			Vec new_d;
			double pdf = 0.0;
			brdf->sampleDir(n, r.d, &new_d, &pdf);
			if (pdf == 0.0) return;
			Vec brdf_f;
			brdf->evaluateBRDF(n, r.d, f, new_d, &brdf_f);
			if (brdf_f.x == 0.0 && brdf_f.y == 0.0 && brdf_f.z == 0.0) return;
			if (!brdf->isSpecular())
				hitp->direct = hitp->direct + throughput.mul(lightSampleDirect(r, x, n, id, obj));
			trace(Ray(x, new_d, r.time), dpt, eye_ray, fl, throughput.mul(brdf_f) * abs(n.dot(new_d)) * (1.0 / pdf), hitp, brdf->isSpecular());
		}
	}
	else {
		if (!brdf->isSpecular() && !brdf->isGlossy() && (dpt > 1 || !curr_scene_desc->direct_sampling)) // ignore direct light, was already accounted for in the measurement point
			// Strictly speaking, '#pragma omp critical' should be used here.
			// It usually works without artifacts since photons rarely
			// contribute to the same measurement points at the same time.
			// It is significantly faster this way but can be changed if needed.
			hitpoint_kdtree->update(hitpoint_kdtree->root, x, fl, r, n, obj);
		Vec new_d;
		double pdf = 0.0;
		brdf->sampleDir(n, r.d, &new_d, &pdf);
		if (pdf == 0.0) return;
		Vec brdf_f = Vec();
		brdf->evaluateBRDF(n, r.d, obj.mat.c, new_d, &brdf_f);
		if (brdf_f.x == 0.0 && brdf_f.y == 0.0 && brdf_f.z == 0.0) return;
		if (rand01() < p)
			trace(Ray(x, new_d, r.time), dpt, eye_ray, fl.mul(brdf_f) * (1. / p) * abs(n.dot(new_d)) * (1.0 / pdf), throughput);
	}
}

void sampleCameraRay(Vec &cx, int x, int w, Vec &cy, int y, int h, Vec &sensor_origin, Vec &sensor_direction, double S_i, double aperture_r, double time_sample, Ray *new_camera_ray)
{
    double offset_x = rand01() - 0.5;
    double offset_y = rand01() - 0.5;
    Vec sensor_sample = cx * ((x + 0.5 + offset_x) / w - 0.5) + cy * (-(y + 0.5 + offset_y) / h + 0.5);
    sensor_sample = sensor_sample + sensor_origin;
    Vec lens_center = sensor_origin + sensor_direction * S_i;
    Vec d, start_point;
    if (curr_scene_desc->dof) { // if depth of field is enabled, sample a lens position
        Ray center_ray(lens_center, (lens_center - sensor_sample).norm());
        Vec focus_plane_point = lens_center + sensor_direction * curr_scene_desc->S_o;
        Vec focus_plane_normal = sensor_direction * -1;
        double t = 0.0;
        if (!rayPlaneIntersection(focus_plane_point, focus_plane_normal, center_ray, &t))
            printf("Something went very wrong!\n");
        Vec focus_point = center_ray.o + center_ray.d * t;
        Vec lens_sample = uniformDiskSample(aperture_r, lens_center);
        start_point = lens_sample;
        d = (focus_point - lens_sample).norm();
    }
    else {
        start_point = lens_center;
        d = (lens_center - sensor_sample).norm();
    }
    double time = 0.0;
    if (curr_scene_desc->motion_blur)
        time = time_sample;
    *new_camera_ray = Ray(start_point, d, time);
}

int main(int argc, char *argv[]) {
	int photons_per_pass = 1000000;
	int w = 960, h = 720, samps = (argc >= 2) ? MAX(atoi(argv[1]), 1) : 20;
	std::vector<int> checkpoints = {10, 50, 100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000};
	size_t scene_nr = (argc >= 3) ? atoi(argv[2]) : 1;
	double max_time = (argc == 4) ? atof(argv[3]) : 0.0;
	Vec sensor_origin, sensor_direction;
	double sensor_width = 0.036, sensor_height = 0.024; // 4:3
	double S_i = 0.03;

	if (scene_nr > sizeof(scene_descriptors) / sizeof(SceneDescription)) {
		printf("Scene not found. Make sure you entered the number of an existing scene!\n");
		exit(-1);
	}

	// set up scene and camera
	curr_scene_desc = &(scene_descriptors[scene_nr - 1]);
	sensor_origin = curr_scene_desc->sensor_origin;
	sensor_direction = curr_scene_desc->sensor_direction;

	Vec cx = Vec(w * sensor_height / h), cy = (cx % sensor_direction).norm() * sensor_height, r, vw;
	double f = (S_i * curr_scene_desc->S_o) / (S_i + curr_scene_desc->S_o);
	double aperture_diameter = f / curr_scene_desc->f_stop;
	double aperture_r = aperture_diameter / 2.0;

	// set up output folder
	namespace fs = std::filesystem;
	auto time_stamp = std::time(nullptr);
	auto time_stamp_m = *std::localtime(&time_stamp);
	std::ostringstream oss;
	oss << std::put_time(&time_stamp_m, "%Y-%m-%d_%H-%M-%S");
	std::string date_time_string = oss.str();
	std::string folder_name = ("./output/sppm_scene_" + std::to_string(scene_nr) + "_" + date_time_string).c_str();
	fs::create_directories(folder_name);

	printf("Rendering Scene %lld\n", scene_nr);
	auto tstart = std::chrono::system_clock::now();

	// initialize hitpoints
	auto hitpoints = new std::vector<HPoint *>;
	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			HPoint *hitpoint = new HPoint;
			hitpoint->flux = Vec();
			hitpoint->direct = Vec();
			hitpoint->r2 = curr_scene_desc->r_0 * curr_scene_desc->r_0;
			hitpoint->n = 0;
			hitpoint->pix = x + y * w;
			hitpoints->push_back(hitpoint);
		}
	}

	for (int round = 0; round < samps; round++) { // render loop
		double time_sample = rand01();
		#pragma omp parallel for schedule(dynamic, 1) // RAY TRACING PASS
		for (int y = 0; y < h; y++) {
			for (int x = 0; x < w; x++) {
				Ray new_camera_ray;
                sampleCameraRay(cx, x, w, cy, y, h, sensor_origin, sensor_direction, S_i, aperture_r, time_sample, &new_camera_ray);
				HPoint *curr_hp = hitpoints->at(y * w + x);
				curr_hp->valid = false;
				trace(new_camera_ray, 0, true, Vec(), Vec(1, 1, 1), curr_hp);
			}
		}

		// set up kdtree and populate it with hitpoints
		if (hitpoint_kdtree)
			delete hitpoint_kdtree;
		hitpoint_kdtree = new HitPointKDTree(hitpoints);

		#pragma omp parallel for schedule(dynamic, 1) // PHOTON PASS
		for (int j = 0; j < photons_per_pass; j++) {
			Ray r;
			Vec f;
			Vec initial_throughput = Vec(1, 1, 1);
			lightSampleE(curr_scene_desc->light, &r, &f, time_sample);
			trace(r, 0, 0 > 1, f, initial_throughput);
		}
		fprintf(stderr, "\rFinished Round %d/%d", round + 1, samps);

		auto tround = std::chrono::system_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(tround - tstart).count();
		if (max_time > EPS && elapsed >= max_time) {// Automatically stop after max_time elapsed
			fprintf(stderr, "\nElapsed Time: %f\nFinished after %d rounds\n", elapsed, round + 1);
			saveImage(hitpoints, w, h, round + 1, photons_per_pass, elapsed, scene_nr, folder_name, true);
			break;
		}
		if (std::find(checkpoints.begin(), checkpoints.end(), round + 1) != checkpoints.end() || (round + 1) == samps) { // save image when checkpoints are reached
			saveImage(hitpoints, w, h, round + 1, photons_per_pass, elapsed, scene_nr, folder_name, true);
		}
	}
	delete hitpoints;
	delete hitpoint_kdtree;
}
