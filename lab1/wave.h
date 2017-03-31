#pragma once

#include <math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

const __constant__ int NUM_STEPS = 8;
const __constant__ float PI = 3.1415f;
const __constant__ float EPSILON = 1e-3f;
#define EPSILON_NRM (0.1 / iResolution.x)

// sea
float __device__ iGlobalTime = 0.0f;
float2 __constant__ iResolution = {640, 480};
const __constant__ int ITER_GEOMETRY = 3;
const __constant__ int ITER_FRAGMENT = 5;
const __constant__ float SEA_HEIGHT = 0.6f;
const __constant__ float SEA_CHOPPY = 4.0f;
const __constant__ float SEA_SPEED = 0.8f;
const __constant__ float SEA_FREQ = 0.16f;

const __constant__ float3 SEA_BASE = { 0.1f, 0.19f, 0.22f };
const __constant__ float3 SEA_WATER_COLOR = { 0.8f, 0.9f, 0.6f };
#define SEA_TIME (1.0 + iGlobalTime * SEA_SPEED)
const __constant__ float octave_m[4] = { 1.6f, 1.2f, -1.2f, 1.6f };

// math
void __device__  fromEuler(float3 ang, float3 &v1, float3 &v2, float3 &v3)  {
	float2 a1 = { sin(ang.x), cos(ang.x) };
	float2 a2 = { sin(ang.y), cos(ang.y) };
	float2 a3 = { sin(ang.z), cos(ang.z) };
	//float3 m[3];
	v1 = { a1.y*a3.y + a1.x*a2.x*a3.x, a1.y*a2.x*a3.x + a3.y*a1.x, -a2.y*a3.x };
	v2 = { -a2.y*a1.x, a1.y*a2.y, a2.x };
	v3 = { a3.y*a1.x*a2.x + a1.y*a3.x, a1.x*a3.x - a1.y*a3.y*a2.x, a2.y*a3.y };
	//return m;
}

float __device__ hash(float2 p) {
	float2 n = { 127.1f, 311.7f };
	//dot p,n
	float h = p.x * n.x + p.y * n.y;
	//return the fractional part
	return ((sin(h)*43758.5453123f) - floor((sin(h)*43758.5453123f)));
}

float __device__ mix(float a, float b, float c) {
	return a * (1 - c) + b*c;
}
float __device__ maximum(float a, float b) {
	return a > b ? a : b;
}
float __device__ clamp(float a, float min_a, float max_a) {
	if (a > max_a) return max_a;
	else if (a < min_a)	return min_a;
	else return a;
}
float __device__ smoothstep(float edge0, float edge1, float x) {
	float t;  /* Or genDType t; */
	t = clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
	return t * t * (3.0f - 2.0f * t);
}
float3 __device__ reflect(float3 I, float3 N) {
	//return I - 2.0 * dot(N, I) * N
	return{ I.x - 2.0f * (N.x * I.x + N.y*I.y + N.z*I.z) * N.x, I.y - 2.0f * (N.x * I.x + N.y*I.y + N.z*I.z) * N.y, I.z - 2.0f * (N.x * I.x + N.y*I.y + N.z*I.z) * N.z };
}
float __device__ noise(float2 p) {
	float2 i = { floorf(p.x), floorf(p.y) };
	float2 f = { p.x - floorf(p.x), p.y - floorf(p.y) };
	float2 u = { f.x*f.x*(3.0f - 2.0f*f.x) , f.y*f.y*(3.0f - 2.0f*f.y) };
	float2 v1 = { 0.0f + i.x,0.0f +i.y};
	float2 v2 = { 1.0f + i.x,0.0f +i.y};
	float2 v3 = { 0.0f + i.x,1.0f +i.y};
	float2 v4 = { 1.0f + i.x,1.0f +i.y};


	return -1.0f + 2.0f*mix(mix(hash(v1),
		hash(v2), u.x),
		mix(hash(v3),
			hash(v4), u.x), u.y);
}
// lighting
float __device__ diffuse(float3 n, float3 l, float p) {
	return pow((n.x * l.x + n.y * l.y + n.z * l.z) * 0.4f + 0.6f, p);
}
float __device__ specular(float3 n, float3 l, float3 e, float s) {
	float nrm = (s + 8.0f) / (3.1415f * 8.0f);
	float3 reflect_e = reflect(e, n);
	return pow(maximum((reflect_e.x * l.x + reflect_e.y * l.y + reflect_e.z * l.z), 0.0), s) * nrm;
}

// sky
float3 __device__ getSkyColor(float3 e) {
	e.y = maximum(e.y, 0.0f);
	float3 skycolor = { pow(1.0f - e.y, 2.0f), 1.0f - e.y, 0.6f + (1.0f - e.y)*0.4f };
	return skycolor;
}

// sea
float __device__ sea_octave(float2 uv, float choppy) {
	uv.x += noise(uv);
	uv.y += noise(uv);
	float2 wv = { 1.0f - abs(sin(uv.x)) , 1.0f - abs(sin(uv.y)) };
	float2 swv = { abs(cos(uv.x)) ,abs(cos(uv.y)) };
	wv = { mix(wv.x, swv.x, wv.x), mix(wv.y, swv.y, wv.y) };
	return pow(1.0f - pow(wv.x * wv.y, 0.65f), choppy);
}

float __device__ map(float3 p) {
	float freq = SEA_FREQ;
	float amp = SEA_HEIGHT;
	float choppy = SEA_CHOPPY;
	float2 uv = { p.x, p.z }; uv.x *= 0.75f;

	float d, h = 0.0f;
	for (int i = 0; i < ITER_GEOMETRY; i++) {
		float2 uv_plus_SEATIME_times_freq = { (uv.x + SEA_TIME)*freq, (uv.y + SEA_TIME)*freq };
		float2 uv_minus_SEATIME_times_freq = { (uv.x - SEA_TIME)*freq, (uv.y - SEA_TIME)*freq };
		d = sea_octave(uv_plus_SEATIME_times_freq, choppy);
		d += sea_octave(uv_minus_SEATIME_times_freq, choppy);
		h += d * amp;
		//vec2 * mat2
		uv = {uv.x * octave_m[0] + uv.y * octave_m[2], uv.x * octave_m[1] + uv.y * octave_m[3] };
		freq *= 1.9f; amp *= 0.22f;
		choppy = mix(choppy, 1.0f, 0.2f);
	}
	return p.y - h;
}

float __device__ map_detailed(float3 p) {
	float freq = SEA_FREQ;
	float amp = SEA_HEIGHT;
	float choppy = SEA_CHOPPY;
	float2 uv = { p.x, p.z }; uv.x *= 0.75;

	float d, h = 0.0f;
	for (int i = 0; i < ITER_FRAGMENT; i++) {
		float2 uv_plus_SEATIME_times_freq = { (uv.x + SEA_TIME)*freq, (uv.y + SEA_TIME)*freq };
		float2 uv_minus_SEATIME_times_freq = { (uv.x - SEA_TIME)*freq, (uv.y - SEA_TIME)*freq };
		d = sea_octave(uv_plus_SEATIME_times_freq, choppy);
		d += sea_octave(uv_minus_SEATIME_times_freq, choppy);
		h += d * amp;
		uv = { uv.x * octave_m[0] + uv.y * octave_m[2],uv.x * octave_m[1] + uv.y * octave_m[3] };
		freq *= 1.9f; amp *= 0.22f;
		choppy = mix(choppy, 1.0f, 0.2f);
	}
	return p.y - h;
}

float3 __device__ getSeaColor(float3 p, float3 n, float3 l, float3 eye, float3 dist) {
	float fresnel = clamp(1.0f - (n.x * -eye.x + n.y * -eye.y + n.z * -eye.z), 0.0f, 1.0f);
	
	fresnel = pow(fresnel, 3.0f) * 0.65f;

	float3 reflected = getSkyColor(reflect(eye, n));
	float3 refracted = { SEA_BASE.x + diffuse(n, l, 80.0f) * SEA_WATER_COLOR.x * 0.12f , SEA_BASE.y + diffuse(n, l, 80.0f) * SEA_WATER_COLOR.y * 0.12f, SEA_BASE.z + diffuse(n, l, 80.0f) * SEA_WATER_COLOR.z * 0.12f };

	float3 color = { mix(refracted.x, reflected.x, fresnel), mix(refracted.y, reflected.y, fresnel), mix(refracted.z, reflected.z, fresnel) };

	float atten = maximum(1.0f - (dist.x * dist.x + dist.y * dist.y + dist.z * dist.z) * 0.001f, 0.0f);
	color.x += SEA_WATER_COLOR.x * (p.y - SEA_HEIGHT) * 0.18f * atten + specular(n, l, eye, 60.0f);
	color.y += SEA_WATER_COLOR.y * (p.y - SEA_HEIGHT) * 0.18f * atten + specular(n, l, eye, 60.0f);
	color.z += SEA_WATER_COLOR.z * (p.y - SEA_HEIGHT) * 0.18f * atten + specular(n, l, eye, 60.0f);

	return color;
}

// tracing
float3 __device__ getNormal(float3 p, float eps) {
	float3 n;
	n.y = map_detailed(p);
	float3 v1 = { p.x + eps, p.y, p.z };
	float3 v2 = { p.x, p.y, p.z + eps };
	n.x = map_detailed(v1) - n.y;
	n.z = map_detailed(v2) - n.y;
	n.y = eps;
	float distance = sqrtf(powf(n.x, 2.0f) + powf(n.y, 2.0f) + powf(n.z, 2.0f));
	n = { n.x / distance, n.y / distance, n.z / distance };

	return n;
}

float __device__ heightMapTracing(float3 ori, float3 dir, float3 &p) {
	float tm = 0.0f;
	float tx = 1000.0f;
	float3 v1 = { ori.x + dir.x * tx, ori.y + dir.y * tx, ori.z + dir.z * tx };
	float hx = map(v1);
	if (hx > 0.0f) return tx;
	float3 v2 = { ori.x + dir.x * tm, ori.y + dir.y * tm, ori.z + dir.z * tm };
	float hm = map(v2);
	float tmid = 0.0f;
	for (int i = 0; i < NUM_STEPS; i++) {
		tmid = mix(tm, tx, hm / (hm - hx));
		p = { ori.x + dir.x * tmid, ori.y + dir.y * tmid, ori.z + dir.z * tmid };
		float hmid = map(p);
		if (hmid < 0.0f) {
			tx = tmid;
			hx = hmid;
		}
		else {
			tm = tmid;
			hm = hmid;
		}
	}
	return tmid;
}

// main
void __device__ mainImage(float4 &fragColor, /*in*/ float2 fragCoord, float currTime) {
	iGlobalTime = currTime/24;
	float2 uv = { fragCoord.x / iResolution.x , fragCoord.y / iResolution.y };
	uv = { uv.x * 2.0f - 1.0f, uv.y * 2.0f - 1.0f };
	uv.x *= iResolution.x / iResolution.y;
	float time = iGlobalTime * 0.3f;

	// ray
	float3 ang = { sin(time*3.0f)*0.1f, sin(time)*0.2f + 0.3f, time };
	float3 ori = { 0.0f, 3.5f, time*5.0f };
	float3 dir = { uv.x, uv.y, -2.0f }; 
	float length_dir = sqrtf(powf(dir.x, 2.0f) + powf(dir.y, 2.0f) + powf(dir.z, 2.0f));
	float length_uv = sqrtf(powf(uv.x, 2.0f) + powf(uv.y, 2.0f));
	dir = { dir.x / length_dir , dir.y / length_dir, dir.z / length_dir };
	dir.z += length_uv * 0.15f;
	//dir = normalize(dir) * fromEuler(ang);
	//length_dir = sqrtf(powf(dir.x, 2.0f) + powf(dir.y, 2.0f) + powf(dir.z, 2.0f));
	//float3 v1, v2, v3;
	//fromEuler(ang, v1, v2, v3);
	//dir = { dir.x / length_dir , dir.y / length_dir, dir.z / length_dir };
	//dir = { dir.x * v1.x + dir.y * v2.x + dir.z * v3.x, dir.x * v1.y + dir.y * v2.y + dir.z * v3.y , dir.x * v1.z + dir.y * v2.z + dir.z * v3.z };
	

	// tracing
	float3 p;
	heightMapTracing(ori, dir, p);
	float3 dist = { p.x - ori.x , p.y - ori.y , p.z - ori.z };
	float3 n = getNormal(p, (dist.x * dist.x + dist.y * dist.y + dist.z * dist.z) * EPSILON_NRM);
	//sqrt(0.0 * 0.0 + 1.0 * 1.0 + 0.8 * 0.8)
	float distance_light = sqrtf(1.64f);
	float3 light = { 0.0f, 1.0f / distance_light, 0.8f / distance_light };
	float3 color;
	if (fragCoord.y < iResolution.y/2 - 1) {
		//color
		color = {
			mix(
			getSkyColor(dir).x,
				getSeaColor(p, n, light, dir, dist).x,
				pow(smoothstep(0.0f, -0.05f, dir.y), 0.3f)),
			mix(
				getSkyColor(dir).y,
				getSeaColor(p, n, light, dir, dist).y,
				pow(smoothstep(0.0f, -0.05f, dir.y), 0.3f)),
			mix(
				getSkyColor(dir).z,
				getSeaColor(p, n, light, dir, dist).z,
				pow(smoothstep(0.0f, -0.05f, dir.y), 0.3f))
		};
	}
	else {
		color = getSkyColor(dir);
	}
	
	
	// post
	fragColor = { clamp( pow(color.x,0.75f) * 255.0f, 0.0f, 255.0f), clamp(pow(color.y,0.75f) * 255.0f, 0.0f, 255.0f), clamp(pow(color.z,0.75f) * 255.0f, 0.0f, 255.0f), 1.0f };
	//fragColor = { getSkyColor(dir).x *255.0f, getSkyColor(dir).y *255.0f, getSkyColor(dir).z *255.0f, 1.0f};
}