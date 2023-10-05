/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   nerfloader.cu
 *  @author Alex Evans & Thomas Müller, NVIDIA
 *  @brief  Loads a NeRF data set from NeRF's original format
 */

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/common_device.cuh>
#include <neural-graphics-primitives/nerf_loader.h>
#include <neural-graphics-primitives/thread_pool.h>
#include <neural-graphics-primitives/tinyexr_wrapper.h>

#include <json/json.hpp>

#include <filesystem/path.h>

#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION

#if defined(__NVCC__)
#if defined __NVCC_DIAG_PRAGMA_SUPPORT__
#  pragma nv_diag_suppress 550
#else
#  pragma diag_suppress 550
#endif
#endif
#include <stb_image/stb_image.h>
#if defined(__NVCC__)
#if defined __NVCC_DIAG_PRAGMA_SUPPORT__
#  pragma nv_diag_default 550
#else
#  pragma diag_default 550
#endif
#endif

using namespace tcnn;
using namespace std::literals;
using namespace Eigen;
namespace fs = filesystem;

NGP_NAMESPACE_BEGIN

// A CUDA kernel written in C++ code.
// For each pixel, do conditional checks based on the white_2_transparent, black_2_transparent, and mask_color flags,
// and modify the pixels accordingly. (However, these are never used in NeuS2.)
// - white_2_transparent: whether to convert white pixels to transparent 
// - black_2_transparent: whether to convert black pixels to transparent
// - mask_color: A color value that should be treated as a mask

/**
 * @brief CUDA kernel function to convert one pixel to RGBA format. Copy from "pixels" to "out".
 * 
 * @param num_pixels Number of pixels in one image. This is also the number of threads in total. 
 * @param pixels This is a pointer to the input pixel data. It is expected to be an array of RGBA values where each pixel occupies 4 bytes (32 bits). The data is provided as an array of uint8_t, which represents the color channels (red, green, blue, alpha) for each pixel.
 * @param out This is a pointer to the output buffer where the modified pixel data will be stored. Like the pixels parameter, it is also expected to be an array of RGBA values, and each pixel occupies 4 bytes.
 * @param white_2_transparent default = false
 * @param black_2_transparent default = false
 * @param mask_color default = 0
 * 
*/
__global__ void convert_rgba32(const uint64_t num_pixels, const uint8_t* __restrict__ pixels, 
							   uint8_t* __restrict__ out, bool white_2_transparent = false, 
							   bool black_2_transparent = false, uint32_t mask_color = 0) {
	// Multiple threads (number of threads = num_pixels)
	// Each pixel has a thread to do calculation
	const uint64_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_pixels) return;

	uint8_t rgba[4]; // an array of size 4 bytes (representing RGBA values, 1 byte for 1 channel)
	// It interprets the input pixel data at index i*4 as a 32-bit integer (uint32_t) and 
	// stores the four individual color channels (red, green, blue, and alpha) in the rgba array.
	*((uint32_t*)&rgba[0]) = *((uint32_t*)&pixels[i*4]);

	// NSVF dataset has 'white = transparent' madness
	if (white_2_transparent && rgba[0] == 255 && rgba[1] == 255 && rgba[2] == 255) {
		rgba[3] = 0;
	}
	
	if (black_2_transparent && rgba[0] == 0 && rgba[1] == 0 && rgba[2] == 0) {
		rgba[3] = 0;
	}

	if (mask_color != 0 && mask_color == *((uint32_t*)&rgba[0])) {
		// turn the mask into hot pink
		rgba[0] = 0xFF; rgba[1] = 0x00; rgba[2] = 0xFF; rgba[3] = 0x00;
	}

	// stores the modified (actually not used) rgba values back into the out array at index i*4
	*((uint32_t*)&out[i*4]) = *((uint32_t*)&rgba[0]);
}

__global__ void from_fullp(const uint64_t num_elements, const float* __restrict__ pixels, __half* __restrict__ out) {
	const uint64_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;

	out[i] = (__half)pixels[i];
}

template <typename T>
__global__ void copy_depth(const uint64_t num_elements, float* __restrict__ depth_dst, const T* __restrict__ depth_pixels, float depth_scale) {
	const uint64_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;

	if (depth_pixels == nullptr || depth_scale <= 0.f) {
		depth_dst[i] = 0.f; // no depth data for this entire image. zero it out
	} else {
		depth_dst[i] = depth_pixels[i] * depth_scale;
	}
}

template <typename T>
__global__ void sharpen(const uint64_t num_pixels, const uint32_t w, const T* __restrict__ pix, T* __restrict__ destpix, float center_w, float inv_totalw) {
	const uint64_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_pixels) return;

	float rgba[4] = {
		(float)pix[i*4+0]*center_w,
		(float)pix[i*4+1]*center_w,
		(float)pix[i*4+2]*center_w,
		(float)pix[i*4+3]*center_w
	};

	int64_t i2=i-1; if (i2<0) i2=0; i2*=4;
	for (int j=0;j<4;++j) rgba[j]-=(float)pix[i2++];
	i2=i-w; if (i2<0) i2=0; i2*=4;
	for (int j=0;j<4;++j) rgba[j]-=(float)pix[i2++];
	i2=i+1; if (i2>=num_pixels) i2-=num_pixels; i2*=4;
	for (int j=0;j<4;++j) rgba[j]-=(float)pix[i2++];
	i2=i+w; if (i2>=num_pixels) i2-=num_pixels; i2*=4;
	for (int j=0;j<4;++j) rgba[j]-=(float)pix[i2++];
	for (int j=0;j<4;++j) destpix[i*4+j]=(T)max(0.f, rgba[j] * inv_totalw);
}

__device__ inline float luma(const Array4f& c) {
	return c[0] * 0.2126f + c[1] * 0.7152f + c[2] * 0.0722f;
}

__global__ void compute_sharpness(Eigen::Vector2i sharpness_resolution, Eigen::Vector2i image_resolution, uint32_t n_images, const void* __restrict__ images_data, EImageDataType image_data_type, float* __restrict__ sharpness_data) {
	const uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
	const uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;
	const uint32_t i = threadIdx.z + blockIdx.z * blockDim.z;
	if (x >= sharpness_resolution.x() || y >= sharpness_resolution.y() || i>=n_images) return;
	const size_t sharp_size = sharpness_resolution.x() * sharpness_resolution.y();
	sharpness_data += sharp_size * i + x + y * sharpness_resolution.x();

	// overlap patches a bit
	int x_border = 0; // (image_resolution.x()/sharpness_resolution.x())/4;
	int y_border = 0; // (image_resolution.y()/sharpness_resolution.y())/4;

	int x1 = (x*image_resolution.x())/sharpness_resolution.x()-x_border, x2 = ((x+1)*image_resolution.x())/sharpness_resolution.x()+x_border;
	int y1 = (y*image_resolution.y())/sharpness_resolution.y()-y_border, y2 = ((y+1)*image_resolution.y())/sharpness_resolution.y()+y_border;
	// clamp to 1 pixel in from edge
	x1=max(x1,1); y1=max(y1,1);
	x2=min(x2,image_resolution.x()-2); y2=min(y2,image_resolution.y()-2);
	// yes, yes I know I should do a parallel reduction and shared memory and stuff. but we have so many tiles in flight, and this is load-time, meh.
	float tot_lap=0.f,tot_lap2=0.f,tot_lum=0.f;
	float scal=1.f/((x2-x1)*(y2-y1));
	for (int yy=y1;yy<y2;++yy) {
		for (int xx=x1; xx<x2; ++xx) {
			Array4f n, e, s, w, c;
			c = read_rgba(Vector2i{xx, yy}, image_resolution, images_data, image_data_type, i);
			n = read_rgba(Vector2i{xx, yy-1}, image_resolution, images_data, image_data_type, i);
			w = read_rgba(Vector2i{xx-1, yy}, image_resolution, images_data, image_data_type, i);
			s = read_rgba(Vector2i{xx, yy+1}, image_resolution, images_data, image_data_type, i);
			e = read_rgba(Vector2i{xx+1, yy}, image_resolution, images_data, image_data_type, i);
			float lum = luma(c);
			float lap = lum * 4.f - luma(n) - luma(e) - luma(s) - luma(w);
			tot_lap += lap;
			tot_lap2 += lap*lap;
			tot_lum += lum;
		}
	}
	tot_lap*=scal;
	tot_lap2*=scal;
	tot_lum*=scal;
	float variance_of_laplacian = tot_lap2 - tot_lap * tot_lap;
	*sharpness_data = (variance_of_laplacian) ; // / max(0.00001f,tot_lum*tot_lum); // var / (tot+0.001f);
}

bool ends_with(const std::string& str, const std::string& suffix) {
	return str.size() >= suffix.size() && 0 == str.compare(str.size()-suffix.size(), suffix.size(), suffix);
}

NerfDataset create_empty_nerf_dataset(size_t n_images, int aabb_scale, bool is_hdr) {
	NerfDataset result{};
	result.n_images = n_images;
	result.sharpness_resolution = { 128, 72 };
	result.sharpness_data.enlarge( result.sharpness_resolution.x() * result.sharpness_resolution.y() *  result.n_images );
	result.xforms.resize(n_images);
	result.metadata.resize(n_images);
	result.pixelmemory.resize(n_images);
	result.depthmemory.resize(n_images);
	result.raymemory.resize(n_images);
	result.scale = NERF_SCALE;
	result.offset = {0.5f, 0.5f, 0.5f};
	result.aabb_scale = aabb_scale;
	result.is_hdr = is_hdr;
	for (size_t i = 0; i < n_images; ++i) {
		result.xforms[i].start = Eigen::Matrix<float, 3, 4>::Identity();
		result.xforms[i].end = Eigen::Matrix<float, 3, 4>::Identity();
	}
	return result;
}

// NerfDataset load_nerf(const std::vector<filesystem::path>& jsonpaths, float sharpen_amount, bool is_downsample) {
NerfDataset load_nerf(const std::vector<filesystem::path>& jsonpaths, float sharpen_amount) {
	if (jsonpaths.empty()) {
		throw std::runtime_error{"Cannot load NeRF data from an empty set of paths."};
	}

	tlog::info() << "Loading NeRF dataset from";

	NerfDataset result{};
	std::ifstream f{jsonpaths.front().str()};
	nlohmann::json transforms = nlohmann::json::parse(f, nullptr, true, true); // parse the json file

	ThreadPool pool;

	struct LoadedImageInfo {
		Eigen::Vector2i res = Eigen::Vector2i::Zero(); // resolution
		bool image_data_on_gpu = false;
		EImageDataType image_type = EImageDataType::None;
		bool white_transparent = false;
		bool black_transparent = false;
		uint32_t mask_color = 0;
		void *pixels = nullptr; // the image data
		uint16_t *depth_pixels = nullptr;
		Ray *rays = nullptr;
		float depth_scale = -1.f;
	};
	std::vector<LoadedImageInfo> images; // a vector of struct LoadedImageInfo
	LoadedImageInfo info = {};

	if (transforms["camera"].is_array()) {
		throw std::runtime_error{"hdf5 is no longer supported. please use the hdf52nerf.py conversion script"};
	}

	// auto transfer_to_downsample_json = [] (const auto& path, const bool is_downsample) {
	// 	if (!is_downsample) {
	// 		return path.str();
	// 	}
	// 	std::string downsample_path = path.stem().str() + std::string{"_downsample.json"};
	// 	printf("load downsample json: %s\n", downsample_path.c_str());
	// 	return downsample_path;
	// };
	// nerf original format

	std::vector<nlohmann::json> jsons;
	std::transform(
		jsonpaths.begin(), jsonpaths.end(),
		std::back_inserter(jsons), [=] (const auto& path) {
			// return nlohmann::json::parse(std::ifstream{transfer_to_downsample_json(path, is_downsample)}, nullptr, true, true);
			return nlohmann::json::parse(std::ifstream{path.str()}, nullptr, true, true);
		}
	);

	// Get the total number of images in all the json files (all the frames in a dynamic scene)
	result.n_images = 0;
	for (size_t i = 0; i < jsons.size(); ++i) {
		// for each json file, get the number of views for that json file (1 frame)
		auto& json = jsons[i];
		fs::path basepath = jsonpaths[i].parent_path();
		if (!json.contains("frames") || !json["frames"].is_array()) {
			tlog::warning() << "  " << jsonpaths[i] << " does not contain any frames. Skipping.";
			continue;
		}
		tlog::info() << "  " << jsonpaths[i];
		auto& frames = json["frames"]; // the "frames" key in each json file contains all the camera views

		float sharpness_discard_threshold = json.value("sharpness_discard_threshold", 0.0f); // Keep all by default

		// std::sort(frames.begin(), frames.end(), [](const auto& frame1, const auto& frame2) {
		// 	return frame1["file_path"] < frame2["file_path"];
		// });

		if (json.contains("n_frames")) {
			size_t cull_idx = std::min(frames.size(), (size_t)json["n_frames"]);
			frames.get_ptr<nlohmann::json::array_t*>()->resize(cull_idx);
		}

		if (frames[0].contains("sharpness")) { // by default, false
			auto frames_copy = frames;
			frames.clear();

			// Kill blurrier frames than their neighbors
			const int neighborhood_size = 3;
			for (int i = 0; i < (int)frames_copy.size(); ++i) {
				float mean_sharpness = 0.0f;
				int mean_start = std::max(0, i-neighborhood_size);
				int mean_end = std::min(i+neighborhood_size, (int)frames_copy.size()-1);
				for (int j = mean_start; j < mean_end; ++j) {
					mean_sharpness += float(frames_copy[j]["sharpness"]);
				}
				mean_sharpness /= (mean_end - mean_start);

				// Compatibility with Windows paths on Linux. (Breaks linux filenames with "\\" in them, which is acceptable for us.)
				frames_copy[i]["file_path"] = replace_all(frames_copy[i]["file_path"], "\\", "/");

				if ((basepath / fs::path(std::string(frames_copy[i]["file_path"]))).exists() && frames_copy[i]["sharpness"] > sharpness_discard_threshold * mean_sharpness) {
					frames.emplace_back(frames_copy[i]);
				}
			}
		}

		// Set the number of images provided in the json file
		// Total number of images plus current json file's number of views
		result.n_images += frames.size();
	}

	images.resize(result.n_images); // resize the "images" vector to the size equal to the number of images
	result.xforms.resize(result.n_images);
	result.metadata.resize(result.n_images);
	result.pixelmemory.resize(result.n_images);
	result.depthmemory.resize(result.n_images);
	result.raymemory.resize(result.n_images);

	result.scale = NERF_SCALE;
	result.offset = {0.5f, 0.5f, 0.5f};

	// A vector of std::futures
	// The class template std::future provides a mechanism to access the result of asynchronous operations:
	std::vector<std::future<void>> futures;

	size_t image_idx = 0;
	if (result.n_images==0) {
		throw std::invalid_argument{"No training images were found for NeRF training!"};
	}

	auto progress = tlog::progress(result.n_images);

	result.from_mitsuba = false;
	result.from_na = false;
	bool fix_premult = false;
	bool enable_ray_loading = true;
	bool enable_depth_loading = true;
	std::atomic<int> n_loaded{0}; // Each instantiation and full specialization of the std::atomic template defines an atomic type. If one thread writes to an atomic object while another thread reads from it, the behavior is well-defined (see memory model for details on data races).
	BoundingBox cam_aabb; // axis-aligned boubding box for the cameras
	for (size_t i = 0; i < jsons.size(); ++i) { // for all the json files (only 1 for NeuS2 static scenes)
		auto& json = jsons[i];

		fs::path basepath = jsonpaths[i].parent_path();
		std::string jp = jsonpaths[i].str();
		auto lastdot=jp.find_last_of('.'); if (lastdot==std::string::npos) lastdot=jp.length();
		auto lastunderscore=jp.find_last_of('_'); if (lastunderscore==std::string::npos) lastunderscore=lastdot; else lastunderscore++;
		std::string part_after_underscore(jp.begin()+lastunderscore,jp.begin()+lastdot);

		if (json.contains("enable_ray_loading")) {
			enable_ray_loading = bool(json["enable_ray_loading"]);
			tlog::info() << "enable_ray_loading=" << enable_ray_loading;
		}
		if (json.contains("enable_depth_loading")) {
			enable_depth_loading = bool(json["enable_depth_loading"]);
			tlog::info() << "enable_depth_loading is " << enable_depth_loading;
		}

		if (json.contains("normal_mts_args")) {
			result.from_mitsuba = true;
		}

		if (json.contains("from_na")) {
			// result.from_na = true;
			result.from_na = bool(json["from_na"]);
		}

		if (json.contains("fix_premult")) {
			fix_premult = (bool)json["fix_premult"];
		}

		if (result.from_mitsuba) {
			result.scale = 0.66f;
			result.offset = {0.25f * result.scale, 0.25f * result.scale, 0.25f * result.scale};
		}

		if (json.contains("render_aabb")) {
			result.render_aabb.min={float(json["render_aabb"][0][0]),float(json["render_aabb"][0][1]),float(json["render_aabb"][0][2])};
			result.render_aabb.max={float(json["render_aabb"][1][0]),float(json["render_aabb"][1][1]),float(json["render_aabb"][1][2])};
		}

		if (json.contains("sharpen")) {
			sharpen_amount = json["sharpen"];
		}

		if (json.contains("white_transparent")) {
			info.white_transparent = bool(json["white_transparent"]);
		}

		if (json.contains("black_transparent")) {
			info.black_transparent = bool(json["black_transparent"]);
		}

		// In NeuS2
		if (json.contains("scale")) {
			result.scale = json["scale"]; // set the scale to be applied to camera positions
		}

		if (json.contains("importance_sampling")) {
			result.wants_importance_sampling = json["importance_sampling"];
		}

		if (json.contains("n_extra_learnable_dims")) {
			result.n_extra_learnable_dims = json["n_extra_learnable_dims"];
		}

		CameraDistortion camera_distortion = {};
		Vector2f principal_point = Vector2f::Constant(0.5f);
		Vector4f rolling_shutter = Vector4f::Zero();

		if (json.contains("integer_depth_scale")) {
			info.depth_scale = json["integer_depth_scale"];
		}

		// Camera distortion
		{
			if (json.contains("k1")) {
				camera_distortion.params[0] = json["k1"];
				if (camera_distortion.params[0] != 0.f) {
					camera_distortion.mode = ECameraDistortionMode::Iterative;
				}
			}

			if (json.contains("k2")) {
				camera_distortion.params[1] = json["k2"];
				if (camera_distortion.params[1] != 0.f) {
					camera_distortion.mode = ECameraDistortionMode::Iterative;
				}
			}

			if (json.contains("p1")) {
				camera_distortion.params[2] = json["p1"];
				if (camera_distortion.params[2] != 0.f) {
					camera_distortion.mode = ECameraDistortionMode::Iterative;
				}
			}

			if (json.contains("p2")) {
				camera_distortion.params[3] = json["p2"];
				if (camera_distortion.params[3] != 0.f) {
					camera_distortion.mode = ECameraDistortionMode::Iterative;
				}
			}

			if (json.contains("cx")) {
				principal_point.x() = (float)json["cx"] / (float)json["w"];
			}

			if (json.contains("cy")) {
				principal_point.y() = (float)json["cy"] / (float)json["h"];
			}

			if (json.contains("rolling_shutter")) {
				// the rolling shutter is a float3 of [A,B,C] where the time
				// for each pixel is t= A + B * u + C * v
				// where u and v are the pixel coordinates (0-1),
				// and the resulting t is used to interpolate between the start
				// and end transforms for each training xform
				float motionblur_amount = 0.f;
				if (json["rolling_shutter"].size() >= 4) {
					motionblur_amount = float(json["rolling_shutter"][3]);
				}

				rolling_shutter = {float(json["rolling_shutter"][0]), float(json["rolling_shutter"][1]), float(json["rolling_shutter"][2]), motionblur_amount};
			}

			if (json.contains("ftheta_p0")) {
				camera_distortion.params[0] = json["ftheta_p0"];
				camera_distortion.params[1] = json["ftheta_p1"];
				camera_distortion.params[2] = json["ftheta_p2"];
				camera_distortion.params[3] = json["ftheta_p3"];
				camera_distortion.params[4] = json["ftheta_p4"];
				camera_distortion.params[5] = json["w"];
				camera_distortion.params[6] = json["h"];
				camera_distortion.mode = ECameraDistortionMode::FTheta;
			}
		}

		// In NeuS2
		if (json.contains("aabb_scale")) {
			result.aabb_scale = json["aabb_scale"];
		}

		// In NeuS2
		if (json.contains("offset")) {
			result.offset = // set the offset to be applied to camera positions
				json["offset"].is_array() ?
				Vector3f{float(json["offset"][0]), float(json["offset"][1]), float(json["offset"][2])} :
				Vector3f{float(json["offset"]), float(json["offset"]), float(json["offset"])};
		}

		if (json.contains("aabb")) {
			// map the given aabb of the form [[minx,miny,minz],[maxx,maxy,maxz]] via an isotropic scale and translate to fit in the (0,0,0)-(1,1,1) cube, with the given center at 0.5,0.5,0.5
			const auto& aabb=json["aabb"];
			float length = std::max(0.000001f,std::max(std::max(std::abs(float(aabb[1][0])-float(aabb[0][0])),std::abs(float(aabb[1][1])-float(aabb[0][1]))),std::abs(float(aabb[1][2])-float(aabb[0][2]))));
			result.scale = 1.f/length;
			result.offset = { ((float(aabb[1][0])+float(aabb[0][0]))*0.5f)*-result.scale + 0.5f , ((float(aabb[1][1])+float(aabb[0][1]))*0.5f)*-result.scale + 0.5f,((float(aabb[1][2])+float(aabb[0][2]))*0.5f)*-result.scale + 0.5f};
		}

		// In NeuS2
		// Set Axis-Aligned Bounding Box for all the cameras
		if (json.contains("frames") && json["frames"].is_array()) { // json["frames"]: list of {"file_path", "transform_matrix", "intrinsic_matrix"}
			for (int j = 0; j < json["frames"].size(); ++j) { // loop through each {"file_path", "transform_matrix", "intrinsic_matrix"}
				auto& frame = json["frames"][j]; // each frame: {"file_path", "transform_matrix", "intrinsic_matrix"}

				// load "transform_matrix" (camera to world transformation)
				// In NeuS2 static scenes, jsonmatrix_start and jsonmatrix_end are both just "transform_matrix"
				nlohmann::json& jsonmatrix_start = frame.contains("transform_matrix_start") ? frame["transform_matrix_start"] : frame["transform_matrix"];
				nlohmann::json& jsonmatrix_end = frame.contains("transform_matrix_end") ? frame["transform_matrix_end"] : jsonmatrix_start;

				// Get camera position in world frame
				// Apply scaling and shifting
				//                 Vector3f{float(jsonmatrix_start[0][3]), float(jsonmatrix_start[1][3]), float(jsonmatrix_start[2][3])}: camera origin coordinate in world frame
				// 		   	  p/q = camera origin * scale + offset
				const Vector3f p = Vector3f{float(jsonmatrix_start[0][3]), float(jsonmatrix_start[1][3]), float(jsonmatrix_start[2][3])} * result.scale + result.offset;
				const Vector3f q = Vector3f{float(jsonmatrix_end[0][3]), float(jsonmatrix_end[1][3]), float(jsonmatrix_end[2][3])} * result.scale + result.offset;
				
				// Set the min and max of the bounding box to be the smallest/largest coordinates among all the camera positions after scaling and shifting
				// Initially, min = (inf, inf, inf), max = (-inf, -inf, -inf), so it is guaranteed to be reset by the camera coordinates
				// After enlarging, the min = (min_x, min_y, min_z) will have the smallest xyz values that all the cameras have
				// 				        max = (Max_x, Max_y, Max_z) will have the largest xyz values that all the cameras have
				cam_aabb.enlarge(p);
				cam_aabb.enlarge(q);
			}
		}

		if (json.contains("up")) {
			// axes are permuted as for the xforms below
			result.up[0] = float(json["up"][1]);
			result.up[1] = float(json["up"][2]);
			result.up[2] = float(json["up"][0]);
		}

		if (json.contains("envmap") && result.envmap_resolution.isZero()) {
			std::string json_provided_path = json["envmap"];
			fs::path envmap_path = basepath / json_provided_path;
			if (!envmap_path.exists()) {
				throw std::runtime_error{std::string{"Environment map path "} + envmap_path.str() + " does not exist."};
			}

			if (equals_case_insensitive(envmap_path.extension(), "exr")) {
				result.envmap_data = load_exr(envmap_path.str(), result.envmap_resolution.x(), result.envmap_resolution.y());
				result.is_hdr = true;
			} else {
				result.envmap_data = load_stbi(envmap_path.str(), result.envmap_resolution.x(), result.envmap_resolution.y());
			}
		}

		// In NeuS2
		// Load all the images
		if (json.contains("frames") && json["frames"].is_array()) {
			
			// Load all the images with multiple threads
							  // (start = 0,   end = json["frames"].size() = number of frames provided in json file,                            																																				
			pool.parallelForAsync<size_t>(0, json["frames"].size(), 
			// body = the entire lambda function below
			// Lambda function: Variables that have the ampersand (&) prefix are accessed by reference and variables that don't have it are accessed by value.
			[&progress, &n_loaded, &result, &images, &json, basepath, image_idx, info, rolling_shutter, principal_point, 
			camera_distortion, part_after_underscore, fix_premult, enable_depth_loading, enable_ray_loading] (size_t i) 
			{
				size_t i_img = i + image_idx;
				auto& frame = json["frames"][i]; // For neus2: frame only has {"file_path", "transform_matrix", "intrinsic_matrix"}
				LoadedImageInfo& dst = images[i_img];
				dst = info; // copy defaults, should be empty if nothing is set

				// Load the images
				std::string json_provided_path(frame["file_path"]); // get image path
				if (json_provided_path == "") {
					char buf[256];
					snprintf(buf, 256, "%s_%03d/rgba.png", part_after_underscore.c_str(), (int)i);
					json_provided_path = buf;
				}
				fs::path path = basepath / json_provided_path; // full image path

				if (path.extension() == "") {
					path = path.with_extension("png");
					if (!path.exists()) {
						path = path.with_extension("exr");
					}
					if (!path.exists()) {
						throw std::runtime_error{ "Could not find image file: " + path.str()};
					}
				}

				std::string img_path = path.str(); // get path string
				replace(img_path.begin(),img_path.end(),'\\','/'); // handle slash and back slash delimiters

				int comp = 0; // number of color channels

				// ============= Load the image data using stb library =============
				// If image is .exr format:
				// (An EXR file is a raster image stored in the OpenEXR format, a high dynamic-range (HDR) image format developed by Academy Software Foundation (ASWF))
				if (equals_case_insensitive(path.extension(), "exr")) {
					// dst.pixels = load_exr_to_gpu(&dst.res.x(), &dst.res.y(), path.str().c_str(), fix_premult);
					dst.pixels = load_exr_to_gpu(&dst.res.x(), &dst.res.y(), img_path.c_str(), fix_premult);
					dst.image_type = EImageDataType::Half;
					dst.image_data_on_gpu = true;
					result.is_hdr = true;
				} else { // If image is not .exr format
					dst.image_data_on_gpu = false;

					// Uses the library stb (https://github.com/nothings/stb/tree/master) 
					// Reference: https://solarianprogrammer.com/2019/06/10/c-programming-reading-writing-images-stb_image-libraries/
					// stb_image.h: image loading/decoding from file/memory: JPG, PNG, TGA, BMP, PSD, GIF, HDR, PIC
					// - stbi_load(): read an image from the disk
					//             stbi_load(filename,         width,        height,  number of color channels,  desired number of color channels = 4)
					uint8_t* img = stbi_load(img_path.c_str(), &dst.res.x(), &dst.res.y(), &comp, 4);

					// Get alphapath (not used)
					fs::path alphapath = basepath / (std::string{frame["file_path"]} + ".alpha."s + path.extension());
					if (alphapath.exists()) {
						printf("alphapath exsists: %s\n", alphapath.str().c_str());
						int wa=0,ha=0;
						uint8_t* alpha_img = stbi_load(alphapath.str().c_str(), &wa, &ha, &comp, 4);
						if (!alpha_img) {
							throw std::runtime_error{"Could not load alpha image "s + alphapath.str()};
						}
						ScopeGuard mem_guard{[&]() { stbi_image_free(alpha_img); }};
						if (wa != dst.res.x() || ha != dst.res.y()) {
							throw std::runtime_error{std::string{"Alpha image has wrong resolution: "} + alphapath.str()};
						}
						tlog::success() << "Alpha loaded from " << alphapath;
						for (int i=0;i<dst.res.prod();++i) {
							img[i*4+3] = uint8_t(255.0f*srgb_to_linear(alpha_img[i*4]*(1.f/255.f))); // copy red channel of alpha to alpha.png to our alpha channel
						}
					}

					// Get maskpath (not used)
					fs::path maskpath = path.parent_path()/(std::string{"dynamic_mask_"} + path.basename() + ".png");
					if (maskpath.exists()) {
						printf("alphapath exsists: %s\n", alphapath.str().c_str());
						int wa=0,ha=0;
						uint8_t* mask_img = stbi_load(maskpath.str().c_str(), &wa, &ha, &comp, 4);
						if (!mask_img) {
							throw std::runtime_error{std::string{"Could not load mask image "} + maskpath.str()};
						}
						ScopeGuard mem_guard{[&]() { stbi_image_free(mask_img); }};
						if (wa != dst.res.x() || ha != dst.res.y()) {
							throw std::runtime_error{std::string{"Mask image has wrong resolution: "} + maskpath.str()};
						}
						dst.mask_color = 0x00FF00FF; // HOT PINK
						for (int i = 0; i < dst.res.prod(); ++i) {
							if (mask_img[i*4] != 0) {
								*(uint32_t*)&img[i*4] = dst.mask_color;
							}
						}
					}

					dst.pixels = img; // the image data
					dst.image_type = EImageDataType::Byte;
				}

				// printf("Image type = %d\n", dst.image_type); 
				// ==================== End loading image data =====================

				// If no image data is loaded
				if (!dst.pixels) {
					// throw std::runtime_error{ "image not found: " + path.str() };
					throw std::runtime_error{ "image not found: " + img_path };
				}

				// Get depth_path (not used)
				// enable_depth_loading = true by default
				// info.depth_scale = -1.f by default
				if (enable_depth_loading && info.depth_scale > 0.f && frame.contains("depth_path")) {
					fs::path depthpath = basepath / std::string{frame["depth_path"]};
					if (depthpath.exists()) {
						int wa=0,ha=0;
						dst.depth_pixels = stbi_load_16(depthpath.str().c_str(), &wa, &ha, &comp, 1);
						if (!dst.depth_pixels) {
							throw std::runtime_error{"Could not load depth image "s + depthpath.str()};
						}
						if (wa != dst.res.x() || ha != dst.res.y()) {
							throw std::runtime_error{std::string{"Depth image has wrong resolution: "} + depthpath.str()};
						}
						//tlog::success() << "Depth loaded from " << depthpath;
					}
				}

				// Get rayspath (not used)
				fs::path rayspath = path.parent_path()/(std::string{"rays_"} + path.basename() + ".dat");
				if (enable_ray_loading && rayspath.exists()) {
					printf("Loading rays...\n");
					uint32_t n_pixels = dst.res.prod();
					dst.rays = (Ray*)malloc(n_pixels * sizeof(Ray));

					std::ifstream rays_file{rayspath.str(), std::ios::binary};
					rays_file.read((char*)dst.rays, n_pixels * sizeof(Ray));

					std::streampos fsize = 0;
					fsize = rays_file.tellg();
					rays_file.seekg(0, std::ios::end);
					fsize = rays_file.tellg() - fsize;

					if (fsize > 0) {
						tlog::warning() << fsize << " bytes remaining in rays file " << rayspath;
					}

					for (uint32_t px = 0; px < n_pixels; ++px) {
						result.nerf_ray_to_ngp(dst.rays[px]);
					}
					result.has_rays = true;
				}

				// load "transform_matrix" (camera to world transformation)
				// In NeuS2 static scenes, jsonmatrix_start and jsonmatrix_end are both just "transform_matrix"
				nlohmann::json& jsonmatrix_start = frame.contains("transform_matrix_start") ? frame["transform_matrix_start"] : frame["transform_matrix"];
				nlohmann::json& jsonmatrix_end =   frame.contains("transform_matrix_end") ? frame["transform_matrix_end"] : jsonmatrix_start;

				// (not used)
				if (frame.contains("driver_parameters")) {
					Eigen::Vector3f light_dir(
						frame["driver_parameters"].value("LightX", 0.f),
						frame["driver_parameters"].value("LightY", 0.f),
						frame["driver_parameters"].value("LightZ", 0.f)
					);
					result.metadata[i_img].light_dir = result.nerf_direction_to_ngp(light_dir.normalized());
					result.has_light_dirs = true;
					result.n_extra_learnable_dims = 0;
				}

				// lambda function to read focal length
				auto read_focal_length = [&](int resolution, const std::string& axis) {
					if (frame.contains(axis + "_fov")) { // (not used)
						return fov_to_focal_length(resolution, (float)frame[axis + "_fov"]);
					} else if (json.contains("fl_"s + axis)) { // (not used)
						return (float)json["fl_"s + axis];
					} else if (json.contains("camera_angle_"s + axis)) { // (not used)
						return fov_to_focal_length(resolution, (float)json["camera_angle_"s + axis] * 180 / PI());
					} else {
						// printf("read_focal_length returns 0.0f\n"); // debug
						return 0.0f;
					}
				};

				// x_fov is in degrees, camera_angle_x in radians. Yes, it's silly.
				float x_fl = read_focal_length(dst.res.x(), "x"); // x focal length = 0
				float y_fl = read_focal_length(dst.res.y(), "y"); // y focal length = 0


				// Set result.metadata
				if (x_fl != 0) {
					result.metadata[i_img].focal_length = Vector2f::Constant(x_fl);
					if (y_fl != 0) {
						result.metadata[i_img].focal_length.y() = y_fl;
					}
				} else if (y_fl != 0) {
					result.metadata[i_img].focal_length = Vector2f::Constant(y_fl);
				} else { // x focal length and y focal length are all 0's, use intrinsic matrix (NeuS2 used)
					if (frame.contains("intrinsic_matrix")){
						const auto& intrinsic = frame["intrinsic_matrix"]; // get intrinsic matrix

						// Get fx
						result.metadata[i_img].focal_length.x() = float(intrinsic[0][0]);

						// Get fy
						result.metadata[i_img].focal_length.y() = float(intrinsic[1][1]);

						// Get principal points
						// principal point is the intersection of camera z-axis and image plane (where the principal planes cross the optical axis)
						result.metadata[i_img].principal_point.x() = float(intrinsic[0][2])/(float)json["w"];
						result.metadata[i_img].principal_point.y() = float(intrinsic[1][2])/(float)json["h"];
					}
					else{
						throw std::runtime_error{"Couldn't read fov."};
					}
				}

				// Set result.xforms[i_img] (transformation matrix from camera to world)
				for (int m = 0; m < 3; ++m) { // m = 0, 1, 2
					for (int n = 0; n < 4; ++n) { // n = 0, 1, 2, 3
						// In NeuS2 static scenes, jsonmatrix_start and jsonmatrix_end are both just "transform_matrix"
						// Here, we are just taking the first three rows of "transform_matrix" and set result.xforms[i_img]
						// transform_matrix[0][0], transform_matrix[0][1], transform_matrix[0][2], transform_matrix[0][3]
						// transform_matrix[1][0], transform_matrix[1][1], transform_matrix[1][2], transform_matrix[1][3]
						// transform_matrix[2][0], transform_matrix[2][1], transform_matrix[2][2], transform_matrix[2][3]

						result.xforms[i_img].start(m, n) = float(jsonmatrix_start[m][n]);
						result.xforms[i_img].end(m, n) = float(jsonmatrix_end[m][n]);
					}
				}

				// Set rolling shutter (https://en.wikipedia.org/wiki/Rolling_shutter)
				// By default: rolling_shutter = Vector4f::Zero()
				result.metadata[i_img].rolling_shutter = rolling_shutter;

				// result.metadata[i_img].principal_point = principal_point;

				// Set camera distortion
				// By default: camera_distortion = {}
				result.metadata[i_img].camera_distortion = camera_distortion;


				// Apply scaling and shifring to result.xforms[i_img] (transformation matrix from camera to world)
				// Convert nerf matrix to ngp matrix (apply scaling and shifting)
				result.xforms[i_img].start = result.nerf_matrix_to_ngp(result.xforms[i_img].start);
				result.xforms[i_img].end = result.nerf_matrix_to_ngp(result.xforms[i_img].end);

				// Update progress for printing 
				progress.update(++n_loaded);
			}, 

			futures); // futures = futures, provides a mechanism to access the result of asynchronous operations:

		}

		// In NeuS2
		// Update the number of images
		if (json.contains("frames")) {
			image_idx += json["frames"].size();
		}

	}

	waitAll(futures);

	tlog::success() << "Loaded " << images.size() << " images after " << tlog::durationToString(progress.duration());
	tlog::info() << "  cam_aabb=" << cam_aabb;

	if (result.has_rays) { // By default, result.has_rays = false (not used)
		tlog::success() << "Loaded per-pixel rays.";
	}
	if (!images.empty() && images[0].mask_color) {
		tlog::success() << "Loaded dynamic masks.";
	}

	result.sharpness_resolution = { 128, 72 };

									// 128 * 72 * num_images
	result.sharpness_data.enlarge( result.sharpness_resolution.x() * result.sharpness_resolution.y() *  result.n_images );

	// copy / convert images to the GPU
	for (uint32_t i = 0; i < result.n_images; ++i) {
		// i is the id of each image
		const LoadedImageInfo& m = images[i];
		
		// Set the i-th training image
		result.set_training_image(i, m.res, m.pixels, m.depth_pixels, m.depth_scale * result.scale, 
								  m.image_data_on_gpu, m.image_type, EDepthDataType::UShort, sharpen_amount, 
								  m.white_transparent, m.black_transparent, m.mask_color, m.rays);
		CUDA_CHECK_THROW(cudaDeviceSynchronize());
	}
	CUDA_CHECK_THROW(cudaDeviceSynchronize());
	
	// free memory
	for (uint32_t i = 0; i < result.n_images; ++i) {
		if (images[i].image_data_on_gpu) {
			CUDA_CHECK_THROW(cudaFree(images[i].pixels));
		} else {
			free(images[i].pixels);
		}
		free(images[i].rays);
		free(images[i].depth_pixels);
	}

	return result;
}

void NerfDataset::set_training_image(int frame_idx, const Eigen::Vector2i& image_resolution, const void* pixels, 
									const void* depth_pixels, float depth_scale, bool image_data_on_gpu, 
									EImageDataType image_type, EDepthDataType depth_type, float sharpen_amount, 
									bool white_transparent, bool black_transparent, uint32_t mask_color, 
									const Ray *rays) 
{
	if (frame_idx < 0 || frame_idx >= n_images) {
		throw std::runtime_error{"NerfDataset::set_training_image: invalid frame index"};
	}
	size_t n_pixels = image_resolution.prod(); // height x width
	size_t img_size = n_pixels * 4; // 4 channels: height x width x 4
	size_t image_type_stride = image_type_size(image_type); // EImageDataType::Byte = 1

	// copy to gpu if we need to do a conversion
	GPUMemory<uint8_t> images_data_gpu_tmp;
	GPUMemory<uint8_t> depth_tmp;
	if (!image_data_on_gpu && image_type == EImageDataType::Byte) {
		images_data_gpu_tmp.resize(img_size * image_type_stride);
		images_data_gpu_tmp.copy_from_host((uint8_t*)pixels);
		pixels = images_data_gpu_tmp.data();

		if (depth_pixels) {
			depth_tmp.resize(n_pixels * depth_type_size(depth_type));
			depth_tmp.copy_from_host((uint8_t*)depth_pixels);
			depth_pixels = depth_tmp.data();
		}

		image_data_on_gpu = true;
	}

	// copy or convert the pixels
	pixelmemory[frame_idx].resize(img_size * image_type_size(image_type));
	void* dst = pixelmemory[frame_idx].data();

	// image_type = EImageDataType::Byte is used
	// convert_rgba32() is called (apply modifications on the pixels, in NeuS2 actually nothing is actually modified, just copying "pixels" to "dst")
	switch (image_type) {
		default: throw std::runtime_error{"unknown image type in set_training_image"};
		case EImageDataType::Byte: 
			// GPU parallel computing for all the pixels in the image to call convert_rgba32()
			// kernel = convert_rgba32( n_pixels,  pixels, dst, white_transparent, black_transparent, mask_color), 
			//                     	shmem_size = 0
			// 								stream = nullptr
			// 									n_elements = n_pixels (number of pixels)
			// 													.. args = {pixels, dst, white_transparent, black_transparent, mask_color}
			linear_kernel(convert_rgba32, 0, nullptr, n_pixels, (uint8_t*)pixels, (uint8_t*)dst, 
						  										white_transparent, black_transparent, mask_color); 
			break;
		case EImageDataType::Half: // fallthrough is intended
		case EImageDataType::Float: CUDA_CHECK_THROW(cudaMemcpy(dst, pixels, img_size * image_type_size(image_type), image_data_on_gpu ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice)); break;
	}

	// copy over depths if provided (not used)
	if (depth_scale >= 0.f) {
		depthmemory[frame_idx].resize(img_size);
		float* depth_dst = depthmemory[frame_idx].data();

		if (depth_pixels && !image_data_on_gpu) {
			depth_tmp.resize(n_pixels * depth_type_size(depth_type));
			depth_tmp.copy_from_host((uint8_t*)depth_pixels);
			depth_pixels = depth_tmp.data();
		}

		switch (depth_type) {
			default: throw std::runtime_error{"unknown depth type in set_training_image"};
			case EDepthDataType::UShort: linear_kernel(copy_depth<uint16_t>, 0, nullptr, n_pixels, depth_dst, (const uint16_t*)depth_pixels, depth_scale); break;
			case EDepthDataType::Float: linear_kernel(copy_depth<float>, 0, nullptr, n_pixels, depth_dst, (const float*)depth_pixels, depth_scale); break;
		}
	} else {
		depthmemory[frame_idx].free_memory();
	}

	// apply requested sharpening (this will not be executed)
	// sharpen_amout = 0 by default
	if (sharpen_amount > 0.f) {
		printf("sharpen_amout > 0.f, sharpen_amount = %f\n", sharpen_amount);
		if (image_type == EImageDataType::Byte) {
			tcnn::GPUMemory<uint8_t> images_data_half(img_size * sizeof(__half));
			linear_kernel(from_rgba32<__half>, 0, nullptr, n_pixels, (uint8_t*)pixels, (__half*)images_data_half.data(), white_transparent, black_transparent, mask_color);
			pixelmemory[frame_idx] = std::move(images_data_half);
			dst = pixelmemory[frame_idx].data();
			image_type = EImageDataType::Half;
		}

		assert(image_type == EImageDataType::Half || image_type == EImageDataType::Float);

		tcnn::GPUMemory<uint8_t> images_data_sharpened(img_size * image_type_size(image_type));

		float center_w = 4.f + 1.f / sharpen_amount; // center_w ranges from 5 (strong sharpening) to infinite (no sharpening)
		if (image_type == EImageDataType::Half) {
			linear_kernel(sharpen<__half>, 0, nullptr, n_pixels, image_resolution.x(), (__half*)dst, (__half*)images_data_sharpened.data(), center_w, 1.f / (center_w - 4.f));
		} else {
			linear_kernel(sharpen<float>, 0, nullptr, n_pixels, image_resolution.x(), (float*)dst, (float*)images_data_sharpened.data(), center_w, 1.f / (center_w - 4.f));
		}

		pixelmemory[frame_idx] = std::move(images_data_sharpened);
		dst = pixelmemory[frame_idx].data();
	}

	// sharpness_data (this will be executed)
	// sharpness_data.size() = 128 * 72 * num_images
	if (sharpness_data.size()>0) {
		// printf("sharpness_data.size() > 0, sharpness_data.size() = %d\n", sharpness_data.size());
		// compute overall sharpness
		const dim3 threads = { 16, 8, 1 };
		const dim3 blocks = { div_round_up((uint32_t)sharpness_resolution.x(), threads.x), div_round_up((uint32_t)sharpness_resolution.y(), threads.y), 1 };
		sharpness_data.enlarge(sharpness_resolution.x() * sharpness_resolution.y());
		compute_sharpness<<<blocks, threads, 0, nullptr>>>(sharpness_resolution, image_resolution, 1, dst, image_type, sharpness_data.data() + sharpness_resolution.x() * sharpness_resolution.y() * (size_t)frame_idx);
	}

	// Set metadata for NerfDataset
	metadata[frame_idx].pixels = pixelmemory[frame_idx].data();
	metadata[frame_idx].depth = depthmemory[frame_idx].data();
	metadata[frame_idx].resolution = image_resolution;
	metadata[frame_idx].image_data_type = image_type;
	if (rays) { // (not used)
		raymemory[frame_idx].resize(n_pixels);
		CUDA_CHECK_THROW(cudaMemcpy(raymemory[frame_idx].data(), rays, n_pixels * sizeof(Ray), cudaMemcpyHostToDevice));
	} else {
		raymemory[frame_idx].free_memory();
	}
	metadata[frame_idx].rays = raymemory[frame_idx].data();
}


NGP_NAMESPACE_END
