// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// While the Apache Version 2.0 licence grants copyright and patent licence, it requires written
// notification of any change made to this file. This file has been edited to the point where it
// no longer serves its intended purpose. The boilerplate code of this demonstration code has been
// re-used to facilitate execution and timing of various CLBlast subroutines and variants of it
// can be found in all .cpp files of this repository.
//
// Edited by: Simon Rovder
//
// =================================================================================================

#include <cstdio>
#include <chrono>
#include <vector>

#define CL_USE_DEPRECATED_OPENCL_1_1_APIS // to disable deprecation warnings
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS // to disable deprecation warnings

#include "CL/cl.hpp"

#include <clblast.h>

int main(int argc, char ** argv) {

	// OpenCL platform/device settings
	const auto platform_id = 0;
	const auto device_id = 0;

	// Initializes the OpenCL platform
	auto platforms = std::vector<cl::Platform>();
	cl::Platform::get(&platforms);
	if (platforms.size() == 0 || platform_id >= platforms.size()) { return 1; }
	auto platform = platforms[platform_id];

	// Initializes the OpenCL device
	auto devices = std::vector<cl::Device>();
	platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
	if (devices.size() == 0 || device_id >= devices.size()) { return 1; }
	auto device = devices[device_id];

	// Creates the OpenCL context, queue, and an event
	auto device_as_vector = std::vector<cl::Device>{ device };
	auto context = cl::Context(device_as_vector);
	auto queue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);
	auto event = cl_event{ nullptr };


	auto queue_plain = queue();

	// Prepare the command line arguments. These specify how large the dummy data should be for this test.
	const size_t Xc = atoi(argv[1]);
	const size_t Xh = atoi(argv[2]);
	const size_t Xw = atoi(argv[3]);
	const size_t Fh = atoi(argv[4]);
	const size_t Fw = atoi(argv[5]);
	const size_t Sx = atoi(argv[6]);
	const size_t Sy = atoi(argv[7]);

	const size_t Yw = (Xw - Fw) / Sx + 1;
	const size_t Yh = (Xh - Fh) / Sy + 1;

	// Initialize the dummy data.
	auto host_x = std::vector<float>(Xc*Xw*Xh);
	auto host_i = std::vector<float>(Yw*Yh*Fw*Fh*Xc);
	for (auto &item : host_x) { item = -0.24; }
	for (auto &item : host_i) { item = 0.784; }

	// Copy the data to the device
	// Copy the data to the device
	auto device_x = cl::Buffer(context, CL_MEM_READ_WRITE, host_x.size() * sizeof(float));
	auto device_i = cl::Buffer(context, CL_MEM_READ_WRITE, host_i.size() * sizeof(float));
	queue.enqueueWriteBuffer(device_x, CL_TRUE, 0, host_x.size() * sizeof(float), host_x.data());
	queue.enqueueWriteBuffer(device_i, CL_TRUE, 0, host_i.size() * sizeof(float), host_i.data());
	queue.flush();

    // Run the subroutine
	auto start_time = std::chrono::steady_clock::now();
	auto im2ColStatus = clblast::Im2col<float>(Xc, Xh, Xw, Fh, Fw, 0, 0, Sy, Sx, 1, 1, device_x(), 0, device_i(), 0, &queue_plain, &event);

	// Record the execution time
	if (im2ColStatus != clblast::StatusCode::kSuccess) {
		printf("ERROR! Im2Col Status : %d\n", static_cast<int>(im2ColStatus));
		exit(-1);
	}

	clWaitForEvents(1, &event);
	queue.flush();

	auto elapsed_time = std::chrono::steady_clock::now() - start_time;
	auto time_ms = std::chrono::duration<double, std::micro>(elapsed_time).count();
	printf("%.3lf microseconds - Im2Col with status: %d\n", time_ms, static_cast<int>(im2ColStatus));

	clReleaseEvent(event);

	return 0;
}
