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
	const size_t m = atoi(argv[1]);
	const size_t k = atoi(argv[2]);
	const float alpha = 1.0f;
	const float beta = 1.0f;
	const auto a_ld = k;
	const auto b_ld = 1;
	const auto c_ld = 1;

    // Initialize the dummy data.
	auto host_a = std::vector<float>(m*k);
	auto host_b = std::vector<float>(k);
	auto host_c = std::vector<float>(m);
	for (auto &item : host_a) { item = 0.784; }
	for (auto &item : host_b) { item = -2.199f; }
	for (auto &item : host_c) { item = 0.0f; }

	// Copy the data to the device
	auto device_a = cl::Buffer(context, CL_MEM_READ_WRITE, host_a.size() * sizeof(float));
	auto device_b = cl::Buffer(context, CL_MEM_READ_WRITE, host_b.size() * sizeof(float));
	auto device_c = cl::Buffer(context, CL_MEM_READ_WRITE, host_c.size() * sizeof(float));
	queue.enqueueWriteBuffer(device_a, CL_TRUE, 0, host_a.size() * sizeof(float), host_a.data());
	queue.enqueueWriteBuffer(device_b, CL_TRUE, 0, host_b.size() * sizeof(float), host_b.data());
	queue.enqueueWriteBuffer(device_c, CL_TRUE, 0, host_c.size() * sizeof(float), host_c.data());
	queue.flush();

    // Run the subroutine
	auto start_time = std::chrono::steady_clock::now();
	auto status = clblast::Gemv(clblast::Layout::kRowMajor,
		clblast::Transpose::kNo,
		m, k,
		alpha,
		device_a(), 0, a_ld,
		device_b(), 0, b_ld,
		beta,
		device_c(), 0, c_ld,
		&queue_plain, &event);



	// Record the execution time
	if (status == clblast::StatusCode::kSuccess) {

		clWaitForEvents(1, &event);
		cl_int event_status;
		clGetEventInfo(event, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(event_status), &event_status, NULL);

		auto elapsed_time = std::chrono::steady_clock::now() - start_time;
		auto time_ms = std::chrono::duration<double, std::micro>(elapsed_time).count();
		printf("%.3lf microseconds - Fully Connected Layer in with status: %d\n", time_ms, static_cast<int>(status));

		clReleaseEvent(event);
	}

	return 0;
}
