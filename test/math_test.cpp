#include<cmath>

#include<core/math/vec3.hpp>
#include<core/math/vec3packed.hpp>
#include<core/math/vec4.hpp>

#include<gtest/gtest.h>

using namespace engine::math;

TEST(SimdArch, PrintDetectedArchitecture){
	std::string arch = engine::math::simd::detected_arch();
	std::cout << "[		] SIMD Backend: " << arch << std::endl;
	SUCCEED();
}

TEST(Vec3Test, MemoryLayout){
	EXPECT_EQ(sizeof(Vec3), 16);
	EXPECT_EQ(alignof(Vec3), 16);

	Vec3 v(1.0f, 2.0f, 3.0f);
	EXPECT_FLOAT_EQ(v.x, 1.0f);
	EXPECT_FLOAT_EQ(v.y, 2.0f);
	EXPECT_FLOAT_EQ(v.z, 3.0f);
}
