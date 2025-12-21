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

TEST(Vec3Test, Arithmetic){
	Vec3 a(1.0f, 2.0f, 3.0f);
	Vec3 b(4.0f, 5.0f, 6.0f);

	Vec3 sum = a+b;
	EXPECT_FLOAT_EQ(sum.x,5.0f);
	EXPECT_FLOAT_EQ(sum.y,7.0f);
	EXPECT_FLOAT_EQ(sum.z,9.0f);

	Vec3 diff = a-b;
	EXPECT_FLOAT_EQ(diff.x,-3.0f);
	EXPECT_FLOAT_EQ(diff.y,-3.0f);
	EXPECT_FLOAT_EQ(diff.z,-3.0f);

	Vec3 scaled = a*2.0f;
	EXPECT_FLOAT_EQ(scaled.x,2.0f);
	EXPECT_FLOAT_EQ(scaled.y,4.0f);
	EXPECT_FLOAT_EQ(scaled.z,6.0f);

	Vec3 mul = a*b;
	EXPECT_FLOAT_EQ(mul.x,4.0f);
	EXPECT_FLOAT_EQ(mul.y,10.0f);
	EXPECT_FLOAT_EQ(mul.z,18.0f);
}

TEST(Vec3Test, DotProduct){
	float ax = 0.5f;
	float ay = 4.1f;
	float az = 3.0f;

	float bx = 44.0f;
	float by = 2.3f;
	float bz = 3.5f;

	Vec3 a(ax,ay,az);
	Vec3 b(bx,by,bz);

	float dot = ax*bx + ay*by + az*bz;

	EXPECT_FLOAT_EQ(a.dot(b), dot);

	Vec3 zone = Vec3{0.0f,0.0f,1.0f};
	Vec3 yone = Vec3{0.0f,1.0f,0.0f};

	EXPECT_FLOAT_EQ(zone.dot(yone), 0.0f);
}

TEST(Vec3Test, CrossProduct){
	Vec3 right(1.0f, 0.0f, 0.0f);
	Vec3 up(0.0f, 1.0f, 0.0f);

	Vec3 forward = right.cross(up);

	EXPECT_FLOAT_EQ(forward.x, 0.0f);
	EXPECT_FLOAT_EQ(forward.y, 0.0f);
	EXPECT_FLOAT_EQ(forward.z, 1.0f);

	Vec3 back = up.cross(right);
	EXPECT_FLOAT_EQ(back.z, -1.0f);
}

TEST(Vec3Test, Normalization){
	Vec3 v(3.0f, 0.0f, 0.0f);
	Vec3 nf = v.normalized_fast();

	Vec3 res(1.0f,0.0f,0.0f);
	EXPECT_TRUE(nf.is_close(res,1e-3f));

	Vec3 n = v.normalized();
	EXPECT_TRUE(n.is_close(res,1e-7f));
}

TEST(Vec3Test, Comparison){
	Vec3 a(1.1f,2.2f,3.3f);
	Vec3 b(1.1f,2.2f,3.3f);
	Vec3 c(1.1f,2.2f,3.4f);

	EXPECT_TRUE(a == b);
	EXPECT_TRUE(a != c);

	Vec3 d(1.10001f, 2.2f, 3.3f);
	EXPECT_TRUE(a.is_close(d,1e-4f));
	EXPECT_FALSE(a.is_close(d,1e-6f));
}


TEST(Vec4Test, MemoryLayout){
	EXPECT_EQ(sizeof(Vec4), 16);
	EXPECT_EQ(alignof(Vec4), 16);

	Vec4 v(1,2,3,4);
	EXPECT_FLOAT_EQ(v.w,4.0f);
}

TEST(Vec4Test, Arithmetic){
	Vec4 a(1,2,3,4);
	Vec4 b(10,20,30,40);

	Vec4 sum = a + b;
	EXPECT_FLOAT_EQ(sum.x, 11.0f);
	EXPECT_FLOAT_EQ(sum.w, 44.0f);

	Vec4 mul = a*b;
	EXPECT_FLOAT_EQ(mul.y, 40.0f);
	EXPECT_FLOAT_EQ(mul.z, 90.0f);
}

TEST(Vec4Test, DotProduct){
	float ax = 0.5f;
	float ay = 4.1f;
	float az = 3.0f;
	float aw = 66.0f;

	float bx = 44.0f;
	float by = 2.3f;
	float bz = 3.5f;
	float bw = 0.4f;

	Vec4 a(ax,ay,az,aw);
	Vec4 b(bx,by,bz,bw);

	float dot = ax*bx + ay*by + az*bz + aw*bw;

	EXPECT_FLOAT_EQ(a.dot(b), dot);

	Vec4 zone = Vec4{0.0f,0.0f,1.0f,0.0f};
	Vec4 ywone = Vec4{0.0f,1.0f,0.0f,1.0f};

	EXPECT_FLOAT_EQ(zone.dot(ywone), 0.0f);
}

TEST(Vec4Test, Comparison){
	Vec4 a(1,2,3,4);
	Vec4 b(1,2,3,5);

	EXPECT_FALSE(a==b);

	Vec4 c(1,2,3,4);
	EXPECT_TRUE(a==c);

	Vec4 d(1,2,3,3.999f);
	EXPECT_TRUE(a.is_close(d,1e-3f));
}

TEST(Vec4Test, FromVec3){
	Vec3 v3(1.0f, 2.0f, 3.0f);
	Vec4 v4(v3);

	EXPECT_FLOAT_EQ(v4.x, 1.0f);
	EXPECT_FLOAT_EQ(v4.y, 2.0f);
	EXPECT_FLOAT_EQ(v4.z, 3.0f);
	EXPECT_FLOAT_EQ(v4.w, 0.0f);
}
