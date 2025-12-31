#include<cmath>

#include<core/math/vec3.hpp>
#include<core/math/vec3packed.hpp>
#include<core/math/vec4.hpp>
#include<core/math/mat4.hpp>

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

void ExpectMat4Near(
		const Mat4& actual,
		const float expected[16],
		float abs_err = 1e-5f){
	for(int col = 0; col < 4; ++col){
		Vec4 actual_data = actual.cols[col];
		EXPECT_NEAR(actual_data.x, expected[col*4 + 0], abs_err)
			<< "Error at col " << col << ", row 0";
		EXPECT_NEAR(actual_data.y, expected[col*4 + 1], abs_err)
			<< "Error at col " << col << ", row 1";
		EXPECT_NEAR(actual_data.z, expected[col*4 + 2], abs_err)
			<< "Error at col " << col << ", row 2";
		EXPECT_NEAR(actual_data.w, expected[col*4 + 3], abs_err)
			<< "Error at col " << col << ", row 3";
	}
}

TEST(Mat4Test, Identity){
	Mat4 id = Mat4::identity();
	float expected[16] = {
		1,0,0,0,
		0,1,0,0,
		0,0,1,0,
		0,0,0,1
	};
	ExpectMat4Near(id, expected);
}

TEST(Mat4Test, MulByVector){
	Mat4 translation = Mat4::identity();
	translation.cols[3] = Vec4(10.0f, 20.0f, 30.0f, 1.0f);

	Vec4 point(1.0f, 2.0f, 3.0f, 1.0f);
	Vec4 result = translation * point;

	Vec4 expected{11.0f, 22.0f, 33.0f, 1.0f};

	EXPECT_TRUE(result.is_close(expected));
}

TEST(Mat4Test, MatMul){
	Mat4 S = Mat4(Vec4(2,0,0,0), Vec4(0,2,0,0), Vec4(0,0,2,0), Vec4(0,0,0,1));
	Mat4 T = Mat4::identity();

	T.cols[3] = Vec4(10,20,30,1);

	Mat4 result = S*T;

	Vec4 res = result.cols[3];;
	Vec4 expected = Vec4{20.0f, 40.0f, 60.0f, 1.0f};
	EXPECT_TRUE(res.is_close(expected,1e-8f));
}

TEST(Mat4Test, Transpose){
	Mat4 m(
		Vec4(1,2,3,4),
		Vec4(5,6,7,8),
		Vec4(9,10,11,12),
		Vec4(13,14,15,16)
	);

	Mat4 mt = m.transpose();

	Vec4 col0 = mt.cols[0];
	EXPECT_FLOAT_EQ(col0.x, 1.0f);
	EXPECT_FLOAT_EQ(col0.y, 5.0f);
	EXPECT_FLOAT_EQ(col0.z, 9.0f);
	EXPECT_FLOAT_EQ(col0.w, 13.0f);

}

TEST(Mat4Test, PerspectiveProjection){
	float fov = 1.0472f;
	float aspect = 1.777f;
	float near = 0.1f;
	float far = 100.0f;

	Mat4 P = Mat4::perspective(fov, aspect, near, far);

	Vec4 point_near(0.0f, 0.0f, -near, 1.0f);
	Vec4 res_near = P * point_near;

	float w = res_near.w;
	EXPECT_NEAR(res_near.z / w, 0.0f, 1e-5f);
	
	Vec4 point_far(0.0f, 0.0f, -far, 1.0f);
	Vec4 res_far = P * point_far;

	float w_far = res_far.w;
	EXPECT_NEAR(res_far.z / w_far, 1.0f, 1e-5f);
}

TEST(Mat4Test, MulAssociativity){
	Mat4 A = Mat4::perspective(1.0f,1.0f,0.1f,10.0f);
	Mat4 B = Mat4::identity();
	B.cols[3] = Vec4(1,2,3,1);
	Mat4 C = Mat4(Vec4(2,0,0,0), Vec4(0,2,0,0), Vec4(0,0,2,0), Vec4(0,0,0,1));

	Mat4 res1 = (A*B)*C;
	Mat4 res2 = A*(B*C);

	for(int i = 0; i < 4; ++i){
		EXPECT_TRUE(res1.cols[i].is_close(res2.cols[i], 1e-5f));
	}
}

TEST(Mat4Test, TransposeFull){
	Mat4 m(
		Vec4(1, 2, 3, 4),
		Vec4(5, 6, 7, 8),
		Vec4(9, 10, 11, 12),
		Vec4(13, 14, 15, 16)
	);
	Mat4 mt = m.transpose();

	float expected[16] = {
		1, 5, 9, 13,
		2, 6, 10, 14,
		3, 7, 11, 15,
		4, 8, 12, 16
	};
	ExpectMat4Near(mt, expected);
}

TEST(Mat4Test, InverseNoScale){
	Mat4 transform = Mat4::identity();
	transform.cols[0] = Vec4(0,0,-1,0);
	transform.cols[1] = Vec4(0,1,0,0);
	transform.cols[2] = Vec4(1,0,0,0);
	transform.cols[3] = Vec4(10,20,30,1);

	Mat4 inv = transform.inverse_transform_no_scale();

	Mat4 identity_check = transform * inv;

	float expected_id[16] = {
		1,0,0,0,
		0,1,0,0,
		0,0,1,0,
		0,0,0,1
	};

	ExpectMat4Near(identity_check, expected_id, 1e-5f);
}

TEST(Mat4Test, InverseWithScale){
	Mat4 scale_mat = Mat4::identity();
	scale_mat.cols[0] = Vec4(2,0,0,0);
	scale_mat.cols[1] = Vec4(0,0.5f,0,0);
	scale_mat.cols[2] = Vec4(0,0,4,0);
	scale_mat.cols[3] = Vec4(10, -5, 20, 1);

	Mat4 inv = scale_mat.inverse_transform();

	Mat4 res = scale_mat * inv;

	float expected_id[16] = {
		1,0,0,0,
		0,1,0,0,
		0,0,1,0,
		0,0,0,1
	};

	ExpectMat4Near(res, expected_id, 1e-5f);
}

TEST(Mat4Test, RigidBodyBackAndForth){
	Mat4 T = Mat4::identity();
	T.cols[3] = Vec4(5,10,15,1);

	float s = 0.7071f;
	float c = 0.7071f;
	Mat4 R = Mat4::identity();
	R.cols[1] = Vec4(0, c, s, 0);
	R.cols[2] = Vec4(0, -s, c, 0);

	Mat4 M = R * T;
	Mat4 invM = M.inverse_transform();

	Vec4 point(1,2,3,1);
	Vec4 transformed = M*point;
	Vec4 back = invM * transformed;

	EXPECT_TRUE(back.is_close(point, 1e-5f));
}

TEST(Mat4Test, InversePureTranslation){
	Mat4 T = Mat4::identity();
	T.cols[3] = Vec4(100.0f, -50.0f, 25.0f, 1.0f);

	Mat4 inv = T.inverse_transform_no_scale();

	EXPECT_FLOAT_EQ(inv.cols[3].x, -100.0f);
	EXPECT_FLOAT_EQ(inv.cols[3].y, 50.0f);
	EXPECT_FLOAT_EQ(inv.cols[3].z, -25.0f);

	Mat4 res = T*inv;
	float id[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
	ExpectMat4Near(res,id);
}

TEST(Mat4Test, InversePureRotation){
	Mat4 R = Mat4::identity();
	R.cols[0] = Vec4(0,1,0,0);
	R.cols[1] = Vec4(-1,0,0,0);
	R.cols[2] = Vec4(0,0,1,0);

	Mat4 inv = R.inverse_transform_no_scale();

	EXPECT_FLOAT_EQ(inv.cols[0].y, -1.0f);
	EXPECT_FLOAT_EQ(inv.cols[1].x, 1.0f);

	Mat4 res = R * inv;
	float id[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
	ExpectMat4Near(res,id);
}

TEST(Mat4Test, InverseNegativeScale) {
	Mat4 M = Mat4::identity();
	M.cols[0] = Vec4(-1, 0, 0, 0);
	M.cols[3] = Vec4(10, 0, 0, 1);

	Mat4 inv = M.inverse_transform();

	Mat4 res = M * inv;
	float id[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
	ExpectMat4Near(res, id);

	Vec4 p(5, 0, 0, 1);
	EXPECT_TRUE((inv * (M * p)).is_close(p));
}

TEST(Mat4Test, InverseFullComplexTransform) {
	float s = 0.707106f;
	Mat4 R = Mat4::identity();
	R.cols[0] = Vec4(s, 0, -s, 0);
	R.cols[1] = Vec4(0, 1, 0, 0);
	R.cols[2] = Vec4(s, 0, s, 0);

	Mat4 S = Mat4::identity();
	S.cols[0] = S.cols[0] * 2.0f;
	S.cols[1] = S.cols[1] * 0.5f;
	S.cols[2] = S.cols[2] * 3.0f;

	Mat4 T = Mat4::identity();
	T.cols[3] = Vec4(10, 20, 30, 1);

	Mat4 M = T * (R * S);
	Mat4 invM = M.inverse_transform();

	Mat4 res = M * invM;
	float id[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
	ExpectMat4Near(res, id, 1e-4f);
}

TEST(Mat4Test, InverseGeneralMatrix){
	Mat4 M(
		Vec4(1,2,0,0),
		Vec4(0,3,0,0),
		Vec4(1,0,1,1),
		Vec4(0,0,2,1)
	);

	Mat4 inv = M.inverse();
	Mat4 res = M*inv;

	float id[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
	ExpectMat4Near(res, id, 1e-5);

}

TEST(Mat4Test, InverseProperty){
	// (A*B)^(-1) = B^(-1) * A^(-1)
	
	Mat4 A = Mat4::identity();
	A.cols[3] = Vec4(1,2,3,1);

	Mat4 B = Mat4::identity();
	B.cols[0] = Vec4(2,0,0,0);
	B.cols[1] = Vec4(0,2,0,0);
	B.cols[2] = Vec4(0,0,2,0);

	Mat4 invAB = (A*B).inverse();
	Mat4 invB_invA = B.inverse() * A.inverse();

	for(int i = 0; i < 4; ++i){
		EXPECT_TRUE(invAB.cols[i].is_close(invB_invA.cols[i], 1e-5f))
			<< "Asoociativity failed at column " << i;
	}
}

TEST(Mat4Test, InverseIdentity) {
	Mat4 id = Mat4::identity();
	Mat4 inv = id.inverse();

	float expected[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
	ExpectMat4Near(inv, expected);
}

TEST(Mat4Test, InverseSimple){
	Mat4 M = Mat4::identity();
	M.cols[0].x = 1.0f;
	M.cols[1].y = 2.0f;
	M.cols[2].z = 4.0f;
	M.cols[3].w = 8.0f;

	Mat4 inv = M.inverse();

	EXPECT_FLOAT_EQ(inv.cols[0].x, 1.0f);
	EXPECT_FLOAT_EQ(inv.cols[1].y, 0.5f);
	EXPECT_FLOAT_EQ(inv.cols[2].z, 0.25f);
	EXPECT_FLOAT_EQ(inv.cols[3].w, 0.125f);
}

TEST(Mat4Test, InverseSingularMatrix) {
	Mat4 M = Mat4::identity();
	M.cols[0] = Vec4(0, 0, 0, 0); 

	Mat4 inv = M.inverse();

	EXPECT_TRUE(std::isnan(inv.cols[0].x) || std::isinf(inv.cols[0].x));
}

TEST(Mat4Inverse, RotationX) {
	float angle = 0.523599f; // 30 deg
	float s = std::sin(angle);
	float c = std::cos(angle);

	Mat4 R = Mat4::identity();
	R.cols[0] = Vec4(c, s, 0, 0);
	R.cols[1] = Vec4(-s, c, 0,0);

	Mat4 invR = R.inverse();
	Mat4 res = R * invR;

	float identity[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
	ExpectMat4Near(res, identity, 1e-6f);
}
