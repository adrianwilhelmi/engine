#include<cmath>

#include<core/math/vec3.hpp>
#include<core/math/vec3packed.hpp>
#include<core/math/vec4.hpp>
#include<core/math/mat4.hpp>
#include<core/math/quat.hpp>

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

TEST(Vec3Test, Getter){
	float x = 4.333;
	float y = 2.0;
	float z = 3.333;
	Vec3 a(x, y, z);

	EXPECT_FLOAT_EQ(a.get_x(), x);
	EXPECT_FLOAT_EQ(a.get_y(), y);
	EXPECT_FLOAT_EQ(a.get_z(), z);
}

TEST(Vec3Test, Setter){
	Vec3 a(0);

	EXPECT_FLOAT_EQ(a.get_x(), 0.f);
	EXPECT_FLOAT_EQ(a.get_y(), 0.f);
	EXPECT_FLOAT_EQ(a.get_z(), 0.f);

	float x = 4.333;
	float y = 2.0;
	float z = 3.333;

	a.set_x(x);
	a.set_y(y);
	a.set_z(z);

	EXPECT_FLOAT_EQ(a.get_x(), x);
	EXPECT_FLOAT_EQ(a.get_y(), y);
	EXPECT_FLOAT_EQ(a.get_z(), z);
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

TEST(Vec4Test, Getter){
	float x = 4.333;
	float y = 2.0;
	float z = 3.333;
	float w = 1234.222;
	Vec4 a(x, y, z, w);

	EXPECT_FLOAT_EQ(a.get_x(), x);
	EXPECT_FLOAT_EQ(a.get_y(), y);
	EXPECT_FLOAT_EQ(a.get_z(), z);
	EXPECT_FLOAT_EQ(a.get_w(), w);
}

TEST(Vec4Test, Setter){
	Vec4 a(0);

	EXPECT_FLOAT_EQ(a.get_x(), 0.f);
	EXPECT_FLOAT_EQ(a.get_y(), 0.f);
	EXPECT_FLOAT_EQ(a.get_z(), 0.f);
	EXPECT_FLOAT_EQ(a.get_w(), 0.f);

	float x = 4.333;
	float y = 2.0;
	float z = 3.333;
	float w = 4.22201;

	a.set_x(x);
	a.set_y(y);
	a.set_z(z);
	a.set_w(w);

	EXPECT_FLOAT_EQ(a.get_x(), x);
	EXPECT_FLOAT_EQ(a.get_y(), y);
	EXPECT_FLOAT_EQ(a.get_z(), z);
	EXPECT_FLOAT_EQ(a.get_w(), w);
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

TEST(Mat4Test, InverseGeneralMatrix2){
	Mat4 M = Mat4::identity();
	M.cols[0] = Vec4(3.0f, 1.0f, 2.0f, 1.0f);
	M.cols[1] = Vec4(2.0f, 0.0f, 1.0f, 1.0f);
	M.cols[2] = Vec4(-1.0f, 1.0f, 1.0f, 1.0f);
	M.cols[3] = Vec4(1.0f, 2.0f, -1.0f, 0.0f);

	Mat4 inv = M.inverse();
	Mat4 res = M * inv;

	float identity[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
	ExpectMat4Near(res, identity, 1e-6f);
}

TEST(Mat4Test, InverseGeneralMatrix3){
	Mat4 M = Mat4::identity();
	M.cols[0] = Vec4(1.0f, 0.0f, 2.0f, 2.0f);
	M.cols[1] = Vec4(0.0f, 2.0f, 1.0f, 0.0f);
	M.cols[2] = Vec4(0.0f, 1.0f, 0.0f, 1.0f);
	M.cols[3] = Vec4(1.0f, 2.0f, 1.0f, 4.0f);

	Mat4 inv = M.inverse();
	Mat4 res = M * inv;

	float identity[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
	ExpectMat4Near(res, identity, 1e-6f);
}

TEST(Mat4Test, InverseGeneralMatrix4){
	Mat4 M = Mat4::identity();
	M.cols[0] = Vec4(2.0f, 1.0f, 7.0f, 1.0f);
	M.cols[1] = Vec4(5.0f, 4.0f, 8.0f, 5.0f);
	M.cols[2] = Vec4(0.0f, 2.0f, 9.0f, 7.0f);
	M.cols[3] = Vec4(8.0f, 6.0f, 3.0f, 8.0f);

	Mat4 inv = M.inverse();
	Mat4 res = M * inv;

	float identity[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
	ExpectMat4Near(res, identity, 1e-6f);
}

TEST(Mat4Test, InverseGeneralMatrix5){
	Mat4 M = Mat4::identity();
	M.cols[0] = Vec4(5.0f, 0.0f, 0.0f, -1.0f);
	M.cols[1] = Vec4(6.0f, -1.0f, 0.0f, 0.0f);
	M.cols[2] = Vec4(6.0f, -4.0f, 2.0f, 0.0f);
	M.cols[3] = Vec4(-1.0f, 0.0f, 0.0f, -12.0f);

	Mat4 inv = M.inverse();
	Mat4 res = M * inv;

	float identity[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
	ExpectMat4Near(res, identity, 1e-6f);
}

TEST(Mat4Test, InverseGeneralMatrix6){
	Mat4 M = Mat4::identity();
	M.cols[0] = Vec4(4.0f, 2.0f, 2.0f, 2.0f);
	M.cols[1] = Vec4(2.0f, 1.0f, 4.0f, 1.0f);
	M.cols[2] = Vec4(-1.0f, 2.0f, 1.0f, 2.0f);
	M.cols[3] = Vec4(3.0f, -1.0f, 1.0f, 1.0f);

	Mat4 inv = M.inverse();
	Mat4 res = M * inv;

	float identity[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
	ExpectMat4Near(res, identity, 1e-6f);
}

TEST(Mat4Test, InverseGeneralMatrix7){
	Mat4 M = Mat4::identity();
	M.cols[0] = Vec4(1.0f, 1.0f, 1.0f, -1.0f);
	M.cols[1] = Vec4(1.0f, 1.0f, -1.0f, 1.0f);
	M.cols[2] = Vec4(1.0f, -1.0f, 1.0f, 1.0f);
	M.cols[3] = Vec4(-1.0f, 1.0f, 1.0f, 1.0f);

	Mat4 inv = M.inverse();
	Mat4 res = M * inv;

	float identity[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
	ExpectMat4Near(res, identity, 1e-6f);
}

TEST(Mat4Test, InverseGeneralMatrix8){
	Mat4 M = Mat4::identity();
	M.cols[0] = Vec4(1.0f, 3.0f, 4.0f, 2.0f);
	M.cols[1] = Vec4(2.0f, 4.0f, 2.0f, 3.0f);
	M.cols[2] = Vec4(4.0f, 0.0f, 1.0f, 2.0f);
	M.cols[3] = Vec4(2.0f, 1.0f, 1.0f, 4.0f);

	Mat4 inv = M.inverse();
	Mat4 res = M * inv;

	float identity[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
	ExpectMat4Near(res, identity, 1e-6f);
}

TEST(Mat4Test, Rotations){
	const float PI_2 = 1.570796f;
	Vec4 unit_y(0.0f, 1.0f, 0.0f, 0.0f);
	Vec4 unit_x(1.0f, 0.0f, 0.0f, 0.0f);

	Mat4 rx = Mat4::rotate_x(PI_2);
	Vec4 res_x = rx * unit_y;
	EXPECT_TRUE(res_x.is_close(Vec4(0,0,1,0), 1e-5f));

	Mat4 ry = Mat4::rotate_y(PI_2);
	Vec4 res_y = ry*unit_x;
	EXPECT_TRUE(res_y.is_close(Vec4(0,0,-1,0), 1e-5f));

	Mat4 rz = Mat4::rotate_z(PI_2);
	Vec4 res_z = rz * unit_x;
	EXPECT_TRUE(res_z.is_close(Vec4(0,1,0,0), 1e-5f));
}

TEST(Mat4Test, RotationInverse){
	float angle = 0.785398f; // 45 deg
	Mat4 rx = Mat4::rotate_x(angle);
	Mat4 rx_inv = Mat4::rotate_x(-angle);

	Mat4 res = rx * rx_inv;

	for(int i = 0; i < 4; ++i){
		for(int j = 0; j < 4; ++j){
			float expected = (i == j) ? 1.0f : 0.0f;
			EXPECT_NEAR(res[i][j], expected, 1e-5f);
		}
	}
}

TEST(Mat4Test, OrthoProjection){
	float l = -10.0f,	r = 10.0f;
	float b = -5.0f,	t = 5.0f;
	float n = 0.1f,		f = 100.1f;

	Mat4 O = Mat4::ortho(l,r,b,t,n,f);

	Vec4 center_near((l+r) / 2.0f, (t+b) / 2.0f, -n, 1.0f);
	Vec4 res_n = O * center_near;
	EXPECT_NEAR(res_n.x, 0.0f, 1e-5f);
	EXPECT_NEAR(res_n.y, 0.0f, 1e-5f);
	EXPECT_NEAR(res_n.z, 0.0f, 1e-5f);

	Vec4 far_top_right(r,t,-f,1.0f);
	Vec4 res_f = O*far_top_right;
	EXPECT_NEAR(res_f.x, 1.0f, 1e-5f);
	EXPECT_NEAR(res_f.y, 1.0f, 1e-5f);
	EXPECT_NEAR(res_f.z, 1.0f, 1e-5f);
}

TEST(Mat4Test, OrthoFrustumCorners){
	float l = -2.0f,	r = 2.0f;
	float b = -1.0f,	t = 1.0f;
	float n = 0.0f,		f = 10.0f;

	Mat4 O = Mat4::ortho(l,r,b,t,n,f);

	Vec4 lbn(l,b,-n,1.0f);
	Vec4 res_lbn = O * lbn;
	EXPECT_NEAR(res_lbn.x, -1.0f, 1e-5f);
	EXPECT_NEAR(res_lbn.y, -1.0f, 1e-5f);
	EXPECT_NEAR(res_lbn.z, 0.0f, 1e-5f);

	Vec4 rtf(r,t,-f,1.0f);
	Vec4 res_rtf = O * rtf;
	EXPECT_NEAR(res_rtf.x, 1.0f, 1e-5f);
	EXPECT_NEAR(res_rtf.y, 1.0f, 1e-5f);
	EXPECT_NEAR(res_rtf.z, 1.0f, 1e-5f);

}

TEST(Mat4Test, LookAt){
	Vec3 eye(0,0,10);
	Vec3 target(0,0,0);
	Vec3 up(0,1,0);

	Mat4 view = Mat4::look_at(eye, target, up);

	Vec4 view_target = view * Vec4(target, 1.0f);
	EXPECT_NEAR(view_target.x, 0.0f, 1e-5f);
	EXPECT_NEAR(view_target.y, 0.0f, 1e-5f);
	EXPECT_NEAR(std::abs(view_target.z), 10.0f, 1e-5f);
}

TEST(Mat4Test, LookAtDiagnoal){
	Vec3 eye(10,10,10);
	Vec3 target(0,0,0);
	Vec3 up(0,1,0);

	Mat4 view = Mat4::look_at(eye, target, up);

	Vec4 res_target = view * Vec4(target, 1.0f);

	EXPECT_NEAR(res_target.x, 0.0f, 1e-5f);
	EXPECT_NEAR(res_target.y, 0.0f, 1e-5f);
	// sqrt(100 + 100 + 100) = ~17.32
	EXPECT_NEAR(std::abs(res_target.z), 17.3205f, 1e-4f);

	Vec4 world_up(0,1,0,0);
	Vec4 view_up = view*world_up;
	EXPECT_GT(view_up.y, 0.0f);
}

TEST(Mat4Test, LookAtVertical){
	Vec3 eye(0,0,0);
	Vec3 target(0,1,0);
	Vec3 up(0,0,1);

	Mat4 view = Mat4::look_at(eye, target, up);
	Vec4 res_target = view * Vec4(target, 1.0f);

	EXPECT_FALSE(std::isnan(res_target.x));
	EXPECT_NEAR(res_target.x, 0.0f, 1e-5f);
}

TEST(QuatTest, Identity){
	Quat id = Quat::identity();
	
	EXPECT_FLOAT_EQ(id.get_x(), 0.f);
	EXPECT_FLOAT_EQ(id.get_y(), 0.f);
	EXPECT_FLOAT_EQ(id.get_z(), 0.f);
	EXPECT_FLOAT_EQ(id.get_w(), 1.f);
}

TEST(QuatTest, Rotation90Z){
	float pi_half = 1.570796f;

	Quat qz = Quat::from_axis_angle(Vec3(0,0,1), pi_half);

	Vec3 v1(1,0,0);
	Vec3 res = qz.rotate(v1);

	EXPECT_NEAR(res.get_x(), 0, 1e-6f);
	EXPECT_NEAR(res.get_y(), 1, 1e-6f);
	EXPECT_NEAR(res.get_z(), 0, 1e-6f);
}

TEST(QuatTest, Mul){
	// rotate 90 deg X, 90 deg Y
	float pi_half = 1.570796f;

	Quat qx = Quat::from_axis_angle(Vec3(1,0,0), pi_half);
	Quat qy = Quat::from_axis_angle(Vec3(0,1,0), pi_half);

	Quat q_combined = qy*qx;

	Vec3 v2(0,0,1);
	Vec3 res = q_combined.rotate(v2);

	EXPECT_NEAR(res.get_x(), 0, 1e-6f);
	EXPECT_NEAR(res.get_y(), -1, 1e-6f);
	EXPECT_NEAR(res.get_z(), 0, 1e-6f);
}

TEST(QuatTest, Conjugate){
	float pi_half = 1.570796f;

	Quat qz = Quat::from_axis_angle(Vec3(0,0,1), pi_half);
	Quat q_inv = qz.conjugated();
	Quat res = qz * q_inv;

	EXPECT_NEAR(res.get_x(), 0, 1e-6f);
	EXPECT_NEAR(res.get_w(), 1, 1e-6f);
}

TEST(QuatTest, ToMat){
	float pi_half = 1.570796f;

	Quat qz = Quat::from_axis_angle(Vec3(0,0,1), pi_half);
	Mat4 m = qz.to_mat4();

	EXPECT_NEAR(m.cols[0].get_x(), 0, 1e-6f);
	EXPECT_NEAR(m.cols[0].get_y(), 1, 1e-6f);
}

TEST(QuatTest, Normalize){
	Quat q(1.0f, 2.0f, 3.0f, 4.0f);
	Quat qn = q.normalized();

	EXPECT_NEAR(qn.l2(), 1.0f, 1e-6f);

	float ratio = qn.get_x() / qn.get_y();
	EXPECT_NEAR(ratio, 1.0f / 2.0f, 1e-6f);
}

TEST(QuatTest, MulOrder){
	Quat q1 = Quat::from_axis_angle(Vec3(1, 0, 0), 1.570796f); // 90 X
	Quat q2 = Quat::from_axis_angle(Vec3(0, 1, 0), 1.570796f); // 90 Y

	Quat q_combined = q2 * q1;
	Vec3 v(0, 0, 1);

	Vec3 res_combined = q_combined.rotate(v);
	Vec3 res_sequential = q2.rotate(q1.rotate(v));

	EXPECT_NEAR(res_combined.get_x(), res_sequential.get_x(), 1e-6f);
	EXPECT_NEAR(res_combined.get_y(), res_sequential.get_y(), 1e-6f);
	EXPECT_NEAR(res_combined.get_z(), res_sequential.get_z(), 1e-6f);
}

TEST(QuatTest, Rotation180Y){
	Quat q = Quat::from_axis_angle(Vec3(0,1,0), 3.141592f);

	Vec3 v(1,0,0);
	Vec3 res = q.rotate(v);

	EXPECT_NEAR(res.get_x(), -1.0f, 1e-6f);
	EXPECT_NEAR(res.get_y(), 0.0f, 1e-6f);
	EXPECT_NEAR(res.get_z(), 0.0f, 1e-6f);
}

TEST(QuatTest, SlerpBoundaries){
	Quat q1 = Quat::from_axis_angle(Vec3(1, 0, 0), 0.0f);
	Quat q2 = Quat::from_axis_angle(Vec3(1, 0, 0), 1.570796f); // 90 deg

	Quat res0 = Quat::slerp(q1, q2, 0.0f);
	EXPECT_NEAR(res0.get_w(), q1.get_w(), 1e-6f);
	EXPECT_NEAR(res0.get_x(), q1.get_x(), 1e-6f);

	Quat res1 = Quat::slerp(q1, q2, 1.0f);
	EXPECT_NEAR(res1.get_w(), q2.get_w(), 1e-6f);
	EXPECT_NEAR(res1.get_x(), q2.get_x(), 1e-6f);
}

TEST(QuatTest, SlerpMidpoint) {
	Quat q1 = Quat::from_axis_angle(Vec3(0, 1, 0), 0.0f);
	Quat q2 = Quat::from_axis_angle(Vec3(0, 1, 0), 1.0f); // 1 radian

	Quat res = Quat::slerp(q1, q2, 0.5f);
	Quat expected = Quat::from_axis_angle(Vec3(0, 1, 0), 0.5f);

	EXPECT_NEAR(res.get_w(), expected.get_w(), 1e-6f);
	EXPECT_NEAR(res.get_y(), expected.get_y(), 1e-6f);
}

TEST(QuatTest, SlerpShortestPath) {
	Quat q1 = Quat::identity();
	Quat q2 = Quat::from_axis_angle(Vec3(0, 0, 1), 0.174533f); 
	q2.reg = simd::mul(q2.reg, simd::set1(-1.0f)); 

	Quat res = Quat::slerp(q1, q2, 0.5f);

	float d = simd::x(simd::dot4_splat(res.reg, q1.reg));
	float expected_w = std::cos(0.174533f * 0.5f * 0.5f);

	EXPECT_GT(d, 0.0f); 
	EXPECT_NEAR(std::abs(d), expected_w, 1e-4f);
}

TEST(QuatTest, FastSlerpPrecision) {
	Quat q1 = Quat::from_axis_angle(Vec3(1, 1, 0), 0.2f);
	Quat q2 = Quat::from_axis_angle(Vec3(1, 1, 0), 1.2f);
	float t = 0.35f;

	Quat res_slow = Quat::slerp(q1, q2, t);
	Quat res_fast = Quat::slerp_fast(q1, q2, t);

	EXPECT_NEAR(res_slow.get_x(), res_fast.get_x(), 1e-4f);
	EXPECT_NEAR(res_slow.get_y(), res_fast.get_y(), 1e-4f);
	EXPECT_NEAR(res_slow.get_z(), res_fast.get_z(), 1e-4f);
	EXPECT_NEAR(res_slow.get_w(), res_fast.get_w(), 1e-4f);
}

TEST(QuatTest, FastSlerpLargeAnglePrecision) {
	float angle = 2.79253f;
	Quat q1 = Quat::identity();
	Quat q2 = Quat::from_axis_angle(Vec3(0, 1, 0), angle);

	float times[] = {0.25f, 0.5f, 0.75f};

	for(float t : times) {
		Quat res_slow = Quat::slerp(q1, q2, t);
		Quat res_fast = Quat::slerp_fast(q1, q2, t);

		EXPECT_NEAR(res_slow.get_x(), res_fast.get_x(), 1e-3f);
		EXPECT_NEAR(res_slow.get_y(), res_fast.get_y(), 1e-3f);
		EXPECT_NEAR(res_slow.get_z(), res_fast.get_z(), 1e-3f);
		EXPECT_NEAR(res_slow.get_w(), res_fast.get_w(), 1e-3f);
	}
}

TEST(QuatTest, FastSlerpWorstCasePrecision) {
	float angle = 3.124139f; // 179 deg
	Quat q1 = Quat::identity();
	Quat q2 = Quat::from_axis_angle(Vec3(0, 1, 0), angle);

	float times[] = {0.25f, 0.5f, 0.75f};

	for(float t : times) {
		Quat res_slow = Quat::slerp(q1, q2, t);
		Quat res_fast = Quat::slerp_fast(q1, q2, t);

		EXPECT_NEAR(res_slow.get_x(), res_fast.get_x(), 1e-3f);
		EXPECT_NEAR(res_slow.get_y(), res_fast.get_y(), 1e-3f);
		EXPECT_NEAR(res_slow.get_z(), res_fast.get_z(), 1e-3f);
		EXPECT_NEAR(res_slow.get_w(), res_fast.get_w(), 1e-3f);
	}
}

TEST(QuatTest, SlerpIdenticalQuaternions) {
	Quat q1 = Quat::from_axis_angle(Vec3(1, 0, 0), 0.1f);
	Quat q2 = q1;

	Quat res = Quat::slerp(q1, q2, 0.5f);

	EXPECT_FALSE(std::isnan(res.get_x()));
	EXPECT_NEAR(res.get_w(), q1.get_w(), 1e-6f);
	EXPECT_NEAR(res.get_x(), q1.get_x(), 1e-6f);
}

TEST(QuatTest, SlerpNearlyIdentical) {
	Quat q1 = Quat::identity();
	Quat q2 = Quat::from_axis_angle(Vec3(0, 1, 0), 0.0001f);

	Quat res = Quat::slerp(q1, q2, 0.5f);

	EXPECT_NEAR(res.l2(), 1.0f, 1e-6f);
	EXPECT_GT(res.get_w(), 0.999f);
}

TEST(QuatTest, ConsistencyWithMatrix) {
	Quat q = Quat::from_axis_angle(Vec3(1, 2, 3).normalized(), 1.2f);
	Mat4 m = q.to_mat4();
	Vec3 v(1, 0, 0);

	Vec3 res_q = q.rotate(v);
 
	Vec4 res_m = m * Vec4(v, 1.0f);

	EXPECT_NEAR(res_q.get_x(), res_m.get_x(), 1e-5f);
	EXPECT_NEAR(res_q.get_y(), res_m.get_y(), 1e-5f);
	EXPECT_NEAR(res_q.get_z(), res_m.get_z(), 1e-5f);
}

TEST(QuatTest, ConjugateProductLaw) {
	Quat q1 = Quat::from_axis_angle(Vec3(1, 0, 0), 0.5f);
	Quat q2 = Quat::from_axis_angle(Vec3(0, 1, 0), 0.8f);

	Quat left = (q1 * q2).conjugated();
	Quat right = q2.conjugated() * q1.conjugated();

	EXPECT_NEAR(left.get_x(), right.get_x(), 1e-6f);
	EXPECT_NEAR(left.get_y(), right.get_y(), 1e-6f);
	EXPECT_NEAR(left.get_z(), right.get_z(), 1e-6f);
	EXPECT_NEAR(left.get_w(), right.get_w(), 1e-6f);
}
	
TEST(QuatTest, SlerpOpposite) {
	Quat q1 = Quat::from_axis_angle(Vec3(1, 0, 0), 0.0f);
	Quat q2 = Quat::from_axis_angle(Vec3(1, 0, 0), 3.141592f);

	Quat res = Quat::slerp(q1, q2, 0.5f);

	EXPECT_FALSE(std::isnan(res.get_w()));
	EXPECT_NEAR(res.l2(), 1.0f, 1e-6f);
}

TEST(QuatTest, NormalizeSmall) {
	Quat q(1e-10f, 1e-10f, 1e-10f, 1e-10f);
	Quat qn = q.normalized();
 
	EXPECT_NEAR(qn.l2(), 1.0f, 1e-5f);
}

TEST(QuatTest, EulerRoundtrip) {
	float x = 0.3f;
	float y = 0.5f;
	float z = -0.2f;

	Quat q = Quat::from_euler(x,y,z);

	Vec4 recovered = q.to_euler();

	Quat q_recovered = Quat::from_euler(
		recovered.get_x(),
		recovered.get_y(),
		recovered.get_z()
	);

	float d = Quat::dot(q, q_recovered);
	EXPECT_NEAR(std::abs(d), 1.0f, 1e-6f);
}

TEST(QuatTest, EulerSpecificAxes) {
    float pi_half = 1.570796f;

    Quat qx = Quat::from_euler(pi_half, 0, 0);
    Vec3 resX = qx.rotate(Vec3(0, 1, 0));
    EXPECT_NEAR(resX.get_z(), 1.0f, 1e-6f);
    EXPECT_NEAR(resX.get_y(), 0.0f, 1e-6f);

    Quat qy = Quat::from_euler(0, pi_half, 0);
    Vec3 resY = qy.rotate(Vec3(1, 0, 0));
    EXPECT_NEAR(resY.get_z(), -1.0f, 1e-6f);
    EXPECT_NEAR(resY.get_x(), 0.0f, 1e-6f);
}

TEST(QuatTest, InverseNonUnit) {
    Quat q(1.0f, 1.0f, 1.0f, 1.0f); 
    Quat q_inv = q.inversed();

    Quat res = q * q_inv;

    EXPECT_NEAR(res.get_x(), 0.0f, 1e-6f);
    EXPECT_NEAR(res.get_y(), 0.0f, 1e-6f);
    EXPECT_NEAR(res.get_z(), 0.0f, 1e-6f);
    EXPECT_NEAR(res.get_w(), 1.0f, 1e-6f);
}
