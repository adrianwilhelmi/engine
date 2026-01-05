#include<iostream>
#include<fstream>
#include<random>
#include<vector>

void generate_bench_data(std::size_t count){
	std::mt19937 gen(42);
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
	std::uniform_real_distribution<float> t_dist(0.0f, 1.0f);

	std::ofstream file_a("quat_a.txt");
	std::ofstream file_b("quat_b.txt");
	std::ofstream file_t("ts.txt");

	for(std::size_t i = 0; i < count; ++i){
		auto gen_quat = [&](){
			float x, y, z, w, mag2;
			do {
				x = dist(gen);
				y = dist(gen);
				z = dist(gen);
				w = dist(gen);
				mag2 = x*x + y*y + z*z + w*w;
			} while (mag2 > 1.0f || mag2 < 0.00f);

			float inv_mag = 1.0f / std::sqrt(mag2);
			std::vector<float> quat = {
				x*inv_mag,
				y*inv_mag,
				z*inv_mag,
				w*inv_mag
			};
			return quat;
		};

		std::vector<float> q1 = gen_quat();
		std::vector<float> q2 = gen_quat();
		float t = t_dist(gen);

		file_a << q1[0] << "\t" << q1[1] << "\t" 
			<< q1[2] << "\t" << q1[3] << "\n";
		file_b << q2[0] << "\t" << q2[1] << "\t" 
			<< q2[2] << "\t" << q2[3] << "\n";
		file_t << t << "\n";
	}
}

int main(){
	std::size_t n = 1000000;
	generate_bench_data(n);
}
