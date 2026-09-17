#include<vector>
#include<cmath>
#include<limits>
#include<algorithm>


#include<render/renderer.hpp>
#include<render/software_renderer/renderer_software.hpp>
#include<render/color.hpp>
#include<render/scene.hpp>

#include<core/math/mat4.hpp>
#include<core/math/vec3.hpp>
#include<core/math/vec4.hpp>

namespace engine::render{

static float edge_function(const engine::math::Vec3& a, const engine::math::Vec3& b, float px, float py) {
    return (px - a.x) * (b.y - a.y) - (py - a.y) * (b.x - a.x);
}

void SoftwareRenderer::render_frame(
		engine::render::FrameBuffer& frame_buffer,
		const engine::render::Scene& scene
){
	/*
	uint32_t block_size = 80;

	uint32_t colors[] = {
		pack_argb8888(Color::red()),
		pack_argb8888(Color::blue()),
		pack_argb8888(Color::green()),
		pack_argb8888(Color::white()),
		pack_argb8888(Color::black()),
	};

	for(uint32_t i = 0; i < 16; ++i){
		for(uint32_t j = 0; j < 9; ++j){
			uint32_t pixel_color = colors[(i+j)%5];

			for(uint32_t k = 0; k < block_size; ++k){
				for(uint32_t l = 0; l < block_size; ++l){
					frame_buffer.pixel(i*block_size+l,j*block_size+k) = pixel_color;
				}
			}
		}
	}
	*/

	const uint32_t w = frame_buffer.width();
	const uint32_t h = frame_buffer.height();
	if(w==0 || h==0) return;

	std::fill_n(frame_buffer.data(), frame_buffer.size(), scene.clear_color);


	std::vector<float> depth_buffer(frame_buffer.size(),1.0f);


	engine::math::Mat4 vp = scene.camera.projection * scene.camera.view;


	for(const RenderObject& robj : scene.objects){
		engine::math::Mat4 mvp = vp * robj.model;

		// get vertices from RenderObject.Mesh ..
		std::vector<engine::math::Vec3> vertices = {
			{0.0f, 0.5f, 0.0f},
			{-0.5f, -0.5f, 0.0f},
			{0.5f, -0.5f, 0.0f}
		};

		for(std::size_t i = 0; i + 2 < vertices.size(); i+=3){
			engine::math::Vec4 clip0 = mvp * engine::math::Vec4(vertices[i], 1.0f);
			engine::math::Vec4 clip1 = mvp * engine::math::Vec4(vertices[i+1], 1.0f);
			engine::math::Vec4 clip2 = mvp * engine::math::Vec4(vertices[i+2], 1.0f);

			engine::math::Vec3 ndc0 = { clip0.x / clip0.w, clip0.y / clip0.w, clip0.z / clip0.w };
            engine::math::Vec3 ndc1 = { clip1.x / clip1.w, clip1.y / clip1.w, clip1.z / clip1.w };
            engine::math::Vec3 ndc2 = { clip2.x / clip2.w, clip2.y / clip2.w, clip2.z / clip2.w };
			// C. Viewport Transform -> Współrzędne Ekranowe (Piksele)
            auto to_screen = [w, h](const engine::math::Vec3& ndc) -> engine::math::Vec3 {
                return {
                    (ndc.x + 1.0f) * 0.5f * static_cast<float>(w),
                    (1.0f - ndc.y) * 0.5f * static_cast<float>(h), // Odwrócenie osi Y dla ekranu
                    ndc.z
                };
            };

            engine::math::Vec3 v0 = to_screen(ndc0);
            engine::math::Vec3 v1 = to_screen(ndc1);
            engine::math::Vec3 v2 = to_screen(ndc2);

            // D. Bounding Box trójkąta na ekranie
            int min_x = std::clamp(static_cast<int>(std::floor(std::min({v0.x, v1.x, v2.x}))), 0, static_cast<int>(w) - 1);
            int max_x = std::clamp(static_cast<int>(std::ceil(std::max({v0.x, v1.x, v2.x}))), 0, static_cast<int>(w) - 1);
            int min_y = std::clamp(static_cast<int>(std::floor(std::min({v0.y, v1.y, v2.y}))), 0, static_cast<int>(h) - 1);
            int max_y = std::clamp(static_cast<int>(std::ceil(std::max({v0.y, v1.y, v2.y}))), 0, static_cast<int>(h) - 1);

            float area = edge_function(v0, v1, v2.x, v2.y);
            if (area == 0.0f) continue; // Odrzuć zdegenerowane trójkąty

            // E. RASTERYZACJA (Pętla po Bounding Boxie)
            for (int y = min_y; y <= max_y; ++y) {
                for (int x = min_x; x <= max_x; ++x) {
                    float px = static_cast<float>(x) + 0.5f;
                    float py = static_cast<float>(y) + 0.5f;

                    // Obliczenie wag barycentrycznych
                    float w0 = edge_function(v1, v2, px, py);
                    float w1 = edge_function(v2, v0, px, py);
                    float w2 = edge_function(v0, v1, px, py);

                    // Jeśli punkt leży wewnątrz trójkąta
                    if (w0 >= 0.0f && w1 >= 0.0f && w2 >= 0.0f) {
                        w0 /= area;
                        w1 /= area;
                        w2 /= area;

                        // Interpolacja głębokości (Z)
                        float depth = w0 * v0.z + w1 * v1.z + w2 * v2.z;
                        size_t pixel_idx = static_cast<size_t>(y) * w + static_cast<size_t>(x);

                        // Z-Test
                        if (depth < depth_buffer[pixel_idx]) {
                            depth_buffer[pixel_idx] = depth;
                            frame_buffer.pixel(static_cast<uint32_t>(x), static_cast<uint32_t>(y)) = 0xFF00FF00; // Zielony kolor (ARGB)
                        }
                    }
                }
            }
		}

	}
}

} // namespace render
