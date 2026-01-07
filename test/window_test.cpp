#include<platform/window/window.hpp>
#include<platform/window_sdl/window_sdl.hpp>

#include<gtest/gtest.h>

TEST(WindowTest, InitWithDummyDriver){
	setenv("SDL_VIDEODRIVER", "dummy", 1);

	engine::window::WindowDesc desc {"Test", 800, 600};
	engine::window::SDLWindow window;

	EXPECT_TRUE(window.init(desc));
	EXPECT_FALSE(window.should_close());

	EXPECT_EQ(window.width(), 800);
	EXPECT_EQ(window.height(), 600);
}
