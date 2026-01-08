#include<cstdlib>

#include<platform/window/window.hpp>
#include<platform/window_sdl/window_sdl.hpp>

#include<gtest/gtest.h>

void set_env_variable(const char* name, const char* value){
	#ifdef _WIN32
		_putenv_s(name,value);
	#else
		setenv(name, value, 1);
	#endif
}

TEST(WindowTest, InitWithDummyDriver){
	set_env_variable("SDL_VIDEODRIVER", "dummy");

	engine::window::WindowDesc desc {"Test", 800, 600};
	engine::window::SDLWindow window;

	EXPECT_TRUE(window.init(desc));
	EXPECT_FALSE(window.should_close());

	EXPECT_EQ(window.width(), 800);
	EXPECT_EQ(window.height(), 600);
}
