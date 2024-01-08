workspace "COMP5822M-exercise4"
	language "C++"
	cppdialect "C++17"

	platforms { "x64" }
	configurations { "debug", "release" }

	flags "NoPCH"
	flags "MultiProcessorCompile"

	startproject "exercise4"

	debugdir "%{wks.location}"
	objdir "_build_/%{cfg.buildcfg}-%{cfg.platform}-%{cfg.toolset}"
	targetsuffix "-%{cfg.buildcfg}-%{cfg.platform}-%{cfg.toolset}"
	
	-- Default toolset options
	filter "toolset:gcc or toolset:clang"
		linkoptions { "-pthread" }
		buildoptions { "-march=native", "-Wall", "-pthread" }

	filter "toolset:msc-*"
		defines { "_CRT_SECURE_NO_WARNINGS=1" }
		defines { "_SCL_SECURE_NO_WARNINGS=1" }
		buildoptions { "/utf-8" }
	
	filter "*"

	-- default libraries
	filter "system:linux"
		links "dl"
	
	filter "system:windows"

	filter "*"

	-- default outputs
	filter "kind:StaticLib"
		targetdir "lib/"

	filter "kind:ConsoleApp"
		targetdir "bin/"
		targetextension ".exe"
	
	filter "*"

	--configurations
	filter "debug"
		symbols "On"
		defines { "_DEBUG=1" }

	filter "release"
		optimize "On"
		defines { "NDEBUG=1" }

	filter "*"

-- Third party dependencies
include "third_party" 

-- GLSLC helpers
dofile( "util/glslc.lua" )

-- Projects
project "exercise1"
	local sources = { 
		"exercise1/**.cpp",
		"exercise1/**.hpp",
		"exercise1/**.hxx"
	}

	kind "ConsoleApp"
	location "exercise1"

	files( sources )

	links "labutils"
	links "x-volk"

project "exercise2"
	local sources = { 
		"exercise2/**.cpp",
		"exercise2/**.hpp",
		"exercise2/**.hxx"
	}

	kind "ConsoleApp"
	location "exercise2"

	files( sources )

	dependson "exercise2-shaders"

	links "labutils"
	links "x-volk"
	links "x-stb"

project "exercise2-shaders"
	local shaders = { 
		"exercise2/shaders/*.vert",
		"exercise2/shaders/*.frag"
	}

	kind "Utility"
	location "exercise2/shaders"

	files( shaders )

	handle_glsl_files( "-O", "assets/exercise2/shaders", {} )


project "exercise3"
	local sources = { 
		"exercise3/**.cpp",
		"exercise3/**.hpp",
		"exercise3/**.hxx"
	}

	kind "ConsoleApp"
	location "exercise3"

	files( sources )

	dependson "exercise3-shaders"

	links "labutils"
	links "x-volk"
	links "x-stb"
	links "x-glfw"

project "exercise3-shaders"
	local shaders = { 
		"exercise3/shaders/*.vert",
		"exercise3/shaders/*.frag"
	}

	kind "Utility"
	location "exercise3/shaders"

	files( shaders )

	handle_glsl_files( "-O", "assets/exercise3/shaders", {} )

project "exercise4"
	local sources = { 
		"exercise4/**.cpp",
		"exercise4/**.hpp",
		"exercise4/**.hxx"
	}

	kind "ConsoleApp"
	location "exercise4"

	files( sources )

	dependson "exercise4-shaders"

	links "labutils"
	links "x-volk"
	links "x-stb"
	links "x-glfw"
	links "x-vma"

	dependson "x-glm" 

project "exercise4-shaders"
	local shaders = { 
		"exercise4/shaders/*.vert",
		"exercise4/shaders/*.frag"
	}

	kind "Utility"
	location "exercise4/shaders"

	files( shaders )

	handle_glsl_files( "-O", "assets/exercise4/shaders", {} )

project "labutils"
	local sources = { 
		"labutils/**.cpp",
		"labutils/**.hpp",
		"labutils/**.hxx"
	}

	kind "StaticLib"
	location "labutils"

	files( sources )

--EOF
