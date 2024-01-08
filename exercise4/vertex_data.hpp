#pragma once

#include <cstdint>

#include "../labutils/vulkan_context.hpp"

#include "../labutils/vkbuffer.hpp"
#include "../labutils/allocator.hpp" 

struct ColorizedMesh
{
	labutils::Buffer positions;
	labutils::Buffer colors;

	std::uint32_t vertexCount;
};

struct TexturedMesh 
{ 
	labutils::Buffer positions; 
	labutils::Buffer texcoords; 

	std::uint32_t vertexCount; 
};


ColorizedMesh create_triangle_mesh( labutils::VulkanContext const&, labutils::Allocator const& );
TexturedMesh create_plane_mesh(labutils::VulkanContext const&, labutils::Allocator const&);
TexturedMesh create_sprite_mesh(labutils::VulkanContext const&, labutils::Allocator const&);



