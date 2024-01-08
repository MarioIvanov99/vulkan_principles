#include "vertex_data.hpp"

#include <limits>

#include <cstring> // for std::memcpy()

#include "../labutils/error.hpp"
#include "../labutils/vkutil.hpp"
#include "../labutils/to_string.hpp"
namespace lut = labutils;

ColorizedMesh create_triangle_mesh( labutils::VulkanContext const& aContext, labutils::Allocator const& aAllocator )
{
	// Vertex data
	static float const positions[] = {
		0.0f, -0.8f,
		-0.7f, 0.8f,
		+0.7f, 0.8f
	};
	static float const colors[] = {
		0.f, 0.f, 1.f,
		1.f, 0.f, 0.f,
		0.f, 1.f, 0.f
	};

	lut::Buffer vertexPosGPU = lut::create_buffer(
		aAllocator, 
		sizeof(positions), 
		VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 
		VMA_MEMORY_USAGE_GPU_ONLY 
	); 

	lut::Buffer vertexColGPU = lut::create_buffer(
		aAllocator, 
		sizeof(colors), 
		VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		VMA_MEMORY_USAGE_GPU_ONLY 
	);

	lut::Buffer posStaging = lut::create_buffer(
		aAllocator, 
		sizeof(positions), 
		VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 
		VMA_MEMORY_USAGE_CPU_TO_GPU 
	); 

	lut::Buffer colStaging = lut::create_buffer(
		aAllocator, 
		sizeof(colors), 
		VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 
		VMA_MEMORY_USAGE_CPU_TO_GPU 
	);

	void* posPtr = nullptr; 
	if(auto const res = vmaMapMemory(aAllocator.allocator, posStaging.allocation, &posPtr); VK_SUCCESS != res)
	{
		throw lut::Error("Mapping memory for writing\n" "vmaMapMemory() returned %s", lut::to_string(res).c_str());
			
	} 

	std::memcpy(posPtr, positions, sizeof(positions)); 
	vmaUnmapMemory(aAllocator.allocator, posStaging.allocation);
			
	void * colPtr = nullptr; 
	if(auto const res = vmaMapMemory(aAllocator.allocator, colStaging.allocation, &colPtr); VK_SUCCESS != res)
	{
		throw lut::Error("Mapping memory for writing\n" "vmaMapMemory() returned %s", lut::to_string(res).c_str()); 
				
	} 

	std::memcpy(colPtr, colors, sizeof(colors));
	vmaUnmapMemory(aAllocator.allocator, colStaging.allocation);

	lut::Fence uploadComplete = create_fence(aContext);

	lut::CommandPool uploadPool = create_command_pool(aContext);
	VkCommandBuffer uploadCmd = alloc_command_buffer(aContext, uploadPool.handle);

	VkCommandBufferBeginInfo beginInfo{}; 
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO; 
	beginInfo.flags = 0; 
	beginInfo.pInheritanceInfo = nullptr; 
		
	if(auto const res = vkBeginCommandBuffer(uploadCmd, &beginInfo); VK_SUCCESS != res)
	{
		throw lut::Error("Beginning command buffer recording\n" "vkBeginCommandBuffer() returned %s", lut::to_string(res).c_str());
			
	} 
		
	VkBufferCopy pcopy{}; 
	pcopy.size = sizeof(positions); 
			
	vkCmdCopyBuffer(uploadCmd, posStaging.buffer, vertexPosGPU.buffer, 1, &pcopy); 
			
	lut::buffer_barrier(uploadCmd, 
		vertexPosGPU.buffer, 
		VK_ACCESS_TRANSFER_WRITE_BIT, 
		VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT, 
		VK_PIPELINE_STAGE_TRANSFER_BIT, 
		VK_PIPELINE_STAGE_VERTEX_INPUT_BIT 
	); 
			
	VkBufferCopy ccopy{}; 
	ccopy.size = sizeof(colors); 
			
	vkCmdCopyBuffer(uploadCmd, colStaging.buffer, vertexColGPU.buffer, 1, &ccopy); 
			
	lut::buffer_barrier(uploadCmd, 
		vertexColGPU.buffer, 
		VK_ACCESS_TRANSFER_WRITE_BIT, 
		VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT, 
		VK_PIPELINE_STAGE_TRANSFER_BIT, 
		VK_PIPELINE_STAGE_VERTEX_INPUT_BIT 
	); 
			
	if(auto const res = vkEndCommandBuffer(uploadCmd); VK_SUCCESS != res) 
	{ 
		throw lut::Error("Ending command buffer recording\n" "vkEndCommandBuffer() returned %s", lut::to_string(res).c_str()); 
	}

	// Submit transfer commands 
	VkSubmitInfo submitInfo{}; 
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO; 
	submitInfo.commandBufferCount = 1; 
	submitInfo.pCommandBuffers = &uploadCmd; 
	
	if(auto const res = vkQueueSubmit(aContext.graphicsQueue, 1, &submitInfo, uploadComplete.handle); VK_SUCCESS != res)
	{
		throw lut::Error("Submitting commands\n" "vkQueueSubmit() returned %s", lut::to_string(res).c_str()); 
			
	} 


	if(auto const res = vkWaitForFences(aContext.device, 1, &uploadComplete.handle,  VK_TRUE, std::numeric_limits<std::uint64_t>::max()); VK_SUCCESS != res)
	{
			throw lut::Error("Waiting for upload to complete\n" "vkWaitForFences() returned %s", lut::to_string(res).c_str() ); 
				
	}

	return ColorizedMesh{ 
		std::move(vertexPosGPU), 
		std::move(vertexColGPU), 
		sizeof(positions) / sizeof(float) / 3 // two floats per position 
	};
}

TexturedMesh create_plane_mesh(labutils::VulkanContext const& aContext, labutils::Allocator const& aAllocator)
{

	// Vertex data 1
	static float const positions[] = { 
		- 1.f, 0.f, -6.f, // v0 
		- 1.f, 0.f, +6.f, // v1 
		+ 1.f, 0.f, +6.f, // v2 
	
		- 1.f, 0.f, -6.f, // v0 
		+ 1.f, 0.f, +6.f, // v2 
		+ 1.f, 0.f, -6.f // v3 
	}; 

	static float const texcoord[] = {
		0.f, -6.f, // t0 
		0.f, +6.f, // t1 
		1.f, +6.f, // t2 

		0.f, -6.f, // t0 
		1.f, +6.f, // t2 
		1.f, -6.f // t3 
	};

	lut::Buffer vertexPosGPU = lut::create_buffer(
		aAllocator,
		sizeof(positions),
		VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		VMA_MEMORY_USAGE_GPU_ONLY
	);

	lut::Buffer vertexColGPU = lut::create_buffer(
		aAllocator,
		sizeof(texcoord),
		VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		VMA_MEMORY_USAGE_GPU_ONLY
	);

	lut::Buffer posStaging = lut::create_buffer(
		aAllocator,
		sizeof(positions),
		VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VMA_MEMORY_USAGE_CPU_TO_GPU
	);

	lut::Buffer colStaging = lut::create_buffer(
		aAllocator,
		sizeof(texcoord),
		VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VMA_MEMORY_USAGE_CPU_TO_GPU
	);

	void* posPtr = nullptr;
	if (auto const res = vmaMapMemory(aAllocator.allocator, posStaging.allocation, &posPtr); VK_SUCCESS != res)
	{
		throw lut::Error("Mapping memory for writing\n" "vmaMapMemory() returned %s", lut::to_string(res).c_str());

	}

	std::memcpy(posPtr, positions, sizeof(positions));
	vmaUnmapMemory(aAllocator.allocator, posStaging.allocation);

	void* colPtr = nullptr;
	if (auto const res = vmaMapMemory(aAllocator.allocator, colStaging.allocation, &colPtr); VK_SUCCESS != res)
	{
		throw lut::Error("Mapping memory for writing\n" "vmaMapMemory() returned %s", lut::to_string(res).c_str());

	}

	std::memcpy(colPtr, texcoord, sizeof(texcoord));
	vmaUnmapMemory(aAllocator.allocator, colStaging.allocation);

	lut::Fence uploadComplete = create_fence(aContext);

	lut::CommandPool uploadPool = create_command_pool(aContext);
	VkCommandBuffer uploadCmd = alloc_command_buffer(aContext, uploadPool.handle);

	VkCommandBufferBeginInfo beginInfo{};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags = 0;
	beginInfo.pInheritanceInfo = nullptr;

	if (auto const res = vkBeginCommandBuffer(uploadCmd, &beginInfo); VK_SUCCESS != res)
	{
		throw lut::Error("Beginning command buffer recording\n" "vkBeginCommandBuffer() returned %s", lut::to_string(res).c_str());

	}

	VkBufferCopy pcopy{};
	pcopy.size = sizeof(positions);

	vkCmdCopyBuffer(uploadCmd, posStaging.buffer, vertexPosGPU.buffer, 1, &pcopy);

	lut::buffer_barrier(uploadCmd,
		vertexPosGPU.buffer,
		VK_ACCESS_TRANSFER_WRITE_BIT,
		VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
		VK_PIPELINE_STAGE_TRANSFER_BIT,
		VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
	);

	VkBufferCopy ccopy{};
	ccopy.size = sizeof(texcoord);

	vkCmdCopyBuffer(uploadCmd, colStaging.buffer, vertexColGPU.buffer, 1, &ccopy);

	lut::buffer_barrier(uploadCmd,
		vertexColGPU.buffer,
		VK_ACCESS_TRANSFER_WRITE_BIT,
		VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
		VK_PIPELINE_STAGE_TRANSFER_BIT,
		VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
	);

	if (auto const res = vkEndCommandBuffer(uploadCmd); VK_SUCCESS != res)
	{
		throw lut::Error("Ending command buffer recording\n" "vkEndCommandBuffer() returned %s", lut::to_string(res).c_str());
	}

	// Submit transfer commands 
	VkSubmitInfo submitInfo{};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &uploadCmd;

	if (auto const res = vkQueueSubmit(aContext.graphicsQueue, 1, &submitInfo, uploadComplete.handle); VK_SUCCESS != res)
	{
		throw lut::Error("Submitting commands\n" "vkQueueSubmit() returned %s", lut::to_string(res).c_str());

	}


	if (auto const res = vkWaitForFences(aContext.device, 1, &uploadComplete.handle, VK_TRUE, std::numeric_limits<std::uint64_t>::max()); VK_SUCCESS != res)
	{
		throw lut::Error("Waiting for upload to complete\n" "vkWaitForFences() returned %s", lut::to_string(res).c_str());

	}

	return TexturedMesh{ 
		std::move(vertexPosGPU), 
		std::move(vertexColGPU), 
		sizeof(positions) / sizeof(float) / 3 // now three floats per position 
	};

}

TexturedMesh create_sprite_mesh(labutils::VulkanContext const& aContext, labutils::Allocator const& aAllocator)
{

	// Vertex data 1
	static float const positions[] = {
		- 1.5f, +1.5f, -4.f, // v0 
		- 1.5f, -0.5f, -4.f, // v1 
		+ 1.5f, -0.5f, -4.f, // v2 

		- 1.5f, +1.5f, -4.f, // v0 
		+ 1.5f, -0.5f, -4.f, // v2 
		+ 1.5f, +1.5f, -4.f // v3 
	}; 

	static float const texcoord[] = { 
		0.f, 1.f, // t0 
		0.f, 0.f, // t1 
		1.f, 0.f, // t2 

		0.f, 1.f, // t0 
		1.f, 0.f, // t2 
		1.f, 1.f // t3 
	};

	lut::Buffer vertexPosGPU = lut::create_buffer(
		aAllocator,
		sizeof(positions),
		VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		VMA_MEMORY_USAGE_GPU_ONLY
	);

	lut::Buffer vertexColGPU = lut::create_buffer(
		aAllocator,
		sizeof(texcoord),
		VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		VMA_MEMORY_USAGE_GPU_ONLY
	);

	lut::Buffer posStaging = lut::create_buffer(
		aAllocator,
		sizeof(positions),
		VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VMA_MEMORY_USAGE_CPU_TO_GPU
	);

	lut::Buffer colStaging = lut::create_buffer(
		aAllocator,
		sizeof(texcoord),
		VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VMA_MEMORY_USAGE_CPU_TO_GPU
	);

	void* posPtr = nullptr;
	if (auto const res = vmaMapMemory(aAllocator.allocator, posStaging.allocation, &posPtr); VK_SUCCESS != res)
	{
		throw lut::Error("Mapping memory for writing\n" "vmaMapMemory() returned %s", lut::to_string(res).c_str());

	}

	std::memcpy(posPtr, positions, sizeof(positions));
	vmaUnmapMemory(aAllocator.allocator, posStaging.allocation);

	void* colPtr = nullptr;
	if (auto const res = vmaMapMemory(aAllocator.allocator, colStaging.allocation, &colPtr); VK_SUCCESS != res)
	{
		throw lut::Error("Mapping memory for writing\n" "vmaMapMemory() returned %s", lut::to_string(res).c_str());

	}

	std::memcpy(colPtr, texcoord, sizeof(texcoord));
	vmaUnmapMemory(aAllocator.allocator, colStaging.allocation);

	lut::Fence uploadComplete = create_fence(aContext);

	lut::CommandPool uploadPool = create_command_pool(aContext);
	VkCommandBuffer uploadCmd = alloc_command_buffer(aContext, uploadPool.handle);

	VkCommandBufferBeginInfo beginInfo{};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags = 0;
	beginInfo.pInheritanceInfo = nullptr;

	if (auto const res = vkBeginCommandBuffer(uploadCmd, &beginInfo); VK_SUCCESS != res)
	{
		throw lut::Error("Beginning command buffer recording\n" "vkBeginCommandBuffer() returned %s", lut::to_string(res).c_str());

	}

	VkBufferCopy pcopy{};
	pcopy.size = sizeof(positions);

	vkCmdCopyBuffer(uploadCmd, posStaging.buffer, vertexPosGPU.buffer, 1, &pcopy);

	lut::buffer_barrier(uploadCmd,
		vertexPosGPU.buffer,
		VK_ACCESS_TRANSFER_WRITE_BIT,
		VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
		VK_PIPELINE_STAGE_TRANSFER_BIT,
		VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
	);

	VkBufferCopy ccopy{};
	ccopy.size = sizeof(texcoord);

	vkCmdCopyBuffer(uploadCmd, colStaging.buffer, vertexColGPU.buffer, 1, &ccopy);

	lut::buffer_barrier(uploadCmd,
		vertexColGPU.buffer,
		VK_ACCESS_TRANSFER_WRITE_BIT,
		VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
		VK_PIPELINE_STAGE_TRANSFER_BIT,
		VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
	);

	if (auto const res = vkEndCommandBuffer(uploadCmd); VK_SUCCESS != res)
	{
		throw lut::Error("Ending command buffer recording\n" "vkEndCommandBuffer() returned %s", lut::to_string(res).c_str());
	}

	// Submit transfer commands 
	VkSubmitInfo submitInfo{};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &uploadCmd;

	if (auto const res = vkQueueSubmit(aContext.graphicsQueue, 1, &submitInfo, uploadComplete.handle); VK_SUCCESS != res)
	{
		throw lut::Error("Submitting commands\n" "vkQueueSubmit() returned %s", lut::to_string(res).c_str());

	}


	if (auto const res = vkWaitForFences(aContext.device, 1, &uploadComplete.handle, VK_TRUE, std::numeric_limits<std::uint64_t>::max()); VK_SUCCESS != res)
	{
		throw lut::Error("Waiting for upload to complete\n" "vkWaitForFences() returned %s", lut::to_string(res).c_str());

	}

	return TexturedMesh{
		std::move(vertexPosGPU),
		std::move(vertexColGPU),
		sizeof(positions) / sizeof(float) / 3 // now three floats per position 
	};

}

