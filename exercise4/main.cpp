#include <volk/volk.h>

#include <tuple>
#include <limits>
#include <vector>
#include <stdexcept>

#include <cstdio>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include <stb_image_write.h>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#if !defined(GLM_FORCE_RADIANS)
#	define GLM_FORCE_RADIANS
#endif
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "../labutils/to_string.hpp"
#include "../labutils/vulkan_window.hpp"

#include "../labutils/angle.hpp"
using namespace labutils::literals;

#include "../labutils/error.hpp"
#include "../labutils/vkutil.hpp"
#include "../labutils/vkimage.hpp"
#include "../labutils/vkobject.hpp"
#include "../labutils/vkbuffer.hpp"
#include "../labutils/allocator.hpp" 
namespace lut = labutils;

#include "vertex_data.hpp"

namespace
{
	namespace cfg
	{
		// Compiled shader code for the graphics pipeline
		// See sources in exercise4/shaders/*. 
#		define SHADERDIR_ "assets/exercise4/shaders/"
		constexpr char const* kVertShaderPath = SHADERDIR_ "shaderTex.vert.spv";
		constexpr char const* kFragShaderPath = SHADERDIR_ "shaderTex.frag.spv";
		constexpr char const* kAlphaVertShaderPath = SHADERDIR_ "shaderTex.vert.spv";
		constexpr char const* kAlphaFragShaderPath = SHADERDIR_ "shaderTexAlpha.frag.spv";
#		undef SHADERDIR_




		// General rule: with a standard 24 bit or 32 bit float depth buffer,
		// you can support a 1:1000 ratio between the near and far plane with
		// minimal depth fighting. Larger ratios will introduce more depth
		// fighting problems; smaller ratios will increase the depth buffer's
		// resolution but will also limit the view distance.
		constexpr float kCameraNear  = 0.1f;
		constexpr float kCameraFar   = 100.f;

		constexpr auto kCameraFov    = 60.0_degf;

		#define ASSERTDIR_ "assets/exercise4/" 
		constexpr char const* kFloorTextures = ASSERTDIR_ "asphalt-%u.png"; 
		constexpr char const* kSpriteTextures = ASSERTDIR_ "explosion-%u.png";
		#undef ASSERTDIR_

		constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT;
	}

	// GLFW callbacks
	void glfw_callback_key_press( GLFWwindow*, int, int, int, int );

	// Uniform data
	namespace glsl
	{
		struct SceneUniform
		{
			// Note: need to be careful about the packing/alignment here! 3
			glm::mat4 camera; 
			glm::mat4 projection; 
			glm::mat4 projCam; 
		};

		static_assert(sizeof(SceneUniform) <= 65536, "SceneUniform must be less than 65536 bytes for vkCmdUpdateBuffer" );
		static_assert(sizeof(SceneUniform) % 4 == 0, "SceneUniform size must be a multiple of 4 bytes" );

	}

	// Helpers:
	lut::RenderPass create_render_pass( lut::VulkanWindow const& );

	lut::DescriptorSetLayout create_scene_descriptor_layout( lut::VulkanWindow const& );
	lut::DescriptorSetLayout create_object_descriptor_layout( lut::VulkanWindow const& );

	lut::PipelineLayout create_pipeline_layout( lut::VulkanContext const&, VkDescriptorSetLayout aSceneLayout, VkDescriptorSetLayout aObjectlayout);
	lut::Pipeline create_pipeline( lut::VulkanWindow const&, VkRenderPass, VkPipelineLayout );
	lut::Pipeline create_alpha_pipeline(lut::VulkanWindow const&, VkRenderPass, VkPipelineLayout);

	std::tuple<lut::Image, lut::ImageView> create_depth_buffer( lut::VulkanWindow const&, lut::Allocator const& );

	void create_swapchain_framebuffers( 
		lut::VulkanWindow const&, 
		VkRenderPass,
		std::vector<lut::Framebuffer>&, 
		VkImageView aDepthView
	);

	void update_scene_uniforms(
		glsl::SceneUniform&,
		std::uint32_t aFramebufferWidth,
		std::uint32_t aFramebufferHeight
	);

	void record_commands( 
		VkCommandBuffer,
		VkRenderPass,
		VkFramebuffer,
		VkPipeline,
		VkExtent2D const&,
		VkBuffer aPositionBuffer,
		VkBuffer aTexCoordBuffer,
		std::uint32_t aVertexCount,
		VkBuffer aSceneUBO, 
		glsl::SceneUniform const&, 
		VkPipelineLayout, 
		VkDescriptorSet aSceneDescriptors,
		VkDescriptorSet aObjectDescriptors,
		VkBuffer aSpritePosBuffer, 
		VkBuffer aSpriteTexBuffer,
		std::uint32_t aSpriteVertexCount, 
		VkDescriptorSet aSpriteObjDescriptors,
		VkPipeline aAlphaPipeline
	);
	void submit_commands(
		lut::VulkanWindow const&,
		VkCommandBuffer,
		VkFence,
		VkSemaphore,
		VkSemaphore
	);
	void present_results( 
		VkQueue, 
		VkSwapchainKHR, 
		std::uint32_t aImageIndex, 
		VkSemaphore,
		bool& aNeedToRecreateSwapchain
	);
}


int main() try
{
	// Create Vulkan Window
	auto window = lut::make_vulkan_window();

	// Configure the GLFW window
	glfwSetKeyCallback( window.window, &glfw_callback_key_press );

	// Create VMA allocator
	lut::Allocator allocator = lut::create_allocator( window );

	// Intialize resources
	lut::RenderPass renderPass = create_render_pass( window );

	lut::DescriptorSetLayout sceneLayout = create_scene_descriptor_layout(window);
	lut::DescriptorSetLayout objectLayout = create_object_descriptor_layout(window);


	lut::PipelineLayout pipeLayout = create_pipeline_layout( window, sceneLayout.handle, objectLayout.handle );
	lut::Pipeline pipe = create_pipeline( window, renderPass.handle, pipeLayout.handle );
	lut::Pipeline alphaPipe = create_alpha_pipeline(window, renderPass.handle, pipeLayout.handle);

	auto[depthBuffer, depthBufferView] = create_depth_buffer(window, allocator);

	std::vector<lut::Framebuffer> framebuffers;
	create_swapchain_framebuffers( window, renderPass.handle, framebuffers, depthBufferView.handle);

	lut::CommandPool cpool = lut::create_command_pool( window, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT );

	std::vector<VkCommandBuffer> cbuffers;
	std::vector<lut::Fence> cbfences;
	
	for( std::size_t i = 0; i < framebuffers.size(); ++i )
	{
		cbuffers.emplace_back( lut::alloc_command_buffer( window, cpool.handle ) );
		cbfences.emplace_back( lut::create_fence( window, VK_FENCE_CREATE_SIGNALED_BIT ) );
	}

	lut::Semaphore imageAvailable = lut::create_semaphore( window );
	lut::Semaphore renderFinished = lut::create_semaphore( window );

	// Load data
	//ColorizedMesh triangleMesh = create_triangle_mesh( window, allocator );
	TexturedMesh planeMesh = create_plane_mesh(window, allocator);
	TexturedMesh spriteMesh = create_sprite_mesh(window, allocator);

	lut::Buffer sceneUBO = lut::create_buffer(allocator, sizeof(glsl::SceneUniform), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY );

	lut::DescriptorPool dpool = lut::create_descriptor_pool(window);

	VkDescriptorSet sceneDescriptors = lut::alloc_desc_set(window, dpool.handle, sceneLayout.handle);
	{ 
		VkWriteDescriptorSet desc[1]{}; 
		
		VkDescriptorBufferInfo sceneUboInfo{}; 
		sceneUboInfo.buffer = sceneUBO.buffer; 
		sceneUboInfo.range = VK_WHOLE_SIZE; 
		
		desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; 
		desc[0].dstSet = sceneDescriptors; 
		desc[0].dstBinding = 0; 
		desc[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; 
		desc[0].descriptorCount = 1; 
		desc[0].pBufferInfo = &sceneUboInfo; 
		
		
		constexpr auto numSets = sizeof(desc) / sizeof(desc[0]); 
		vkUpdateDescriptorSets(window.device, numSets, desc, 0, nullptr); 
	}
	//TODO- (Section 3) initialize descriptor set with vkUpdateDescriptorSets

	lut::Image floorTex; 
	
	{ 
		lut::CommandPool loadCmdPool = lut::create_command_pool(window, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT);
		
		floorTex = lut::load_image_texture2d(cfg::kFloorTextures, window, loadCmdPool.handle, allocator);
	}
	lut::ImageView floorView = lut::create_image_view_texture2d(window, floorTex.image, VK_FORMAT_R8G8B8A8_SRGB);

	lut::Sampler defaultSampler = lut::create_default_sampler(window);

	//TODO- (Section 4) allocate and initialize descriptor sets for texture
	VkDescriptorSet floorDescriptors = lut::alloc_desc_set(window, dpool.handle,  objectLayout.handle);
	{ 
		VkWriteDescriptorSet desc[1]{}; 
		
		VkDescriptorImageInfo textureInfo{}; 
		textureInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL; 
		textureInfo.imageView = floorView.handle; 
		textureInfo.sampler = defaultSampler.handle; 
		
		desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; 
		desc[0].dstSet = floorDescriptors; 
		desc[0].dstBinding = 0; 
		desc[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; 
		desc[0].descriptorCount = 1; 
		desc[0].pImageInfo = &textureInfo; 
		
		
		constexpr auto numSets = sizeof(desc) / sizeof(desc[0]); 
		vkUpdateDescriptorSets(window.device, numSets, desc, 0, nullptr); 
	}
	
	lut::Image spriteTex; 
	{ 
		lut::CommandPool loadCmdPool = lut::create_command_pool(window, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT);
		
		spriteTex = lut::load_image_texture2d(cfg::kSpriteTextures, window, loadCmdPool.handle, allocator);
	} 
		
	lut::ImageView spriteView = lut::create_image_view_texture2d(window, spriteTex.image, VK_FORMAT_R8G8B8A8_SRGB);

	VkDescriptorSet spriteDescriptors = lut::alloc_desc_set(window, dpool.handle, objectLayout.handle);
	{ 
		VkWriteDescriptorSet desc[1]{}; 
		
		VkDescriptorImageInfo textureInfo{}; 
		textureInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL; 
		textureInfo.imageView = spriteView.handle; 
		textureInfo.sampler = defaultSampler.handle; 
		
		desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; 
		desc[0].dstSet = spriteDescriptors; 
		desc[0].dstBinding = 0; 
		desc[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; 
		desc[0].descriptorCount = 1; 
		desc[0].pImageInfo = &textureInfo; 
		
		
		constexpr auto numSets = sizeof(desc) / sizeof(desc[0]); 
		vkUpdateDescriptorSets(window.device, numSets, desc, 0, nullptr); 
	}
	// Application main loop
	bool recreateSwapchain = false;

	while( !glfwWindowShouldClose( window.window ) )
	{
		// Let GLFW process events.
		// glfwPollEvents() checks for events, processes them. If there are no
		// events, it will return immediately. Alternatively, glfwWaitEvents()
		// will wait for any event to occur, process it, and only return at
		// that point. The former is useful for applications where you want to
		// render as fast as possible, whereas the latter is useful for
		// input-driven applications, where redrawing is only needed in
		// reaction to user input (or similar).
		glfwPollEvents(); // or: glfwWaitEvents()

		glsl::SceneUniform sceneUniforms{}; 
		update_scene_uniforms(sceneUniforms, window.swapchainExtent.width, window.swapchainExtent.height);

		// Recreate swap chain?
		if( recreateSwapchain )
		{
			// We need to destroy several objects, which may still be in use by 
			// the GPU. Therefore, first wait for the GPU to finish processing. 
			vkDeviceWaitIdle(window.device);

			// Recreate them 
			auto const changes = recreate_swapchain(window);

			if (changes.changedFormat)
				renderPass = create_render_pass(window);

			if (changes.changedSize) 
				std::tie(depthBuffer, depthBufferView) = create_depth_buffer(window, allocator);

			framebuffers.clear();
			create_swapchain_framebuffers(window, renderPass.handle, framebuffers, depthBufferView.handle);

			if (changes.changedSize) {
				pipe = create_pipeline(window, renderPass.handle, pipeLayout.handle);
				alphaPipe = create_alpha_pipeline(window, renderPass.handle, pipeLayout.handle);
			}

			recreateSwapchain = false;
			continue;
			//TODO: (Section 6) re-create depth buffer image
			//recreateSwapchain = false;
		}

		// Acquire next swap chain image 1
		std::uint32_t imageIndex = 0;
		auto const acquireRes = vkAcquireNextImageKHR(window.device, window.swapchain, std::numeric_limits<std::uint64_t>::max(), imageAvailable.handle, VK_NULL_HANDLE, &imageIndex);

		if (VK_SUBOPTIMAL_KHR == acquireRes || VK_ERROR_OUT_OF_DATE_KHR == acquireRes)
		{
			// This occurs e.g., when the window has been resized. In this case 
			// we need to recreate the swap chain to match the new dimensions. 
			// Any resources that directly depend on the swap chain need to be 
			// recreated as well. While rare, re-creating the swap chain may 
			// give us a different image format, which we should handle. 
			// 
			// In both cases, we set the flag that the swap chain has to be 
			// re-created and jump to the top of the loop. Technically, with 
			// the VK SUBOPTIMAL KHR return code, we could continue rendering 
			// with the current swap chain (unlike VK ERROR OUT OF DATE KHR, 
			// which does require us to recreate the swap chain). 
			recreateSwapchain = true;
			continue;
		}

		if (VK_SUCCESS != acquireRes)
		{
			throw lut::Error("Unable to acquire enxt swapchain image\n" "vkAcquireNextImageKHR() returned %s", lut::to_string(acquireRes).c_str());
		}

		// Make sure that the command buffer is no longer in use 
		assert(std::size_t(imageIndex) < cbfences.size());

		if (auto const res = vkWaitForFences(window.device, 1, &cbfences[imageIndex].handle, VK_TRUE, std::numeric_limits<std::uint64_t>::max()); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to wait for command buffer fence %u\n" "vkWaitForFences() returned %s", imageIndex, lut::to_string(res).c_str());
		}

		if (auto const res = vkResetFences(window.device, 1, &cbfences[imageIndex].handle); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to reset command buffer fence %u\n" "vkResetFences() returned %s", imageIndex, lut::to_string(res).c_str());


		}

		// Record and submit commands for this frame
		assert(std::size_t(imageIndex) < cbuffers.size());
		assert(std::size_t(imageIndex) < framebuffers.size());

		record_commands(cbuffers[imageIndex], renderPass.handle, framebuffers[imageIndex].handle, pipe.handle, window.swapchainExtent, planeMesh.positions.buffer, planeMesh.texcoords.buffer, planeMesh.vertexCount, sceneUBO.buffer, sceneUniforms, pipeLayout.handle, sceneDescriptors, floorDescriptors, spriteMesh.positions.buffer, spriteMesh.texcoords.buffer, spriteMesh.vertexCount, spriteDescriptors, alphaPipe.handle);

		submit_commands(window, cbuffers[imageIndex], cbfences[imageIndex].handle, imageAvailable.handle, renderFinished.handle);

		// Present the results
		VkPresentInfoKHR presentInfo{};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores = &renderFinished.handle;
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = &window.swapchain;
		presentInfo.pImageIndices = &imageIndex;
		presentInfo.pResults = nullptr;

		auto const presentRes = vkQueuePresentKHR(window.presentQueue, &presentInfo);

		if (VK_SUBOPTIMAL_KHR == presentRes || VK_ERROR_OUT_OF_DATE_KHR == presentRes)
		{
			recreateSwapchain = true;
		}
		else if (VK_SUCCESS != presentRes)
		{
			throw lut::Error("Unable present swapchain image %u\n" "vkQueuePresentKHR() returned %s", imageIndex, lut::to_string(presentRes).c_str());
		}
	}

	// Cleanup takes place automatically in the destructors, but we sill need
	// to ensure that all Vulkan commands have finished before that.
	vkDeviceWaitIdle( window.device );

	return 0;
}
catch( std::exception const& eErr )
{
	std::fprintf( stderr, "\n" );
	std::fprintf( stderr, "Error: %s\n", eErr.what() );
	return 1;
}

namespace
{
	void glfw_callback_key_press( GLFWwindow* aWindow, int aKey, int /*aScanCode*/, int aAction, int /*aModifierFlags*/ )
	{
		if( GLFW_KEY_ESCAPE == aKey && GLFW_PRESS == aAction )
		{
			glfwSetWindowShouldClose( aWindow, GLFW_TRUE );
		}
	}
}

namespace
{
	void update_scene_uniforms( glsl::SceneUniform& aSceneUniforms, std::uint32_t aFramebufferWidth, std::uint32_t aFramebufferHeight )
	{
		float const aspect = aFramebufferWidth / float(aFramebufferHeight); 
			
		aSceneUniforms.projection = glm::perspectiveRH_ZO(
			lut::Radians(cfg::kCameraFov).value(), 
			aspect, 
			cfg::kCameraNear, 
			cfg::kCameraFar 
		); 

		aSceneUniforms.projection[1][1] *= -1.f; // mirror Y axis 9
		
		aSceneUniforms.camera = glm::translate(glm::vec3(0.f, -0.3f, -1.f)); 
			
		aSceneUniforms.projCam = aSceneUniforms.projection * aSceneUniforms.camera;
	}
}

namespace
{
	lut::RenderPass create_render_pass( lut::VulkanWindow const& aWindow )
	{
		VkAttachmentDescription attachments[2]{};
		attachments[0].format = aWindow.swapchainFormat; //changed! 
		attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR; //changed! 7

		attachments[1].format = cfg::kDepthFormat; 
		attachments[1].samples = VK_SAMPLE_COUNT_1_BIT; 
		attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; 
		attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE; 
		attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED; 
		attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkAttachmentReference subpassAttachments[1]{};
		subpassAttachments[0].attachment = 0; // this refers to attachments[0] 
		subpassAttachments[0].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkAttachmentReference depthAttachment{}; 
		depthAttachment.attachment = 1; // this refers to attachments[1] 
		depthAttachment.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL; 
			
		VkSubpassDescription subpasses[1]{}; 
		subpasses[0].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS; 
		subpasses[0].colorAttachmentCount = 1; 
		subpasses[0].pColorAttachments = subpassAttachments; 
		subpasses[0].pDepthStencilAttachment = &depthAttachment; // New!


		// changed: no explicit subpass dependencies 

		VkRenderPassCreateInfo passInfo{};
		passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		passInfo.attachmentCount = 2;
		passInfo.pAttachments = attachments;
		passInfo.subpassCount = 1;
		passInfo.pSubpasses = subpasses;
		passInfo.dependencyCount = 0; //changed! 
		passInfo.pDependencies = nullptr; //changed! 

		VkRenderPass rpass = VK_NULL_HANDLE;
		if (auto const res = vkCreateRenderPass(aWindow.device, &passInfo, nullptr, &rpass); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create render pass\n" "vkCreateRenderPass() returned %s", lut::to_string(res).c_str());

		}

		return lut::RenderPass(aWindow.device, rpass);
	}

	lut::PipelineLayout create_pipeline_layout( lut::VulkanContext const& aContext, VkDescriptorSetLayout aSceneLayout, VkDescriptorSetLayout aObjectLayout)
	{
		VkDescriptorSetLayout layouts[] = { 
		// Order must match the set = N in the shaders 
		aSceneLayout, 
		aObjectLayout
		};
		
		VkPipelineLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		layoutInfo.setLayoutCount = sizeof(layouts) / sizeof(layouts[0]); // updated! 
		layoutInfo.pSetLayouts = layouts; // updated!
		layoutInfo.pushConstantRangeCount = 0;
		layoutInfo.pPushConstantRanges = nullptr;

		VkPipelineLayout layout = VK_NULL_HANDLE;
		if (auto const res = vkCreatePipelineLayout(aContext.device, &layoutInfo, nullptr, &layout); VK_SUCCESS != res)
		{

			throw lut::Error("Unable to create pipeline layout\n" "vkCreatePipelineLayout() returned %s", lut::to_string(res).c_str());

		}

		return lut::PipelineLayout(aContext.device, layout);
	}


	lut::Pipeline create_pipeline( lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, VkPipelineLayout aPipelineLayout )
	{
		
		lut::ShaderModule vert = lut::load_shader_module(aWindow, cfg::kVertShaderPath);
		lut::ShaderModule frag = lut::load_shader_module(aWindow, cfg::kFragShaderPath);

		VkPipelineDepthStencilStateCreateInfo depthInfo{}; 
		depthInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO; 
		depthInfo.depthTestEnable = VK_TRUE; 
		depthInfo.depthWriteEnable = VK_TRUE; 
		depthInfo.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL; 
		depthInfo.minDepthBounds = 0.f; 
		depthInfo.maxDepthBounds = 1.f;
		
		VkPipelineShaderStageCreateInfo stages[2]{};
		stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		stages[0].module = vert.handle;
		stages[0].pName = "main";

		stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		stages[1].module = frag.handle;
		stages[1].pName = "main";
		
		VkVertexInputBindingDescription vertexInputs[2]{}; 
		vertexInputs[0].binding = 0; 
		vertexInputs[0].stride = sizeof(float) * 3; 
		vertexInputs[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX; 
			
		vertexInputs[1].binding = 1; 
		vertexInputs[1].stride = sizeof(float) * 2; 
		vertexInputs[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		VkVertexInputAttributeDescription vertexAttributes[2]{}; 
		vertexAttributes[0].binding = 0; // must match binding above 
		vertexAttributes[0].location = 0; // must match shader 
		vertexAttributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		vertexAttributes[0].offset = 0; 
			
		vertexAttributes[1].binding = 1; // must match binding above 
		vertexAttributes[1].location = 1; // must match shader 
		vertexAttributes[1].format = VK_FORMAT_R32G32_SFLOAT; 
		vertexAttributes[1].offset = 0;

		VkPipelineVertexInputStateCreateInfo inputInfo{};
		inputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

		inputInfo.vertexBindingDescriptionCount = 2; // number of vertexInputs above 
		inputInfo.pVertexBindingDescriptions = vertexInputs; 
		inputInfo.vertexAttributeDescriptionCount = 2; // number of vertexAttributes above 
		inputInfo.pVertexAttributeDescriptions = vertexAttributes;

		// Define which primitive (point, line, triangle, ...) the input is 
		// assembled into for rasterization. 
		VkPipelineInputAssemblyStateCreateInfo assemblyInfo{};
		assemblyInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		assemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		assemblyInfo.primitiveRestartEnable = VK_FALSE;

		// Define viewport and scissor regions 
		VkViewport viewport{};
		viewport.x = 0.f;
		viewport.y = 0.f;
		viewport.width = float(aWindow.swapchainExtent.width);
		viewport.height = float(aWindow.swapchainExtent.height);
		viewport.minDepth = 0.f;
		viewport.maxDepth = 1.f;

		VkRect2D scissor{};
		scissor.offset = VkOffset2D{ 0, 0 };
		scissor.extent = VkExtent2D{ aWindow.swapchainExtent.width, aWindow.swapchainExtent.height };

		VkPipelineViewportStateCreateInfo viewportInfo{};
		viewportInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportInfo.viewportCount = 1;
		viewportInfo.pViewports = &viewport;
		viewportInfo.scissorCount = 1;
		viewportInfo.pScissors = &scissor;

		// Define rasterization options 
		VkPipelineRasterizationStateCreateInfo rasterInfo{};
		rasterInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterInfo.depthClampEnable = VK_FALSE;
		rasterInfo.rasterizerDiscardEnable = VK_FALSE;
		rasterInfo.polygonMode = VK_POLYGON_MODE_FILL;
		rasterInfo.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterInfo.depthBiasEnable = VK_FALSE;
		rasterInfo.lineWidth = 1.f; // required

		// Define multisampling state 
		VkPipelineMultisampleStateCreateInfo samplingInfo{};
		samplingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		samplingInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		// Define blend state 
		// We define one blend state per color attachment - this example uses a 
		// single color attachment, so we only need one. Right now, we donft do any 
		// blending, so we can ignore most of the members. 
		VkPipelineColorBlendAttachmentState blendStates[1]{};
		blendStates[0].blendEnable = VK_FALSE;
		blendStates[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

		VkPipelineColorBlendStateCreateInfo blendInfo{};
		blendInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		blendInfo.logicOpEnable = VK_FALSE;
		blendInfo.attachmentCount = 1;
		blendInfo.pAttachments = blendStates;

		// Create pipeline 
		VkGraphicsPipelineCreateInfo pipeInfo{};
		pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;

		pipeInfo.stageCount = 2; // vertex + fragment stages 
		pipeInfo.pStages = stages;

		pipeInfo.pVertexInputState = &inputInfo;
		pipeInfo.pInputAssemblyState = &assemblyInfo;
		pipeInfo.pTessellationState = nullptr; // no tessellation 
		pipeInfo.pViewportState = &viewportInfo;
		pipeInfo.pRasterizationState = &rasterInfo;
		pipeInfo.pMultisampleState = &samplingInfo;
		pipeInfo.pDepthStencilState = &depthInfo; // no depth or stencil buffers 
		pipeInfo.pColorBlendState = &blendInfo;
		pipeInfo.pDynamicState = nullptr; // no dynamic states

		pipeInfo.layout = aPipelineLayout;
		pipeInfo.renderPass = aRenderPass;
		pipeInfo.subpass = 0; // first subpass of aRenderPass 

		VkPipeline pipe = VK_NULL_HANDLE;
		if (auto const res = vkCreateGraphicsPipelines(aWindow.device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &pipe); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create graphics pipeline\n" "vkCreateGraphicsPipelines() returned %s", lut::to_string(res).c_str());

		}

		return lut::Pipeline(aWindow.device, pipe);
	}

	lut::Pipeline create_alpha_pipeline(lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, VkPipelineLayout aPipelineLayout)
	{

		lut::ShaderModule vert = lut::load_shader_module(aWindow, cfg::kAlphaVertShaderPath);
		lut::ShaderModule frag = lut::load_shader_module(aWindow, cfg::kAlphaFragShaderPath);

		VkPipelineDepthStencilStateCreateInfo depthInfo{};
		depthInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthInfo.depthTestEnable = VK_TRUE;
		depthInfo.depthWriteEnable = VK_TRUE;
		depthInfo.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
		depthInfo.minDepthBounds = 0.f;
		depthInfo.maxDepthBounds = 1.f;

		VkPipelineShaderStageCreateInfo stages[2]{};
		stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		stages[0].module = vert.handle;
		stages[0].pName = "main";

		stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		stages[1].module = frag.handle;
		stages[1].pName = "main";

		VkVertexInputBindingDescription vertexInputs[2]{};
		vertexInputs[0].binding = 0;
		vertexInputs[0].stride = sizeof(float) * 3;
		vertexInputs[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		vertexInputs[1].binding = 1;
		vertexInputs[1].stride = sizeof(float) * 2;
		vertexInputs[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		VkVertexInputAttributeDescription vertexAttributes[2]{};
		vertexAttributes[0].binding = 0; // must match binding above 
		vertexAttributes[0].location = 0; // must match shader 
		vertexAttributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		vertexAttributes[0].offset = 0;

		vertexAttributes[1].binding = 1; // must match binding above 
		vertexAttributes[1].location = 1; // must match shader 
		vertexAttributes[1].format = VK_FORMAT_R32G32_SFLOAT;
		vertexAttributes[1].offset = 0;

		VkPipelineVertexInputStateCreateInfo inputInfo{};
		inputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

		inputInfo.vertexBindingDescriptionCount = 2; // number of vertexInputs above 
		inputInfo.pVertexBindingDescriptions = vertexInputs;
		inputInfo.vertexAttributeDescriptionCount = 2; // number of vertexAttributes above 
		inputInfo.pVertexAttributeDescriptions = vertexAttributes;

		// Define which primitive (point, line, triangle, ...) the input is 
		// assembled into for rasterization. 
		VkPipelineInputAssemblyStateCreateInfo assemblyInfo{};
		assemblyInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		assemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		assemblyInfo.primitiveRestartEnable = VK_FALSE;

		// Define viewport and scissor regions 
		VkViewport viewport{};
		viewport.x = 0.f;
		viewport.y = 0.f;
		viewport.width = float(aWindow.swapchainExtent.width);
		viewport.height = float(aWindow.swapchainExtent.height);
		viewport.minDepth = 0.f;
		viewport.maxDepth = 1.f;

		VkRect2D scissor{};
		scissor.offset = VkOffset2D{ 0, 0 };
		scissor.extent = VkExtent2D{ aWindow.swapchainExtent.width, aWindow.swapchainExtent.height };

		VkPipelineViewportStateCreateInfo viewportInfo{};
		viewportInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportInfo.viewportCount = 1;
		viewportInfo.pViewports = &viewport;
		viewportInfo.scissorCount = 1;
		viewportInfo.pScissors = &scissor;

		// Define rasterization options 
		VkPipelineRasterizationStateCreateInfo rasterInfo{};
		rasterInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterInfo.depthClampEnable = VK_FALSE;
		rasterInfo.rasterizerDiscardEnable = VK_FALSE;
		rasterInfo.polygonMode = VK_POLYGON_MODE_FILL;
		rasterInfo.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterInfo.depthBiasEnable = VK_FALSE;
		rasterInfo.lineWidth = 1.f; // required

		// Define multisampling state 
		VkPipelineMultisampleStateCreateInfo samplingInfo{};
		samplingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		samplingInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		// Define blend state 
		// We define one blend state per color attachment - this example uses a 
		// single color attachment, so we only need one. Right now, we donft do any 
		// blending, so we can ignore most of the members. 
		VkPipelineColorBlendAttachmentState blendStates[1]{};
		blendStates[0].blendEnable = VK_TRUE; // New! Used to be VK FALSE. 
		blendStates[0].colorBlendOp = VK_BLEND_OP_ADD; // New! 
		blendStates[0].srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA; // New! 
		blendStates[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
		blendStates[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

		VkPipelineColorBlendStateCreateInfo blendInfo{};
		blendInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		blendInfo.logicOpEnable = VK_FALSE;
		blendInfo.attachmentCount = 1;
		blendInfo.pAttachments = blendStates;

		// Create pipeline 
		VkGraphicsPipelineCreateInfo pipeInfo{};
		pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;

		pipeInfo.stageCount = 2; // vertex + fragment stages 
		pipeInfo.pStages = stages;

		pipeInfo.pVertexInputState = &inputInfo;
		pipeInfo.pInputAssemblyState = &assemblyInfo;
		pipeInfo.pTessellationState = nullptr; // no tessellation 
		pipeInfo.pViewportState = &viewportInfo;
		pipeInfo.pRasterizationState = &rasterInfo;
		pipeInfo.pMultisampleState = &samplingInfo;
		pipeInfo.pDepthStencilState = &depthInfo; // no depth or stencil buffers 
		pipeInfo.pColorBlendState = &blendInfo;
		pipeInfo.pDynamicState = nullptr; // no dynamic states

		pipeInfo.layout = aPipelineLayout;
		pipeInfo.renderPass = aRenderPass;
		pipeInfo.subpass = 0; // first subpass of aRenderPass 

		VkPipeline pipe = VK_NULL_HANDLE;
		if (auto const res = vkCreateGraphicsPipelines(aWindow.device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &pipe); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create graphics pipeline\n" "vkCreateGraphicsPipelines() returned %s", lut::to_string(res).c_str());

		}

		return lut::Pipeline(aWindow.device, pipe);
	}

	void create_swapchain_framebuffers( lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, std::vector<lut::Framebuffer>& aFramebuffers, VkImageView aDepthView)
	{
		assert( aFramebuffers.empty() );

		for (std::size_t i = 0; i < aWindow.swapViews.size(); ++i)
		{
			VkImageView attachments[2] = { 
				aWindow.swapViews[i], 
				aDepthView
			};

			VkFramebufferCreateInfo fbInfo{};
			fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			fbInfo.flags = 0; // normal framebuffer 
			fbInfo.renderPass = aRenderPass;
			fbInfo.attachmentCount = 2;
			fbInfo.pAttachments = attachments;
			fbInfo.width = aWindow.swapchainExtent.width;
			fbInfo.height = aWindow.swapchainExtent.height;
			fbInfo.layers = 1;

			VkFramebuffer fb = VK_NULL_HANDLE;
			if (auto const res = vkCreateFramebuffer(aWindow.device, &fbInfo, nullptr, &fb); VK_SUCCESS != res)
			{
				throw lut::Error("Unable to create framebuffer for swap chain image %zu\n" "vkCreateFramebuffer() returned %s", i, lut::to_string(res).c_str());

			}

			aFramebuffers.emplace_back(lut::Framebuffer(aWindow.device, fb));
		}

		assert( aWindow.swapViews.size() == aFramebuffers.size() );
	}

	lut::DescriptorSetLayout create_scene_descriptor_layout( lut::VulkanWindow const& aWindow )
	{
		VkDescriptorSetLayoutBinding bindings[1]{}; 
		bindings[0].binding = 0; // number must match the index of the corresponding 
		// binding = N declaration in the shader(s)! 
		bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; 
		bindings[0].descriptorCount = 1; 
		bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = sizeof(bindings) / sizeof(bindings[0]);
		layoutInfo.pBindings = bindings; 
			
		VkDescriptorSetLayout layout = VK_NULL_HANDLE; 
		if(auto const res = vkCreateDescriptorSetLayout(aWindow.device, &layoutInfo, nullptr, &layout); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create descriptor set layout\n" "vkCreateDescriptorSetLayout() returned %s", lut::to_string(res).c_str()); 
				
		} 
			
		return lut::DescriptorSetLayout(aWindow.device, layout);
	}
	lut::DescriptorSetLayout create_object_descriptor_layout( lut::VulkanWindow const& aWindow )
	{
		VkDescriptorSetLayoutBinding bindings[1]{}; 
		bindings[0].binding = 0; // this must match the shaders 
		bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; 
		bindings[0].descriptorCount = 1; 
		bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT; 
			
		VkDescriptorSetLayoutCreateInfo layoutInfo{}; 
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO; 
		layoutInfo.bindingCount = sizeof(bindings) / sizeof(bindings[0]); 
		layoutInfo.pBindings = bindings; 
			
		VkDescriptorSetLayout layout = VK_NULL_HANDLE; 
		if(auto const res = vkCreateDescriptorSetLayout(aWindow.device, &layoutInfo,  nullptr, &layout); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create descriptor set layout\n" "vkCreateDescriptorSetLayout() returned %s", lut::to_string(res).c_str());
				
		} 
			
		return lut::DescriptorSetLayout(aWindow.device, layout);
	}

	void record_commands( VkCommandBuffer aCmdBuff, VkRenderPass aRenderPass, VkFramebuffer aFramebuffer, VkPipeline aGraphicsPipe, VkExtent2D const& aImageExtent, VkBuffer aPositionBuffer, VkBuffer aTexCoordBuffer, std::uint32_t aVertexCount, VkBuffer aSceneUBO, glsl::SceneUniform const& aSceneUniform, VkPipelineLayout aGraphicsLayout, VkDescriptorSet aSceneDescriptors, VkDescriptorSet aObjectDescriptors, VkBuffer aSpritePosBuffer,
		VkBuffer aSpriteTexBuffer, std::uint32_t aSpriteVertexCount, VkDescriptorSet aSpriteObjDescriptors, VkPipeline aAlphaPipeline)
	{
		// Begin recording commands
		VkCommandBufferBeginInfo begInfo{};
		begInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		begInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		begInfo.pInheritanceInfo = nullptr;

		if (auto const res = vkBeginCommandBuffer(aCmdBuff, &begInfo); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to begin recording command buffer\n" "vkBeginCommandBuffer() returned %s", lut::to_string(res).c_str());
		}

		lut::buffer_barrier(aCmdBuff, 
			aSceneUBO,
			VK_ACCESS_UNIFORM_READ_BIT, 
			VK_ACCESS_TRANSFER_WRITE_BIT, 
			VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, 
			VK_PIPELINE_STAGE_TRANSFER_BIT 
		); 
			
		vkCmdUpdateBuffer(aCmdBuff, aSceneUBO, 0, sizeof(glsl::SceneUniform), &aSceneUniform);
		
		lut::buffer_barrier(aCmdBuff, 
			aSceneUBO, 
			VK_ACCESS_TRANSFER_WRITE_BIT, 
			VK_ACCESS_UNIFORM_READ_BIT, 
			VK_PIPELINE_STAGE_TRANSFER_BIT, 
			VK_PIPELINE_STAGE_VERTEX_SHADER_BIT 
		);

		// Begin render pass 
		VkClearValue clearValues[2]{};
		clearValues[0].color.float32[0] = 0.1f; // Clear to a dark gray background. 
		clearValues[0].color.float32[1] = 0.1f; // If we were debugging, this would potentially 
		clearValues[0].color.float32[2] = 0.1f; // help us see whether the render pass took 
		clearValues[0].color.float32[3] = 1.f; // place, even if nothing else was drawn. 

		clearValues[1].depthStencil.depth = 1.f;

		VkRenderPassBeginInfo passInfo{};
		passInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		passInfo.renderPass = aRenderPass;
		passInfo.framebuffer = aFramebuffer;
		passInfo.renderArea.offset = VkOffset2D{ 0, 0 };
		passInfo.renderArea.extent = aImageExtent;
		passInfo.clearValueCount = 2;
		passInfo.pClearValues = clearValues;

		vkCmdBeginRenderPass(aCmdBuff, &passInfo, VK_SUBPASS_CONTENTS_INLINE);

		// Begin drawing with our graphics pipeline 
		vkCmdBindPipeline(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aGraphicsPipe);
		vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aGraphicsLayout, 0, 1, &aSceneDescriptors, 0, nullptr);
		vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aGraphicsLayout, 1, 1, &aObjectDescriptors, 0, nullptr);

		// Bind vertex input 
		VkBuffer buffers[2] = { aPositionBuffer, aTexCoordBuffer };
		VkDeviceSize offsets[2]{}; 
			
		vkCmdBindVertexBuffers(aCmdBuff, 0, 2, buffers, offsets); 
			
		// Draw vertices 
		vkCmdDraw(aCmdBuff, aVertexCount, 1, 0, 0);

		vkCmdBindPipeline(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aAlphaPipeline);

		vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aGraphicsLayout, 1, 1, &aSpriteObjDescriptors, 0, nullptr);
		
		VkBuffer spriteBuffers[2] = { aSpritePosBuffer, aSpriteTexBuffer }; 
			
		
		VkDeviceSize spriteOffsets[2]{}; 
			
		vkCmdBindVertexBuffers(aCmdBuff, 0, 2, spriteBuffers, spriteOffsets); 
			
		vkCmdDraw(aCmdBuff, aSpriteVertexCount, 1, 0, 0);

		// End the render pass 
		vkCmdEndRenderPass(aCmdBuff);

		// End command recording 
		if (auto const res = vkEndCommandBuffer(aCmdBuff); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to end recording command buffer\n" "vkEndCommandBuffer() returned %s", lut::to_string(res).c_str());
		}

	}

	void submit_commands( lut::VulkanWindow const& aWindow, VkCommandBuffer aCmdBuff, VkFence aFence, VkSemaphore aWaitSemaphore, VkSemaphore aSignalSemaphore )
	{
		VkPipelineStageFlags waitPipelineStages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &aCmdBuff;

		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = &aWaitSemaphore;
		submitInfo.pWaitDstStageMask = &waitPipelineStages;

		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = &aSignalSemaphore;

		if (auto const res = vkQueueSubmit(aWindow.graphicsQueue, 1, &submitInfo, aFence); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to submit command buffer to queue\n" "vkQueueSubmit() returned %s", lut::to_string(res).c_str());
		}
	}

	void present_results( VkQueue aPresentQueue, VkSwapchainKHR aSwapchain, std::uint32_t aImageIndex, VkSemaphore aRenderFinished, bool& aNeedToRecreateSwapchain )
	{
		throw lut::Error( "Not yet implemented" ); //TODO: (Section 1/Exercise 3) implement me!
	}
}

namespace
{
	std::tuple<lut::Image, lut::ImageView> create_depth_buffer( lut::VulkanWindow const& aWindow, lut::Allocator const& aAllocator )
	{
		VkImageCreateInfo imageInfo{}; 
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO; 
		imageInfo.imageType = VK_IMAGE_TYPE_2D; 
		imageInfo.format = cfg::kDepthFormat; 
		imageInfo.extent.width = aWindow.swapchainExtent.width; 
		imageInfo.extent.height = aWindow.swapchainExtent.height; 
		imageInfo.extent.depth = 1; 
		imageInfo.mipLevels = 1; 
		imageInfo.arrayLayers = 1; 
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT; 
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL; 
		imageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT; 
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE; 
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED; 
			 
		VmaAllocationCreateInfo allocInfo{}; 
		allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY; 
			
		VkImage image = VK_NULL_HANDLE; 
		VmaAllocation allocation = VK_NULL_HANDLE;

		if (auto const res = vmaCreateImage(aAllocator.allocator, &imageInfo, &allocInfo, &image, &allocation, nullptr); VK_SUCCESS != res)
		{ 
			throw lut::Error("Unable to allocate depth buffer image.\n" "vmaCreateImage() returned %s", lut::to_string(res).c_str()); 
		} 
			
		lut::Image depthImage(aAllocator.allocator, image, allocation); 
			
		// Create the image view 
		VkImageViewCreateInfo viewInfo{}; 
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO; 
		viewInfo.image = depthImage.image; 
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D; 
		viewInfo.format = cfg::kDepthFormat; 
		viewInfo.components = VkComponentMapping{}; 
		viewInfo.subresourceRange = VkImageSubresourceRange{ 
		VK_IMAGE_ASPECT_DEPTH_BIT, 
			0, 1, 
			0, 1 
		}; 
			
		VkImageView view = VK_NULL_HANDLE; 
		if(auto const res = vkCreateImageView(aWindow.device, &viewInfo, nullptr, &view); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create image view\n" "vkCreateImageView() returned %s", lut::to_string(res).c_str()); 
				
		} 
			
		return{ std::move(depthImage), lut::ImageView(aWindow.device, view) };
	}
}

//EOF vim:syntax=cpp:foldmethod=marker:ts=4:noexpandtab: 
