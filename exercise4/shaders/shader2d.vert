#version 450 

layout( location = 0 ) in vec2 iPosition; 
layout( location = 1 ) in vec3 iColor; 

layout( location = 0 ) out vec3 v2fColor; 

void main() 
{ 
	v2fColor = iColor; 
	gl_Position = vec4( iPosition, 0.5f, 1.f ); 
} 