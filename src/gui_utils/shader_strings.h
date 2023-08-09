/*
 * SPDX-FileCopyrightText: Copyright (c) 2021, 2022 Università degli Studi di Milano. All rights reserved.
 *                         Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

static const char *solid_v = R"(
#version 410 core

uniform mat4 modelMatrix;
uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;

uniform vec3 lightDir;

in vec3 position;
in vec3 normal;
in vec4 color;

out vec3 vNormal;
out vec3 vSurfacePos;
out vec4 vColor;

void main(void)
{
    gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4(position, 1.0f);
    vNormal = (viewMatrix * modelMatrix * vec4(normal, 0.0f)).xyz;
    vSurfacePos = (viewMatrix * modelMatrix * vec4(position, 1.0f)).xyz;
    vColor = color;
}
)";

static const char* solid_v_d = R"(
#version 410 core

uniform mat4 modelMatrix;
uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;

uniform vec3 lightDir;

uniform float displacementScale;

in vec3 position;
in vec3 normal;
in vec3 displacement;
in vec4 color;

out vec3 vNormal;
out vec3 vSurfacePos;
out vec4 vColor;

void main(void)
{
    vec4 pos = vec4(position + displacementScale * displacement, 1.0f);
    gl_Position = projectionMatrix * viewMatrix * modelMatrix * pos;
    vNormal = (viewMatrix * modelMatrix * vec4(normal, 0.0f)).xyz;
    vSurfacePos = (viewMatrix * modelMatrix * pos).xyz;
    vColor = color;
}
)";

static const char *solid_f = R"(
#version 410 core

uniform vec3 cameraPos;

uniform float shadingWeight;
uniform int flatShading;

uniform float metallic;
uniform float roughness;
uniform float ao;

uniform int writeId;

uniform vec3 lightDir;

uniform vec3 meshColor;
uniform bool useColorAttribute;

uniform vec3 lightColor;
uniform float lightIntensity;

in vec3 vNormal;
in vec3 vSurfacePos;
in vec4 vColor;

layout(location = 0) out vec4 color;
layout(location = 1) out int id;


const float PI = 3.14159265359;
  
float DistributionGGX(vec3 N, vec3 H, float roughness);
float GeometrySchlickGGX(float NdotV, float roughness);
float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness);
vec3 fresnelSchlick(float cosTheta, vec3 F0);

void main()
{
    // surface normal
    vec3 t1 = dFdx(vSurfacePos);
    vec3 t2 = dFdy(vSurfacePos);
    vec3 N = normalize(vNormal);
    if (flatShading > 0) {
        N = normalize(cross(t1, t2));
    }

    // view direction
    vec3 V = normalize( - vSurfacePos ); 

    vec3 albedo = useColorAttribute ? vColor.rgb : meshColor;

    vec3 F0 = vec3(0.04); 
    F0 = mix(F0, albedo, metallic);

    vec3 lightDirs[2];
    lightDirs[0] = vec3(0.0f, 0.0f, 1.0f);
    lightDirs[1] = vec3(0.0f, 1.0f, 0.0f);

    float intensities[2];
    intensities[0] = lightIntensity;
    intensities[1] = lightIntensity / 2.0f;
	           
    // reflectance equation
    vec3 Lo = vec3(0.0);
    for(int i = 0; i < 2; ++i) 
    {
        // calculate per-light radiance
        vec3 L = normalize(lightDirs[i]);
        vec3 H = normalize(V + L);
        vec3 radiance     = lightColor * intensities[i];        
        
        // cook-torrance brdf
        float NDF = DistributionGGX(N, H, roughness);        
        float G   = GeometrySmith(N, V, L, roughness);      
        vec3 F    = fresnelSchlick(max(dot(H, V), 0.0), F0);       
        
        vec3 kS = F;
        vec3 kD = vec3(1.0) - kS;
        kD *= 1.0 - metallic;	  
        
        vec3 numerator    = NDF * G * F;
        float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
        vec3 specular     = numerator / denominator;  
            
        // add to outgoing radiance Lo
        float NdotL = max(dot(N, L), 0.0);                
        Lo += (kD * albedo / PI + specular) * radiance * NdotL; 
    }   
  
    vec3 ambient = albedo * ao * lightIntensity * 0.1;
    vec3 fcolor = ambient + Lo;

    fcolor = fcolor / (fcolor + vec3(1.0));

    fcolor = mix(albedo, fcolor, shadingWeight);

    if (writeId > 0)
        id = gl_PrimitiveID;

    color = vec4(fcolor, 1.0);
}

float DistributionGGX(vec3 N, vec3 H, float roughness)
{
    float a      = roughness*roughness;
    float a2     = a*a;
    float NdotH  = max(dot(N, H), 0.0);
    float NdotH2 = NdotH*NdotH;

    float num   = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return num / denom;
}

float GeometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r*r) / 8.0;

    float num   = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return num / denom;
}
float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2  = GeometrySchlickGGX(NdotV, roughness);
    float ggx1  = GeometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}

vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}
)";

static const char* wire_v = R"(
#version 410 core

uniform mat4 modelMatrix;
uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;

in vec3 position;
in vec3 normal;

out vec3 vNormal;

void main(void)
{
    gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4(position, 1.0f);
    vNormal = (modelMatrix * vec4(normal, 0.0f)).xyz;
}
)";

static const char* wire_f = R"(
#version 410 core

uniform vec4 color1;
uniform bool shading = false;

in vec3 vNormal;

layout(location = 0) out vec4 color;
layout(location = 1) out int id;

void main(void)
{
    if (!shading)
        color = color1;
    else
        color = vec4(vNormal, 1);
}
)";

static const char* quad_v = R"(
#version 410 core

in vec2 position;
in vec2 texcoord;

out vec2 uv; 

void main(void)
{
    gl_Position = vec4(position, 0.5f, 1.0f);
    uv = texcoord;
}
)";

static const char* quad_f = R"(
#version 410 core

uniform sampler2D offscreen;

in vec2 uv;
out vec4 color;

void main(void)
{
    color = vec4(texture(offscreen, uv).rgb, 1.0f);
}
)";

