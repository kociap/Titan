#include "cinematic_camera.hpp"

#include <glfw/glfw3.h>
#include <glm/gtc/matrix_transform.hpp>

#include <cmath>

namespace titan {

void update_cinematic_camera(OrbitCamera& camera, float delta_time) {
    // delta_time paramter is not used in this function because we already rely on the time anyway.
    // Maybe we will remove this parameter if it turns out we don't need it for any of the cinematic cameras

    float time = glfwGetTime();
    camera.position.x = camera.target.x + std::sin(time * camera.rotation_speed) * camera.distance_to_target.x;
    camera.position.y = camera.target.y + camera.distance_to_target.y;
    camera.position.z = camera.target.z + std::cos(time * camera.rotation_speed) * camera.distance_to_target.z;
}

void update_cinematic_camera(FollowCamera& camera, float delta_time) {
    camera.position = camera.target + camera.distance_to_target;
}

static glm::vec3 get_up_vector(glm::vec3 position, glm::vec3 target, glm::vec3 world_up) {
    // calculate up vector for camera
    glm::vec3 direction = glm::normalize(position - target);
    glm::vec3 cam_right = glm::normalize(glm::cross(world_up, direction));
    return glm::normalize(glm::cross(direction, cam_right));
}

glm::mat4 get_view_matrix(OrbitCamera& camera, glm::vec3 world_up) {
    glm::vec3 cam_up = get_up_vector(camera.position, camera.target, world_up);
    
    // create view matrix
    return glm::lookAt(camera.position, camera.target, cam_up);
}

glm::mat4 get_view_matrix(FollowCamera& camera, glm::vec3 world_up) {
    // Identical to OrbitCamera

    glm::vec3 cam_up = get_up_vector(camera.position, camera.target, world_up);

    // create view matrix
    return glm::lookAt(camera.position, camera.target, cam_up);
}

glm::mat4 get_view_matrix(StationaryCamera& camera, glm::vec3 world_up) {
    glm::vec3 position = camera.target + camera.distance_to_target;
    glm::vec3 cam_up = get_up_vector(position, camera.target, world_up);

    return glm::lookAt(position, camera.target, cam_up);
}

}