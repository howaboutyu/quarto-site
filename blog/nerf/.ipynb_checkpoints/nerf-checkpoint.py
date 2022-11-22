import numpy as np
data = np.load('tiny_nerf_data.npz')
images = data['images']
poses = data['poses']
focal = data['focal']

print(f'Images size : {images.shape}')
print(f'Pose : {poses[0]}')
print(f'Focal length: {focal}')


import jax
import jax.numpy as jnp

def get_ray(H, W, focal, pose):
    x, y = jnp.mgrid[0:W, 0:H]
    x = x - W/2
    y = y - H/2
    y = -y # bender seems to use -y 

    x = x.flatten()
    y = y.flatten()

    direction = jnp.stack([x, y, -jnp.ones_like(x)])
    # Normalize direction
    direction_norm = jnp.linalg.norm(direction, ord=2, axis=0)
    direction = direction/direction_norm

    rot = pose[:3, :3] 
    direction = jnp.matmul(rot, direction)

    translation = pose[:3, 3]
    translation = translation[..., jnp.newaxis]
    origin = jnp.broadcast_to(translation, direction.shape)
    return origin, direction 


def encoding_func(x, L):
    encoded_array = []
    for i in range(L):
        encoded_array.extend([jnp.sin(jnp.power(2, i) * jnp.pi * x), jnp.cos(jnp.power(2,i) * jnp.pi * x)])
    return jnp.array(encoded_array)


from jax import random
def get_params(m, n, key, scale=1e-2):
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

def get_all_params(sizes, key):
    keys = random.split(key, len(sizes))
    param_array = []
    for i in range(len(sizes) -1 ):
        if i !=4:
            m = sizes[i]
            n = sizes[i + 1]
        else:
            # Special case when we are adding a new input
            m = 316
            n= 256
        param_array.append(get_params(m, n, key))
    return param_array  

sizes = [60] + [256]*3 +[256]+[256]*3 +[128] +[3]
params = get_all_params(sizes, jax.random.PRNGKey(0))
params.append(get_params(256, 1, jax.random.PRNGKey(0)))


def predict(params, input):
    x = input
    for i, (w, b) in enumerate(params[:-2]):
        
        x = jnp.dot(w, x) + b
        x = jnp.maximum(0, x)

        
        if i == 3:
            print(i)
            x = jnp.concatenate([x, input])
        
        if i==5:
            print(params[-1][0].shape)
            density = x
            density = jnp.dot(params[-1][0], x) + params[-1][1]
            density = jnp.maximum(0, density)
              
    w_f, b_f = params[-2]
    
    rgb = jnp.dot(w_f, x) + b_f
    rgb = jnp.maximum(0, rgb)
    density = jnp.maximum(0, density)
    return rgb, density


def sum_one_ray(t_delta, rgb_ray, density_ray):
        jax.debug.print('starting sum one ray')
        #jax.debug.print('density_ray shape',(density_ray).shape)
        print(',',(density_ray*t_delta[..., jnp.newaxis]).shape)
        T_i = jnp.exp(-jnp.cumsum(density_ray*t_delta[..., jnp.newaxis], 1))
   
        c = jnp.sum(T_i*(1-jnp.exp(-density_ray*t_delta[..., jnp.newaxis])) * rgb_ray,0)
        
        return c
    
    
total_ray_sum_func = jax.vmap(sum_one_ray, (None, 0, 0))
    
def render( params, poses, near=2., far=6., num_samples=4):
    jax.debug.print('starting render')
    origins, directions = get_ray_batched(poses)

    t = jnp.linspace(near, far, num_samples)
   
    r_func = lambda tt: origins+ directions * tt
    
    r = jax.vmap(r_func, (0))(t)
    
    
    r = jnp.reshape(r, [-1, 3])
    r = encoding_func_batched(r, 10)
    r = jnp.reshape(r, [-1, 60])
    
    rgb, density = model_func(params, r)
    rgb = jnp.reshape(rgb, [-1, 64, 3])
    density = jnp.reshape(density, [-1, 64, 1])
    
    t_delta = t[..., 1:] - t[...,:-1]
    t_delta = jnp.concatenate([t_delta, jnp.array([1e10])])
    
    c_array = total_ray_sum_func(t_delta, rgb, density )
    return c_array


images_flatten = jnp.reshape(images, [-1, 100*100, 3])

image_train = images_flatten[0]

pose = poses[0]
step_size = 0.001
def loss(params, pose, image_train):
    image_pred = render( params,  jnp.expand_dims(pose, 0), 2., 6., 64)
    return jnp.mean(jnp.square(image_pred - image_train))
jax.grad(loss)(params, pose, image_train):w
