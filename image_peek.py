import carla
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Connect to CARLA
client = carla.Client('localhost', 2000)
world = client.get_world()

# Create BEV camera sensor (top-down view)
camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '800')
camera_bp.set_attribute('image_size_y', '800')
camera_bp.set_attribute('fov', '90')  # Wide FOV for BEV

# Get vehicle or use spectator position
try:
    vehicle = world.get_actors().filter('vehicle.*')[0]
    carla_vehicles = world.get_actors().filter('vehicle.*')
    spawn_point = carla.Transform(
        carla.Location(x=0, y=0, z=50),  # 50m above vehicle
        carla.Rotation(pitch=-90)  # Look straight down
    )
    camera = world.spawn_actor(camera_bp, spawn_point, attach_to=vehicle)
except:
    # Use fixed position if no vehicle
    spawn_point = carla.Transform(
        carla.Location(x=0, y=0, z=50),
        carla.Rotation(pitch=-90)
    )
    camera = world.spawn_actor(camera_bp, spawn_point)

# Setup matplotlib
fig, ax = plt.subplots()
img_plot = ax.imshow(np.zeros((800, 800, 3)))
ax.axis('off')
plt.title('CARLA Bird\'s Eye View')

latest_frame = None

def image_callback(image):
    global latest_frame
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    latest_frame = array[:, :, :3]

def update_plot(frame):
    if latest_frame is not None:
        img_plot.set_array(latest_frame)
    return [img_plot]

camera.listen(image_callback)
ani = animation.FuncAnimation(fig, update_plot, interval=50, blit=True)
plt.show()

camera.destroy()