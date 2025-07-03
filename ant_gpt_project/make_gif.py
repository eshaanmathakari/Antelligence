from PIL import Image
import glob

# Collect image files and sort by step number
png_files = glob.glob("ant_simulation_step_*.png")
png_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

# Open images
images = [Image.open(f) for f in png_files]

# Save as GIF
images[0].save(
    'ant_simulation_animation.gif',
    save_all=True,
    append_images=images[1:],
    duration=500,   # milliseconds per frame
    loop=0
)

print("GIF saved as ant_simulation_animation.gif")
