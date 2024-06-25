import os

# Set the directory containing your files
directory = 'Test_images/Slight under focus'

# Get a list of all tiff files
files = [f for f in os.listdir(directory) if f.endswith('.tiff')]
files.sort()  # Sort the files to maintain any existing order, if necessary

# Rename files sequentially
for index, filename in enumerate(files):
    new_filename = f"{index:04d}.tiff"  # Generates '0000.tiff', '0001.tiff', etc.
    old_file = os.path.join(directory, filename)
    new_file = os.path.join(directory, new_filename)

    os.rename(old_file, new_file)
    print(f"Renamed '{filename}' to '{new_filename}'")

print("Renaming complete.")