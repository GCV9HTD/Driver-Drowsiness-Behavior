import os
import shutil

for root, dirs, files in os.walk("XLA2"):
    for i, filename in enumerate(files):
        filepath = os.path.join(root, filename)
        content = filepath.split('/')
        newname = "%04d"%int(content[1]) + "%04d"%int(i+1) + ".jpg"
        dirname = os.path.join(content[0], "%04d"%int(content[1]))
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        shutil.move(filepath, os.path.join(dirname, newname))
