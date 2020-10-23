import os
import requests
import bz2, shutil

def download_dlib_model():
    print_orderly("Get dlib model", 60)
    dlib_model_link = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    print("Downloading dlib model...")
    with requests.get(dlib_model_link, stream=True) as r:
        print("Zip file size: ", np.round(len(r.content) / 1024 / 1024, 2), "MB")
        destination = (
            "dlib_models" + os.path.sep + "shape_predictor_68_face_landmarks.dat.bz2"
        )
        if not os.path.exists(destination.rsplit(os.path.sep, 1)[0]):
            os.mkdir(destination.rsplit(os.path.sep, 1)[0])
        print("Saving dlib model...")
        with open(destination, "wb") as fd:
            for chunk in r.iter_content(chunk_size=32678):
                fd.write(chunk)
    print("Extracting dlib model...")
    with bz2.BZ2File(destination) as fr, open(
        "dlib_models/shape_predictor_68_face_landmarks.dat", "wb"
    ) as fw:
        shutil.copyfileobj(fr, fw)
    print("Saved: ", destination)
    print_orderly("done", 60)

    os.remove(destination)
