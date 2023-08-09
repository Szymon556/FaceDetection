#detector.py
import argparse
from pathlib import Path
import face_recognition
import pickle
from collections import Counter
from PIL import Image,ImageDraw

DEFAULT_ENCODING_PATH = Path("output/encodings.pkl")
BOUNDING_BOX_COLOR = "blue"
TEXT_COLOR = "white"
Path("training").mkdir(exist_ok = True)
Path("output").mkdir(exist_ok = True) 
Path("validation").mkdir(exist_ok = True) 


parser = argparse.ArgumentParser(description="Recognize faces in a image")
parser.add_argument("--train",action="store_true",help="Train on iput argument")
parser.add_argument("--validate",action="store_true",help="Validate trained model")
parser.add_argument("--test",action="store_true",help="Test the model")
parser.add_argument("-m",
action="store",
default="hog",
choices=["hog","cnn"],
help="wchich model to use for training: hog (CPU) cnn (GPU)")
parser.add_argument(
    "-f",action="store",help="Path to an image with unknow face"
)

args = parser.parse_args()

def encode_known_faces(
        model:str = "hog",encoding_location : Path = DEFAULT_ENCODING_PATH

)->None:
    names = []
    encodings = []
# Get the pictures and names and asociate it in dictionary
    for filepath in Path("training").glob("*/*"):
        name = filepath.parent.name
        image = face_recognition.load_image_file(filepath)

        face_locations = face_recognition.face_locations(image, model=model)
        face_encodings = face_recognition.face_encodings(image,face_locations)

        for encoding in face_encodings:
            names.append(name)
            encodings.append(encoding)
    name_encodings = {"names":names, "encodings":encodings}
#save dictionary with names and ascociated face location to file 
    with encoding_location.open(mode="wb") as f:
        pickle.dump(name_encodings,f)

def recognize_faces(
        image_location: str,
        model: str = "hog",
        encodings_location: Path = DEFAULT_ENCODING_PATH,
) -> None:
# load file with faces locations
    with encodings_location.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)
#get the encoding face location of unnkow picture
    input_image = face_recognition.load_image_file(image_location)
    input_face_locations = face_recognition.face_locations(input_image, model=model)
    input_face_encodings = face_recognition.face_encodings(input_image,input_face_locations)



    pillow_image = Image.fromarray(input_image)
    draw = ImageDraw.Draw(pillow_image)
    ''' Get the encodings of new face/faces location/s and compare with 
     locations form training data also use bounding box as a place to draw rectangle '''
    for bounding_box,unknow_encoding in zip(
        input_face_locations,input_face_encodings
    ):
        name = _recognize_face(unknow_encoding,loaded_encodings)
        if not name:
            name = "Unknow"
        #print(name, bounding_box)
        _display_face(draw,bounding_box,name)
    del draw
    pillow_image.show()



def _recognize_face(unknow_encoding,loaded_encodings):
    boolean_matches = face_recognition.compare_faces(
        loaded_encodings["encodings"],unknow_encoding 
    
    )
   
    votes = Counter(
        name 
        for match,name in  zip(boolean_matches,loaded_encodings["names"])
        if match
    )
    if votes:
        return votes.most_common(1)[0][0]
    

    
def _display_face(draw,bounding_box,name):
    top,right,bottom,left = bounding_box
    draw.rectangle(((left, top), (right, bottom)), outline=BOUNDING_BOX_COLOR)
    text_left, text_top, text_right, text_bottom = draw.textbbox(
        (left, bottom), name
    )
  
    draw.rectangle(
        ((text_left,text_top),(text_right,text_bottom)),
        fill = "blue",
        outline = "blue",

    )
    draw.text(
        (text_left,text_top),
        name,
        fill = "white",

    )
def validate(model: str = "hog"):
    for filepath in Path("validation").rglob("*"):
        if filepath.is_file():
            recognize_faces(
                image_location=str(filepath.absolute()),model=model
            )
if __name__ == "__main__":
    if args.test:
        recognize_faces(image_location=args.f,model=args.m)
    if args.validate:
        validate(model=args.m)
    if args.train:
        encode_known_faces(model=args.m)
