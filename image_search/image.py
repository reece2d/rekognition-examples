import boto3, pprint
from PIL import Image, ImageDraw, ImageFont

client = boto3.client("rekognition")

file   = open("image.png", "rb")
binary = file.read()
image  = Image.open(file)
draw   = ImageDraw.Draw(image)

faces = client.detect_faces(Image={ "Bytes": binary }, Attributes=["ALL"])
font  = ImageFont.truetype("roboto.ttf", 20)

for index, face in enumerate(faces["FaceDetails"]):

    box = face["BoundingBox"]

    x1 = image.size[0] * box["Left"]
    x2 = x1 + image.size[0] * box["Width"]

    y1 = image.size[1] * box["Top"]
    y2 = y1 + image.size[1] * box["Height"]

    draw.rectangle([x1, y1, x2, y2])

    x2 -= (x2 - x1)

    emotion_type = ""
    confidence = 0
    for emotion in face["Emotions"]:
        if (emotion["Confidence"] > confidence):
            emotion_type = emotion["Type"]
            confidence = emotion["Confidence"]

    emotion = "Emotion: " + emotion_type
    gender  = "Gender: " + str(face["Gender"]["Value"])
    bright  = "Brightness Rating: {:.2f}%".format(face["Quality"]["Brightness"])
    sharp   = "Sharpness Rating: {:.2f}%".format(face["Quality"]["Sharpness"])

    draw.text([x2, y2], emotion, (255, 0, 0, 255), font)
    draw.text([x2, y2 + 20], gender, (255, 0, 0, 255), font)
    draw.text([x2, y2 + 40], bright, (255, 0, 0, 255), font)
    draw.text([x2, y2 + 60], sharp, (255, 0, 0, 255), font)

    pprint.pprint(face)

image.show()
file.close()