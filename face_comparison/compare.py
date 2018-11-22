import boto3, pprint
from PIL import Image, ImageDraw, ImageFont

client = boto3.client("rekognition")

source = open("source.jpg", "rb")
target = open("target.jpg", "rb")

source_binary = source.read()
target_binary = target.read()

source_image = Image.open(source)
target_image = Image.open(target)

source_draw = ImageDraw.Draw(source_image)
target_draw = ImageDraw.Draw(target_image)

response = client.compare_faces(
    SourceImage={
        "Bytes": source_binary
    },
    TargetImage={
        "Bytes": target_binary
    },
    SimilarityThreshold=0
)

def draw_source_face(response, image, surface, font):
    box = response["SourceImageFace"]["BoundingBox"]

    x1 = image.size[0] * box["Left"]
    x2 = x1 + image.size[0] * box["Width"]

    y1 = image.size[1] * box["Top"]
    y2 = y1 + image.size[1] * box["Height"]

    surface.rectangle([x1, y1, x2, y2])

def draw_target_matches(response, image, surface, font):
    for face in response["FaceMatches"]:

        box = face["Face"]["BoundingBox"]

        x1 = image.size[0] * box["Left"]
        x2 = x1 + image.size[0] * box["Width"]

        y1 = image.size[1] * box["Top"]
        y2 = y1 + image.size[1] * box["Height"]

        surface.rectangle([x1, y1, x2, y2])

        x2 -= (x2 - x1)
        similarity = "Similarity: {:.2f}".format(face["Similarity"])
        surface.text([x2, y2], similarity, (255, 0, 0, 255), font)


roboto = ImageFont.truetype("roboto.ttf", 20)

draw_source_face(response, source_image, source_draw, roboto)
draw_target_matches(response, target_image, target_draw, roboto)

source_image.show()
target_image.show()

source.close()
target.close()