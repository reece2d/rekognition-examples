import cv2, boto3, time, pprint

client = boto3.client("rekognition")
video = {
    "S3Object": {
        "Bucket": "rekognition-testing-bucket",
        "Name": "test.mp4"
    }}
job = client.start_face_detection(Video=video, FaceAttributes="ALL")["JobId"]

print("Waiting for response from AWS")
response = client.get_face_detection(JobId=job)
while (response["JobStatus"] != "SUCCEEDED"):
    time.sleep(3)
    response = client.get_face_detection(JobId=job)
print("Recieved tracking information")

capture = cv2.VideoCapture("test.mp4")

pos1 = (0, 0)
pos2 = (500, 500)
eyepos1 = (100, 100)
eyepos2 = (0, 0)
emotion = ""

while True:
    ret, frame = capture.read()
    if (ret == False):
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = capture.read()

    current_time = (1000 / 30) * capture.get(cv2.CAP_PROP_POS_FRAMES)
    for face in response["Faces"]:
        if (current_time - 90 <= face["Timestamp"] <= current_time + 90):
            x1 = face["Face"]["BoundingBox"]["Left"] * 720
            x2 = x1 + face["Face"]["BoundingBox"]["Width"] * 720
            y1 = face["Face"]["BoundingBox"]["Top"] * 1280
            y2 = y1 + face["Face"]["BoundingBox"]["Height"] * 1280

            pos1 = (int(x1), int(y1))
            pos2 = (int(x2), int(y2))

            for landmark in face["Face"]["Landmarks"]:
                if (landmark["Type"] == "eyeLeft"):
                    eyepos1 = (int(landmark["X"] * 720), int(landmark["Y"] * 1280))
                elif (landmark["Type"] == "eyeRight"):
                    eyepos2 = (int(landmark["X"] * 720), int(landmark["Y"] * 1280))

            emotion = face["Face"]["Emotions"][0]["Type"]

    cv2.rectangle(frame, pos1, pos2, (0, 255, 0), 5)
    cv2.circle(frame, eyepos1, 10, (0, 0, 255, -1))
    cv2.circle(frame, eyepos2, 10, (0, 0, 255, -1))

    pos2 = (pos1[0], pos2[1] + 30)
    cv2.putText(frame, emotion, pos2, cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_AA)
    cv2.imshow("frame", frame)

    if cv2.waitKey(33)== 27:
        break

cv2.destroyAllWindows()