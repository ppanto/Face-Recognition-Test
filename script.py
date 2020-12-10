from imutils import paths
import face_recognition
import pickle
import cv2
import os
import time

# Detection and recognition settings
scaleFactor = 1.2
minSizeTuple = (50, 50)
tolerance = 0.50 # Lower is more strict
minNeighbour = 6

print('[INFO] Entering training model phase.')

startTime = time.time()

imagePaths = list(paths.list_images("images"))
knownEncodings = []
knownNames = []

totalTrainingCases = 0
totalFacesFoundInTraining = 0
imgWithNoFaceFound = []

for (i, imagePath) in enumerate(imagePaths):
    # only images with 01 are for training
    if "02" in imagePath:
        continue
    totalTrainingCases+=1
    
    # extract the person name from the image path
    name = imagePath.split(os.path.sep)[-2]
    
    print("[INFO] Processing image {}/{} for \"{}\"".format(i + 1,
        len(imagePaths), name))
    

    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # find coordinates of bounding box of each face
    boxes = face_recognition.face_locations(rgb, model="hog")

    # compute the facial embeddings
    encodings = face_recognition.face_encodings(rgb, boxes)

    if not encodings:
        imgWithNoFaceFound.append(name)
    for encoding in encodings:
        totalFacesFoundInTraining+=1
        knownEncodings.append(encoding)
        knownNames.append(name)

print('[INFO] Saving model for future uses.')
data = {"encodings": knownEncodings, "names": knownNames}
with open("encodings.pickle", "wb") as f:
    f.write(pickle.dumps(data))
    f.close()

endTime = time.time()
timeForTraining = endTime - startTime

print('[INFO] Training model finished in {:.2f} seconds.'.format(round(timeForTraining,2)))

# ----------

print('[INFO] Starting recognition phase.')
startTime = time.time()

# will use Haar's instead of HOG now
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

currentTestCase = 0
facesFoundAndCorrectlyIdentified = 0
facesFoundButIncorrectlyIdentified = 0
facesFoundButIdentifiedAsUnknown = 0
totalFacesFoundInTesting = 0

for (i, imagePath) in enumerate(imagePaths):
    # only images with 02 are for testing
    if "01" in imagePath:
        continue
    # extract the person name from the image path
    name = imagePath.split(os.path.sep)[-2]
    # skip if face not found during training
    notFoundInTraining = False
    for notFounds in imgWithNoFaceFound:
        if notFounds == name:
            notFoundInTraining = True
    if notFoundInTraining:
        continue
        
    currentTestCase+=1

    print("[INFO] Recognition case {}/{} for \"{}\"".format(currentTestCase,
        totalTrainingCases, name))
    
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # find coordinates of bounding box of each face
    rects = detector.detectMultiScale(gray,
                                          scaleFactor=scaleFactor,
                                          minNeighbors=minNeighbour,
                                          minSize=minSizeTuple,
                                          flags=cv2.CASCADE_SCALE_IMAGE)
    
    # box coordinates are in (x, y, w, h) order
    boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]
    
    # get encodings
    encodings = face_recognition.face_encodings(rgb, boxes)
    
    names = []
    
    # try to match encoding to a name
    for encoding in encodings:
        totalFacesFoundInTesting+=1
        matches = face_recognition.compare_faces(data['encodings'],
                                                     encoding,
                                                     tolerance=tolerance)
        matchedName = "Unknown"
        
        if True in matches:
            matchedIds = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            for i in matchedIds:
                matchedName = data['names'][i]
                counts[matchedName] = counts.get(matchedName, 0) + 1

            matchedName = max(counts, key=counts.get)
        names.append(matchedName)
        
    # test against labeled name and other scenarios
    for matchedName in names:
        if matchedName == name:
            facesFoundAndCorrectlyIdentified+=1
        elif matchedName == "Unknown":
            facesFoundButIdentifiedAsUnknown+=1
        else:
            facesFoundButIncorrectlyIdentified+=1

endTime = time.time()
timeForTesting = endTime - startTime
print("[INFO] Recognition phase finished in {:.2f} seconds.".format(timeForTesting))
print("")
print("[INFO] Total duration of script is {:.2f} seconds".format(timeForTraining + timeForTesting))
print("")
print("[INFO] Test results in training phase:")
print("Number of total images scanned -> {}".format(totalTrainingCases))
print("Number of faces found -> {}".format(totalFacesFoundInTraining))
print("Average number of faces per image -> {:.2f}".format(round(totalFacesFoundInTraining/totalTrainingCases,2)))

print("")
print("[INFO] Test results in recognition phase:")
print("Number of total images scanned -> {}".format(currentTestCase))
print("Number of faces found -> {}".format(totalFacesFoundInTesting))
print("Average number of faces per image -> {:.2f}".format(round(totalFacesFoundInTesting/currentTestCase,2)))
print("Correctly labeled -> {}".format(facesFoundAndCorrectlyIdentified))
print("Correctly labeled in percent against faces found -> {:.2f}%".format(round((100*facesFoundAndCorrectlyIdentified)/
                                                                      totalFacesFoundInTesting,2)))
print("Incorrectly labeled (not counting unknowns) -> {}".format(facesFoundButIncorrectlyIdentified))
print("Incorrectly labeled in percent against faces found -> {:.2f}%".format(round((100*facesFoundButIncorrectlyIdentified)/
                                                                      totalFacesFoundInTesting,2)))
print("Labeled as unknowns -> {}".format(facesFoundButIdentifiedAsUnknown))
print("Labeled as unknowns in percent against faces found -> {:.2f}%".format(round((100*facesFoundButIdentifiedAsUnknown)/
                                                                      totalFacesFoundInTesting,2)))



